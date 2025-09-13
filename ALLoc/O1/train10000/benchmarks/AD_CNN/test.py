import os
import numpy as np
import math
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from model import *
from thop import profile

# Windows系统多线程设置
cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

# GPU设置
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

# 参数设置
batch_size = 1
hidden_size = 32
car_size = 32
ant_size = 32


def DFT_matrix_tensor(num_ant, num_car):
    """生成DFT矩阵"""
    F_delay = torch.zeros([num_car, num_car], dtype=torch.cfloat)
    F_angle = torch.zeros([num_ant, num_ant], dtype=torch.cfloat)

    for i in range(num_car):
        for j in range(num_car):
            F_delay[i, j] = torch.tensor(1 / np.sqrt(num_car) * np.exp(-1j * 2 * np.pi / num_car * i * j))

    for i in range(num_ant):
        for j in range(num_ant):
            F_angle[i, j] = torch.tensor(
                1 / np.sqrt(num_ant) * np.exp(-1j * 2 * np.pi / num_ant * i * (j - num_ant / 2)))

    return F_delay, F_angle


def DFT_tensor(num_ant, num_car, data, F_delay, F_angle):
    """DFT变换处理"""
    num = data.shape[0]
    data = data.reshape([num, num_ant, num_car, 2])
    data_real = data[:, :, :, 0]
    data_imag = data[:, :, :, 1]
    data_complex = data_real + 1j * data_imag

    F_delay_batch = (F_delay.unsqueeze(0)).repeat(num, 1, 1)
    F_angle_batch = (F_angle.unsqueeze(0)).repeat(num, 1, 1)
    data_DFTmatrix_complex = torch.bmm(
        torch.bmm(F_angle_batch, data_complex),
        (F_delay_batch.real - 1j * F_delay_batch.imag).transpose(1, 2)
    )

    data_DFT = torch.zeros([num, num_ant, num_car, 2])
    data_DFT[:, :, :, 0] = data_DFTmatrix_complex.real
    data_DFT[:, :, :, 1] = data_DFTmatrix_complex.imag
    return data_DFT


def DFT_tensor_reverse(num_ant, num_car, data_reverse, F_delay, F_angle):
    """逆DFT变换处理"""
    num = data_reverse.shape[0]
    data_reverse = data_reverse.reshape([num, num_ant, num_car, 2])
    data_reverse_real = data_reverse[:, :, :, 0]
    data_reverse_imag = data_reverse[:, :, :, 1]

    F_delay_inv = torch.linalg.inv(F_delay)
    F_angle_inv = torch.linalg.inv(F_angle)
    F_delay_inv_batch = (F_delay_inv.unsqueeze(0)).repeat(num, 1, 1)
    F_angle_inv_batch = (F_angle_inv.unsqueeze(0)).repeat(num, 1, 1)

    data_reverse_complex = data_reverse_real + 1j * data_reverse_imag
    data_complex = torch.bmm(
        torch.bmm(F_angle_inv_batch, data_reverse_complex),
        ((F_delay_inv_batch.real - 1j * F_delay_inv_batch.imag).transpose(1, 2))
    )

    data = torch.zeros([num, num_ant, num_car, 2])
    data[:, :, :, 0] = data_complex.real
    data[:, :, :, 1] = data_complex.imag
    return data


# 生成DFT矩阵
F_delay, F_angle = DFT_matrix_tensor(num_ant=ant_size, num_car=car_size)

# Windows风格文件路径（使用反斜杠）
path = r'E:\forALLoc\ALLoc\Data\ALLoc_data_share\32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12\train10000'

# 加载数据集（Windows下num_workers设为0避免多进程问题）
train_dataset = DatasetFolder_mapping(os.path.join(path, 'train'))
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Windows下多线程易出问题，设为0
    pin_memory=True,
    drop_last=True
)

test_dataset = DatasetFolder_mapping(os.path.join(path, 'test'))
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Windows下多线程易出问题，设为0
    pin_memory=True,
    drop_last=True
)

# 模型初始化与加载
model = LocNet(num_ant=ant_size, num_car=car_size, hidden_size=hidden_size)

# 设备设置
if len(gpu_list.split(',')) > 1 and torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

# 加载模型（Windows路径适配）
model.load_state_dict(torch.load('./model.pth', map_location=device), strict=False)
model.eval()

# 评估指标计算
rmse_distribution_list = []
sum_rmse = 0

with torch.no_grad():  # 统一管理梯度关闭
    for i, (data, local) in tqdm(enumerate(test_loader), total=len(test_loader)):
        data = data * 10000
        data = DFT_tensor(
            num_ant=ant_size,
            num_car=car_size,
            data=data.float(),
            F_delay=F_delay.to(device),  # 确保DFT矩阵在正确设备上
            F_angle=F_angle.to(device)
        )
        data = data.to(device)
        local = local.to(device).float()
        local_current = local[:, 0:2]

        local_pred = model(data)
        rmse = RMSE(local_pred, local_current)
        rmse_val = rmse.item()
        rmse_distribution_list.append(rmse_val)
        sum_rmse += rmse_val

# 计算并打印结果
avg_rmse = sum_rmse / (i + 1)
print(f"avg_distance: {avg_rmse}")

rmse_distribution_list.sort()
total_len = len(rmse_distribution_list)

# 计算不同比例的RMSE分布
for rate in [0.2, 0.1]:
    print(f"rate: {rate}")
    low_len = int(rate * total_len)
    print(f"low: {rmse_distribution_list[low_len]}")
    high_len = int((1 - rate) * total_len)
    print(f"high: {rmse_distribution_list[high_len]}")

# 保存结果（Windows路径兼容）
rmse_distribution_array = np.array(rmse_distribution_list)
np.save("rmse_distribution.npy", rmse_distribution_array)