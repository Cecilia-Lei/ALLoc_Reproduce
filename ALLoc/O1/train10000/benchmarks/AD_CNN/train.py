import os
import numpy as np
import math
import torch
import torch.nn as nn
import random
import pandas as pd
from model import *
from thop import profile

# Windows下设置CPU线程数
cpu_num = 12
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

# 设置GPU
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

# 超参数设置
batch_size = 500
epochs = 10000
steps = 300000
learning_rate = 1e-3
print_freq = 200
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
    """对数据进行DFT变换"""
    num = data.shape[0]
    data = data.reshape([num, num_ant, num_car, 2])
    data_real = data[:, :, :, 0]
    data_imag = data[:, :, :, 1]
    data_complex = data_real + 1j * data_imag

    F_delay_batch = (F_delay.unsqueeze(0)).repeat(num, 1, 1)
    F_angle_batch = (F_angle.unsqueeze(0)).repeat(num, 1, 1)
    data_DFTmatrix_complex = torch.bmm(torch.bmm(F_angle_batch, data_complex),
                                       (F_delay_batch.real - 1j * F_delay_batch.imag).transpose(1, 2))
    data_DFTmatrix_real = data_DFTmatrix_complex.real
    data_DFTmatrix_imag = data_DFTmatrix_complex.imag
    data_DFT = torch.zeros([num, num_ant, num_car, 2])
    data_DFT[:, :, :, 0] = data_DFTmatrix_real
    data_DFT[:, :, :, 1] = data_DFTmatrix_imag
    return data_DFT


def DFT_tensor_reverse(num_ant, num_car, data_reverse, F_delay, F_angle):
    """DFT逆变换"""
    num = data_reverse.shape[0]
    data_reverse = data_reverse.reshape([num, num_ant, num_car, 2])
    data_reverse_real = data_reverse[:, :, :, 0]
    data_reverse_imag = data_reverse[:, :, :, 1]
    F_delay_inv = torch.linalg.inv(F_delay)
    F_angle_inv = torch.linalg.inv(F_angle)
    F_delay_inv_batch = (F_delay_inv.unsqueeze(0)).repeat(num, 1, 1)
    F_angle_inv_batch = (F_angle_inv.unsqueeze(0)).repeat(num, 1, 1)
    data_reverse_complex = data_reverse_real + 1j * data_reverse_imag

    data_complex = torch.bmm(torch.bmm(F_angle_inv_batch, data_reverse_complex),
                             ((F_delay_inv_batch.real - 1j * F_delay_inv_batch.imag).transpose(1, 2)))
    data_real = data_complex.real
    data_imag = data_complex.imag
    data = torch.zeros([num, num_ant, num_car, 2])
    data[:, :, :, 0] = data_real
    data[:, :, :, 1] = data_imag
    return data


# 初始化DFT矩阵
F_delay, F_angle = DFT_matrix_tensor(num_ant=ant_size, num_car=car_size)

# Windows下文件路径使用反斜杠或双反斜杠
path = r'E:\forALLoc\ALLoc\Data\ALLoc_data_share\32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12\train10000'

# 加载数据集 - Windows下num_workers设为0避免多进程问题
train_dataset = DatasetFolder_mapping(os.path.join(path, 'train'))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0,  # Windows下多进程容易出问题，设为0
    pin_memory=True,
    drop_last=True
)

test_dataset = DatasetFolder_mapping(os.path.join(path, 'test'))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0,  # 同上
    pin_memory=True,
    drop_last=True
)

# 初始化模型
model = LocNet(num_ant=ant_size, num_car=car_size, hidden_size=hidden_size)

# 计算模型复杂度
from calflops import calculate_flops

input1 = torch.randn(1, ant_size, car_size, 2)
inputs = {"data": input1}
flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=True)
print(f"FLOPs:{flops}   MACs:{macs}   Params:{params} \n")

# 模型放到CPU
if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cpu()
else:
    model = model.cpu()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 再次计算FLOPs
input = torch.rand([1, ant_size, car_size, 2]).cpu()
flops, params = profile(model, inputs=(input,))
print("")
print("params : ", params)


# 学习率调度器
class LrSchedule():
    def __init__(self, initial_lr, optimizer=None):
        super(LrSchedule, self).__init__()
        self.steps = 0
        self.initial_lr = initial_lr
        self.optimizer = optimizer

    def step(self):
        self.steps += 1
        if self.steps <= 150000:
            lr = self.initial_lr
        elif 150000 < self.steps <= 200000:
            lr = self.initial_lr * (0.2 ** 1)
        elif 200000 < self.steps <= 250000:
            lr = self.initial_lr * (0.2 ** 2)
        elif 250000 < self.steps <= 300000:
            lr = self.initial_lr * (0.2 ** 3)
        cur_lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = cur_lr
        return cur_lr


lr_scheduler = LrSchedule(initial_lr=learning_rate, optimizer=optimizer)

# 确保保存模型的目录存在
os.makedirs('./', exist_ok=True)  # 当前目录，确保可写

# 新增：初始化存储训练信息的列表
log_data = []
# 新增：定义Excel表头
columns = ['epoch', 'learning_rate', 'step', 'avg_rmse']

# 训练过程
found = False
for epoch in range(epochs):
    for i, (data, local) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        data = data * 10000
        data = DFT_tensor(num_ant=ant_size, num_car=car_size, data=data.float(),
                          F_delay=F_delay, F_angle=F_angle)
        data = data.cpu()
        local = local.cpu().float()
        local_current = local[:, 0:2]

        local_pred = model(data)
        MSE_loss = nn.MSELoss()(local_pred, local_current)
        loss = MSE_loss
        loss.backward()
        lr_scheduler.step()
        if lr_scheduler.steps <= steps:
            optimizer.step()
        else:
            found = True
            break
    if found:
        break

    if epoch % 20 == 0:
        # 获取当前训练信息
        current_lr = optimizer.param_groups[0]["lr"]
        current_step = lr_scheduler.steps

        # 打印信息
        print(f'lr:{current_lr:.4e}')
        print(f"epoch : {epoch}")
        print(f"step : {current_step}")

        # 保存模型，Windows下路径兼容
        torch.save(model.state_dict(), './model.pth')

        # 测试
        model.eval()
        sum_rmse = 0.0
        for i, (data, local) in enumerate(test_loader):
            optimizer.zero_grad()
            data = data * 10000
            data = DFT_tensor(num_ant=ant_size, num_car=car_size, data=data.float(),
                              F_delay=F_delay, F_angle=F_angle)
            data = data.cpu()
            local = local.cpu().float()
            local_current = local[:, 0:2]
            with torch.no_grad():
                local_pred = model(data)
            rmse = RMSE(local_pred, local_current)
            sum_rmse += rmse
        avg_rmse = sum_rmse / (i + 1)
        print(f"avg_distance : {avg_rmse}")

        # 新增：将当前轮次信息添加到列表
        log_data.append({
            'epoch': epoch,
            'learning_rate': current_lr,
            'step': current_step,
            'avg_distance': avg_rmse.item()  # 转换为Python数值
        })

# 将数据写入Excel
df = pd.DataFrame(log_data, columns=columns)
df.to_excel('training_log_AD_CNN.xlsx', index=False)
print("训练日志已保存到 training_log_AD_CNN.xlsx")

# 最后保存一次模型
torch.save(model.state_dict(), './model.pth')