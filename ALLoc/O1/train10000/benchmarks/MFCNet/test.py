import os
import numpy as np
import math
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from model import *
from thop import profile

# Windows系统GPU设置
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

# CPU线程设置
cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

# 超参数
batch_size = 1
hidden_size = 32
car_size = 32
ant_size = 32

# Windows路径格式
path = r'E:\forALLoc\ALLoc\Data\ALLoc_data_share\32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12\train10000'

# Windows下DataLoader设置（num_workers=0）
train_dataset = DatasetFolder_mapping(os.path.join(path, 'train'))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True
)

test_dataset = DatasetFolder_mapping(os.path.join(path, 'test'))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True
)

# 模型初始化与加载
model = LocNet(num_ant=ant_size, num_car=car_size, hidden_size=hidden_size)

# 设备配置
if torch.cuda.is_available() and len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cpu()
elif torch.cuda.is_available():
    model = model.cpu()
else:
    model = model.cpu()

# 加载模型（Windows路径兼容）
model.load_state_dict(
    torch.load('./model.pth', map_location=torch.device('cpu') if not torch.cuda.is_available() else None),
    strict=False)


# 定义RMSE计算函数
def RMSE(pred, target):
    return torch.sqrt(torch.mean(torch.square(pred - target)))


# 测试评估
model.eval()
rmse_distribution_list = []
sum_rmse = 0

with torch.no_grad():
    for i, (data, local) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # 数据预处理和设备迁移
        data = data * 10000
        if torch.cuda.is_available():
            data = data.cpu().float()
            local = local.cpu().float()
        else:
            data = data.cpu().float()
            local = local.cpu().float()

        local_current = local[:, 0:2]
        local_pred = model(data)
        local_pred = local_pred[:, -1, :]

        rmse = RMSE(local_pred, local_current)
        rmse_distribution_list.append(rmse.item())
        sum_rmse += rmse.item()

avg_rmse = sum_rmse / (i + 1)
print(f"avg_distance : {avg_rmse}")

# 计算不同分位数的RMSE
rmse_distribution_list.sort()
total_len = len(rmse_distribution_list)

for rate in [0.2, 0.1]:
    print(f"rate: {rate}")
    low_len = int(rate * total_len)
    print(f"low : {rmse_distribution_list[low_len]}")
    high_len = int((1 - rate) * total_len)
    print(f"high : {rmse_distribution_list[high_len]}")

# 保存结果（Windows路径兼容）
np.save("rmse_distribution.npy", np.array(rmse_distribution_list))