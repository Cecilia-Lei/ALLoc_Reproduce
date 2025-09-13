import os
import numpy as np
import math
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from model import *
from thop import profile

# Windows下多线程设置
cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

# GPU设置
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

batch_size = 1
hidden_size = 32

car_size = 32
ant_size = 32

# Windows路径格式（使用反斜杠或双反斜杠）
path = r'E:\forALLoc\ALLoc\Data\ALLoc_data_share\32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12\train10000'
train_dataset = DatasetFolder_mapping(path + '\\train')  # 使用反斜杠拼接
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0,  # Windows下num_workers设为0避免多进程问题
    pin_memory=True, drop_last=True
)

test_dataset = DatasetFolder_mapping(path + '\\test')
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0,  # 同上
    pin_memory=True, drop_last=True
)

model = LocNet(num_ant=ant_size, num_car=car_size, hidden_size=hidden_size)

# 设备设置（自动判断是否使用GPU）
device = torch.device(f"cuda:{gpu_list}" if torch.cuda.is_available() else "cpu")
if len(gpu_list.split(',')) > 1 and torch.cuda.is_available():
    model = torch.nn.DataParallel(model).to(device)
else:
    model = model.to(device)

# 加载模型（Windows路径适配）
model.load_state_dict(torch.load('./model.pth', map_location=device), strict=False)

model.eval()
rmse_distribution_list = []
sum_rmse = 0

for i, (data, local) in tqdm(enumerate(test_loader), total=len(test_loader)):
    data = data.to(device).float()
    data = data * 10000
    local = local.to(device).float()
    local_current = local[:, 0:2]

    with torch.no_grad():
        local_pred = model(data)

    rmse = RMSE(local_pred, local_current)
    rmse = rmse.item()
    rmse_distribution_list.append(rmse)
    sum_rmse += rmse

avg_rmse = sum_rmse / (i + 1)
print("avg_distance :", avg_rmse)

rmse_distribution_list.sort()
total_len = len(rmse_distribution_list)

rate = 0.2
print("rate:", rate)
low_len = int(rate * total_len)
print("low :", rmse_distribution_list[low_len])
high_len = int((1 - rate) * total_len)
print("high :", rmse_distribution_list[high_len])

rate = 0.1
print("rate:", rate)
low_len = int(rate * total_len)
print("low :", rmse_distribution_list[low_len])
high_len = int((1 - rate) * total_len)
print("high :", rmse_distribution_list[high_len])

rmse_distribution_array = np.array(rmse_distribution_list)
np.save("rmse_distribution.npy", rmse_distribution_array)