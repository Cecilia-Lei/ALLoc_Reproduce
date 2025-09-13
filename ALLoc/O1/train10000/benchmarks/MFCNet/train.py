import os
import numpy as np
import math
import torch
import torch.nn as nn
import random
import pandas as pd
from openpyxl import Workbook
from model import *
from thop import profile

# Windows系统下设置GPU
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

# 设置CPU线程数
cpu_num = 12
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

# 超参数设置
batch_size = 500
epochs = 10000
steps = 300000
learning_rate = 1e-3
print_freq = 200
hidden_size = 32

car_size = 32
ant_size = 32

# Windows路径格式（使用反斜杠或双反斜杠）
path = r'E:\forALLoc\ALLoc\Data\ALLoc_data_share\32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12\train10000'

# Windows下DataLoader的num_workers设为0（避免多进程问题）
train_dataset = DatasetFolder_mapping(os.path.join(path, 'train'))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True  # num_workers=0
)

test_dataset = DatasetFolder_mapping(os.path.join(path, 'test'))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True  # num_workers=0
)

# 模型初始化
model = LocNet(num_ant=ant_size, num_car=car_size, hidden_size=hidden_size)

# 计算FLOPs和参数
from calflops import calculate_flops

input1 = torch.randn(1, ant_size, car_size, 2)
inputs = {"data": input1}
flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=True)
print(f"FLOPs:{flops}   MACs:{macs}   Params:{params}\n")

# 设备配置
if torch.cuda.is_available() and len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cpu()
elif torch.cuda.is_available():
    model = model.cpu()
else:
    model = model.cpu()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 再次验证FLOPs
input = torch.rand([1, ant_size, car_size, 2])
if torch.cuda.is_available():
    input = input.cpu()
flops, params = profile(model, inputs=(input,))
print(f"\nparams : {params}")


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
        else:
            lr = self.initial_lr * (0.2 ** 3)  # 防止超出范围

        cur_lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = cur_lr
        return cur_lr


lr_scheduler = LrSchedule(initial_lr=learning_rate, optimizer=optimizer)


# 定义RMSE计算函数
def RMSE(pred, target):
    return torch.sqrt(torch.mean(torch.square(pred - target)))


# 新增：用于记录训练数据的列表
log_data = []

found = False
for epoch in range(epochs):
    for i, (data, local) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        # 数据预处理和设备迁移
        data = data * 10000
        if torch.cuda.is_available():
            data = data.cpu().float()
            local = local.cpu().float()
        else:
            data = data.cpu().float()
            local = local.cpu().float()

        index = list(range(car_size))
        data = data[:, :, index, :]
        local = local[:, 0:2]

        local_pred = model(data)

        # 计算带权重的MSE损失
        train_loc = local.unsqueeze(1).repeat(1, len(index), 1)
        weight = torch.arange(len(index)) + 1
        sum_weight = torch.sum(weight)
        weight = weight / sum_weight * len(index)
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=2)
        if torch.cuda.is_available():
            weight = weight.cpu().float()
        else:
            weight = weight.cpu().float()
        weight = weight.repeat(local_pred.shape[0], 1, local_pred.shape[2])

        MSE_loss = nn.MSELoss()(local_pred * weight, train_loc * weight)
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
        current_lr = optimizer.param_groups[0]['lr']
        current_step = lr_scheduler.steps

        # 打印信息
        print(f'lr:%.4e' % current_lr)
        print(f"epoch : {epoch}")
        print(f"step : {current_step}")

        # 保存模型
        torch.save(model.state_dict(), './model.pth')

        # 测试评估
        model.eval()
        sum_rmse = 0
        with torch.no_grad():
            for i, (data, local) in enumerate(test_loader):
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
                sum_rmse += rmse

        avg_rmse = sum_rmse / (i + 1)
        print(f"avg_distance : {avg_rmse}")

        log_data.append({
            'epoch': epoch,
            'learning_rate': current_lr,
            'step': current_step,
            'avg_distance': avg_rmse.item()  # 转换为Python数值
        })

# 最终保存模型
torch.save(model.state_dict(), './model.pth')

# 将日志数据保存到Excel
df = pd.DataFrame(log_data)
# 按列顺序排序（确保顺序为epoch, learning_rate, step, avg_distance）
df = df[['epoch', 'learning_rate', 'step', 'avg_distance']]
df.to_excel('training_log_MFCNet.xlsx', index=False)
print("训练日志已保存到 training_log_MFCNet.xlsx")