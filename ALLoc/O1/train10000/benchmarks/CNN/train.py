import os
import numpy as np
import math
import torch
import torch.nn as nn
import random
import pandas as pd
from datetime import datetime
from model import *
from thop import profile

# Windows下设置CPU线程数
cpu_num = 12
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

# 设置GPU（Windows下格式相同）
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

batch_size = 500
epochs = 10000
steps = 300000
learning_rate = 1e-3
print_freq = 200
hidden_size = 32

car_size = 32
ant_size = 32

# Windows路径格式
path = r'E:\forALLoc\ALLoc\Data\ALLoc_data_share\32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12\train10000'
train_dataset = DatasetFolder_mapping(path + '\\train')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0,  # Windows下禁用多进程
    pin_memory=True, drop_last=True
)

test_dataset = DatasetFolder_mapping(path + '\\test')
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0,  # 同上
    pin_memory=True, drop_last=True
)

# 设备设置
device = torch.device(f"cuda:{gpu_list}" if torch.cuda.is_available() else "cpu")
model = LocNet(num_ant=ant_size, num_car=car_size, hidden_size=hidden_size).to(device)

# 计算FLOPs和参数
from calflops import calculate_flops

input1 = torch.randn(1, ant_size, car_size, 2).to(device)
inputs = {"data": input1}
flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=True)
print(f"FLOPs:{flops}   MACs:{macs}   Params:{params}\n")

# 多GPU支持
if len(gpu_list.split(',')) > 1 and torch.cuda.is_available():
    model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 再次计算FLOPs（适配设备）
input = torch.rand([1, ant_size, car_size, 2]).to(device)
flops, params = profile(model, inputs=(input,))
print("\nparams : ", params)


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
            lr = self.initial_lr * (0.2 ** 3)  # 超出步骤后的默认学习率

        cur_lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = cur_lr
        return cur_lr


lr_scheduler = LrSchedule(initial_lr=learning_rate, optimizer=optimizer)

# 初始化Excel日志数据结构（调整字段顺序和名称）
log_data = {
    'epoch': [],
    'learning_rate': [],
    'step': [],
    'avg_distance': [],
}

found = False
for epoch in range(epochs):
    model.train()  # 移到循环开头，确保训练模式
    for i, (data, local) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.to(device).float()
        data = data * 10000
        local = local.to(device).float()
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
        current_lr = optimizer.param_groups[0]["lr"]
        current_step = lr_scheduler.steps

        # 保存模型（Windows路径）
        torch.save(model.state_dict(), './model.pth')

        # 验证
        model.eval()
        sum_rmse = 0
        with torch.no_grad():  # 整个验证过程禁用梯度
            for i, (data, local) in enumerate(test_loader):
                data = data.to(device).float()
                data = data * 10000
                local = local.to(device).float()
                local_current = local[:, 0:2]

                local_pred = model(data)
                rmse = RMSE(local_pred, local_current)
                sum_rmse += rmse

        avg_rmse = sum_rmse / (i + 1)

        # 记录日志数据（同步修改字段名称）
        log_data['epoch'].append(epoch)
        log_data['learning_rate'].append(current_lr)
        log_data['step'].append(current_step)
        log_data['avg_distance'].append(avg_rmse.item())  # 转换为Python数值

        # 打印信息
        print(f'lr:{current_lr:.4e}')
        print(f"epoch : {epoch}")
        print(f"step : {current_step}")
        print(f"avg_distance : {avg_rmse}")

        # 保存为Excel
        log_df = pd.DataFrame(log_data)
        log_df.to_excel('./training_log_CNN.xlsx', index=False)

# 最终保存模型和日志
torch.save(model.state_dict(), './model.pth')
# 确保最后一次数据也被保存
log_df = pd.DataFrame(log_data)
log_df.to_excel('./training_log_CNN.xlsx', index=False)
print("训练完成，日志已保存至 training_log_CNN.xlsx")