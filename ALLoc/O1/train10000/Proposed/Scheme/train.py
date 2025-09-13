import os
import numpy as np
import math
import torch
import torch.nn as nn
import random
from model import *
from thop import profile
import time
from ptflops import get_model_complexity_info
from torchstat import stat
from einops import rearrange, repeat
from tqdm import tqdm
import pandas as pd

def main():
    cpu_num = 12
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    gpu_list = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

    np.random.seed(1)
    random.seed(1)

    batch_size = 500
    epochs = 10000
    steps = 300000
    learning_rate = 1e-3
    print_freq = 200

    ant_size = 32
    car_size = 32
    hidden_size = 32

    depth = 8

    model = Nlocnet(ant_size=ant_size, car_size=car_size, embed_dim=hidden_size, num_heads=4, hidden_dim=hidden_size, depth=depth)
    from calflops import calculate_flops
    input1 = torch.randn(1, 64, ant_size, car_size, 2)
    input2 = torch.randn(1, 64, 2)
    input3 = torch.randn(1, 1, ant_size, car_size, 2)
    inputs = {
        "neigh_channel": input1,
        "neigh_pos": input2,
        "current_channel": input3
    }
    flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=True)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

    if len(gpu_list.split(',')) > 1:
        model = torch.nn.DataParallel(model).cpu()
    else:
        model = model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    path = r'E:/forALLoc/ALLoc/Data/ALLoc_data_share/32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12/train10000'
    train_dataset = DatasetFolder(path + '/sequence')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    test_dataset = DatasetFolder(path + '/sequence_test_len64')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

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
                lr = self.initial_lr * (0.2 **1)
            elif 200000 < self.steps <= 250000:
                lr = self.initial_lr * (0.2** 2)
            elif 250000 < self.steps <= 300000:
                lr = self.initial_lr * (0.2 ** 3)
            cur_lr = lr
            for p in self.optimizer.param_groups:
                p['lr'] = cur_lr
            return cur_lr

    lr_scheduler = LrSchedule(initial_lr=learning_rate, optimizer=optimizer)

    def find_sequence_test(num, data, pos, pos_array, path_now, error_range=1):
        error = np.zeros_like(pos)
        error[0:2] = np.random.uniform(-1 * error_range, error_range, size=2)
        pos = pos + error
        current_pos = repeat(pos, 'l -> train_num l', train_num=pos_array.shape[0])
        distance_array = np.linalg.norm(current_pos - pos_array, axis=1)
        indices = (np.argpartition(distance_array, num)[:num]).tolist()
        neighbor_list = []
        pos_neighbor_list = []
        for j in indices:
            data_neigh = np.load(path_now + '/data_' + str(j) + '.npy')
            neighbor_list.append(data_neigh)
            pos_neighbor = np.load(path_now + '/data_local_' + str(j) + '.npy')
            pos_neighbor_list.append(pos_neighbor)
        neighbor_list.append(data)
        pos_neighbor_list.append(pos)
        data_sequence = np.array(neighbor_list)
        pos_sequence = np.array(pos_neighbor_list)
        return data_sequence, pos_sequence

    test_num = 2000
    total_neighbor_num = 64
    least_neighbor_num = 16
    whole_index = [a for a in range(least_neighbor_num, total_neighbor_num + 1)]
    found = False

    # 初始化列表用于保存数据
    log_data = []

    for epoch in range(epochs):
        for i, (data, local) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            data = data.cpu().float()
            data = data * 10000
            local = local.cpu().float()
            local = local[:, :, 0:2]

            index_neigh = random.sample([b for b in range(total_neighbor_num + 1)], random.choice(whole_index))
            index_current = random.sample([b for b in range(total_neighbor_num + 1)],
                                          random.choice([a for a in range(1, total_neighbor_num + 1)]))
            data_neigh = data[:, index_neigh, ...]
            data_current = data[:, index_current, ...]
            local_neigh = local[:, index_neigh, ...]
            local_current = local[:, index_current, ...]

            local_pred = model(data_neigh, local_neigh, data_current)
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
            current_lr = optimizer.param_groups[0]['lr']
            current_step = lr_scheduler.steps
            print(f'lr:{current_lr:.4e}')
            print(f"epoch : {epoch}")
            print(f"step : {current_step}")
            torch.save(model.state_dict(), './model.pth')

            model.eval()
            sum_rmse = 0
            count = 0
            for i, (data, local) in enumerate(test_loader):
                data = data.cpu().float()
                data = data * 10000
                local = local.cpu().float()
                local = local[:, :, 0:2]
                data_neigh = data[:, 0:-1, ...]
                data_current = data[:, -1:, ...]
                local_neigh = local[:, 0:-1, ...]
                local_current = local[:, -1:, ...]
                with torch.no_grad():
                    local_pred = model(data_neigh, local_neigh, data_current)
                rmse = torch.sqrt(nn.MSELoss()(local_pred[:, -1, :], local_current[:, -1, :]))
                sum_rmse += rmse
                count += 1
            if count > 0:
                avg_rmse = sum_rmse / count
                print(f"avg_distance : {avg_rmse}")
                # 保存当前epoch的记录
                log_data.append({
                    'epoch': epoch,
                    'learning_rate': current_lr,
                    'step': current_step,
                    'avg_distance': avg_rmse.item()  # 转换为Python数值
                })
            else:
                print("No test data available")

    torch.save(model.state_dict(), './model.pth')

    # 将记录转换为DataFrame并保存为Excel
    df = pd.DataFrame(log_data)
    df.to_excel('training_log_Proposed_Scheme.xlsx', index=False)  # 保存为Excel文件，不包含索引
    print("训练日志已保存为 training_log_Proposed_Scheme.xlsx")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()