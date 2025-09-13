import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
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
from einops import rearrange,repeat
from tqdm import tqdm

# 定义RMSE函数（假设之前遗漏了）
def RMSE(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2))

def main():
    cpu_num = 4
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num )

    np.random.seed(1)
    random.seed(1)

    batch_size=1
    ant_size=32
    car_size=32
    hidden_size=32

    depth=8

    model=Nlocnet(ant_size=ant_size,car_size=car_size,embed_dim=hidden_size,num_heads=4,hidden_dim=hidden_size,depth=depth)

    if len(gpu_list.split(',')) > 1:
        model = torch.nn.DataParallel(model).cpu()
    else:
        model = model.cpu()
    model.load_state_dict(torch.load('./model.pth'), strict=False)

    path=r'E:/forALLoc/ALLoc/Data/ALLoc_data_share/32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12/train10000'
    train_dataset = DatasetFolder(path+'/sequence')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=False, drop_last=True)

    test_dataset=DatasetFolder(path+'/sequence_test_len64')
    test_loader=torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=False, drop_last=True)

    model.eval()
    rmse_distribution_list=[]
    sum_rmse = 0
    for i, (data, local) in tqdm(enumerate(test_loader), total=len(test_loader)):
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
        rmse = RMSE(local_pred[:,-1,:], local_current[:,-1,:])
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

if __name__ == '__main__':
    # 在Windows系统中启用多进程支持
    import multiprocessing
    multiprocessing.freeze_support()
    main()