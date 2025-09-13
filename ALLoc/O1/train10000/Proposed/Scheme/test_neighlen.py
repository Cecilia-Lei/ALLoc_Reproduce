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

cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num )

gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

np.random.seed(1)
random.seed(1)

batch_size=500
ant_size=32
car_size=32
hidden_size=256

depth=8

model=Nlocnet(ant_size=ant_size,car_size=car_size,embed_dim=hidden_size,num_heads=4,hidden_dim=hidden_size,depth=depth)

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()
model.load_state_dict(torch.load('./model' +'.pth'),strict=False)

path=r'E:/forALLoc/ALLoc/Data/ALLoc_data_share/32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12/train10000'
train_dataset = DatasetFolder(path+'/sequence')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

test_dataset=DatasetFolder(path+'/sequence_test_len64')
test_loader=torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

model.eval()
neighlen_list=[1,2,4,8,16,32,64]
rmse_list=[]

for neighlen in neighlen_list:
    sum_rmse = 0
    for i, (data, local) in enumerate(test_loader):
        data = data.cuda().float()
        data = data * 10000
        local = local.cuda().float()
        local = local[:, :, 0:2]
        data_neigh = data[:, 0:neighlen, ...]
        data_current = data[:, -1:, ...]
        local_neigh = local[:, 0:neighlen, ...]
        local_current = local[:, -1:, ...]
        with torch.no_grad():
            local_pred = model(data_neigh, local_neigh, data_current)
        rmse = RMSE(local_pred[:,-1,:], local_current[:,-1,:])
        rmse = rmse.item()
        sum_rmse+=rmse
    avg_rmse=sum_rmse/(i+1)
    rmse_list.append(avg_rmse)
print("rmse_list :",rmse_list)