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
test_num=20000

batch_size=1
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

path='/e/forALLoc/ALLoc/Data/ALLoc_data_share/32ant*32car_3.5GHz_40MHz_R501-1400_V1_24.4.12/train10000'

def find_sequence_test(num,data,pos,pos_array,path_now,error_range=1):

    error=np.zeros_like(pos)
    error[0:2]=np.random.uniform(-1 * error_range, error_range, size=2)
    pos_error=pos+error
    current_pos=repeat(pos_error,'l -> train_num l',train_num=pos_array.shape[0])
    distance_array=np.linalg.norm(current_pos-pos_array,axis=1)
    indices = (np.argpartition(distance_array, num)[:num]).tolist()
    neighbor_list=[]
    pos_neighbor_list=[]
    for j in indices:
        data_neigh=np.load(path_now+'/data_'+str(j)+'.npy')
        neighbor_list.append(data_neigh)
        pos_neighbor=np.load(path_now+'/data_local_'+str(j)+'.npy')
        pos_neighbor_list.append(pos_neighbor)
    neighbor_list.append(data)
    pos_neighbor_list.append(pos)
    data_sequence=np.array(neighbor_list)
    pos_sequence=np.array(pos_neighbor_list)
    return data_sequence,pos_sequence

error_range_list=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]
rmse_list=[]
pos_array = np.load(path + '/train' + '/pos_array.npy')
path_now = path + '/train'
for error_range in error_range_list:
    sum_rmse=0
    for i in tqdm(range(test_num)):
        test_data = np.load(path + '/test' + '/data_' + str(i) + '.npy')
        test_local = np.load(path + '/test' + '/data_local_' + str(i) + '.npy')
        data_sequence, pos_sequence = find_sequence_test(num=64, data=test_data, pos=test_local, pos_array=pos_array,
                                                         path_now=path + '/train',error_range=error_range)
        data=torch.tensor(data_sequence).unsqueeze(0)
        local=torch.tensor(pos_sequence).unsqueeze(0)
        data = data.cuda().float()
        data = data * 10000
        local = local.cuda().float()
        local = local[:, :, 0:2]
        data_neigh = data[:, 0:-1, ...]
        data_current = data[:, -1:, ...]
        local_neigh = local[:, 0:-1, ...]
        local_current = local[:, -1:, ...]
        with torch.no_grad():
            local_pred = model(data_neigh, local_neigh, data_current)
        rmse = RMSE(local_pred[:, -1, :], local_current[:, -1, :])
        rmse = rmse.item()
        sum_rmse += rmse
    avg_rmse = sum_rmse / (i + 1)
    rmse_list.append(avg_rmse)
print("rmse_list:",rmse_list)