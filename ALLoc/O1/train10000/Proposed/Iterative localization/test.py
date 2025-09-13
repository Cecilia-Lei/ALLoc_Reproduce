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

np.random.seed(1)
random.seed(1)

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

path_model_notneigh='/e/forALLoc/ALLoc/O1/train10000/Proposed/Scheme_notneigh/'
model.load_state_dict(torch.load(path_model_notneigh+'./model' +'.pth'),strict=False)

path='/e/forALLoc/ALLoc/Data/ALLoc_data_share/32ant*32car_3.5GHz_40MHz_R501-1400_V1_24.4.12/train10000'

test_dataset=DatasetFolder(path+'/sequence_test_len64_notneigh')
test_loader=torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True, drop_last=True)

model.eval()
pos_notneigh_list=[]
sum_rmse = 0
for i, (data, local) in tqdm(enumerate(test_loader),total=len(test_loader)):
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
    rmse = RMSE(local_pred[:,-1,:], local_current[:,-1,:])
    rmse = rmse.item()
    local_pred_numpy=(local_pred[:,-1,:]).cpu().numpy()
    pos_notneigh_list.append(local_pred_numpy)
    sum_rmse+=rmse
avg_rmse=sum_rmse/(i+1)
print("avg_distance :", avg_rmse)


path_model='/e/forALLoc/ALLoc/O1/train10000/Proposed/Scheme/'
model.load_state_dict(torch.load(path_model+'./model' +'.pth'),strict=False)

def find_sequence_test_noisedata(num,data,noise_pos,real_pos,pos_array,path_now):

    pos_error=noise_pos
    pos_array=pos_array[:,0:2]
    current_pos=repeat(pos_error,'l -> train_num l',train_num=pos_array.shape[0])
    distance_array=np.linalg.norm(current_pos-pos_array,axis=1)
    indices = (np.argpartition(distance_array, num)[:num]).tolist()
    neighbor_list=[]
    pos_neighbor_list=[]
    for j in indices:
        data_neigh=np.load(path_now+'/data_'+str(j)+'.npy')
        neighbor_list.append(data_neigh)
        pos_neighbor=np.load(path_now+'/data_local_'+str(j)+'.npy')
        pos_neighbor_list.append(pos_neighbor[0:2])
    neighbor_list.append(data)
    pos_neighbor_list.append(real_pos)
    data_sequence=np.array(neighbor_list)
    pos_sequence=np.array(pos_neighbor_list)
    return data_sequence,pos_sequence

pos_array = np.load(path + '/train' + '/pos_array.npy')

model.eval()
sum_rmse=0
test_num=20000
rmse_distribution_list=[]
for i in tqdm(range(test_num)):
    test_data = np.load(path + '/test' + '/data_' + str(i) + '.npy')
    test_local = pos_notneigh_list[i].squeeze(0)
    real_local=(np.load(path + '/test' + '/data_local_' + str(i) + '.npy'))[0:2]
    data_sequence, pos_sequence = find_sequence_test_noisedata(num=64,data=test_data, noise_pos=test_local,real_pos=real_local, pos_array=pos_array,
                                                     path_now=path + '/train')
    data=torch.tensor(data_sequence).unsqueeze(0)
    local=torch.tensor(pos_sequence).unsqueeze(0)
    data = data.cuda().float()
    data = data * 10000
    local = local.cuda().float()
    data_neigh = data[:, 0:-1, ...]
    data_current = data[:, -1:, ...]
    local_neigh = local[:, 0:-1, ...]
    local_current = local[:, -1:, ...]
    with torch.no_grad():
        local_pred = model(data_neigh, local_neigh, data_current)
    rmse = RMSE(local_pred[:, -1, :], local_current[:, -1, :])
    rmse = rmse.item()
    rmse_distribution_list.append(rmse)
    sum_rmse += rmse
avg_rmse = sum_rmse / (i + 1)
print("avg_rme:",avg_rmse)

rmse_distribution_list.sort()
total_len=len(rmse_distribution_list)
rate=0.2
print("rate:",rate)
low_len=int(rate*total_len)
print("low :",rmse_distribution_list[low_len])
high_len=int((1-rate)*total_len)
print("high :",rmse_distribution_list[high_len])
rate=0.1
print("rate:",rate)
low_len=int(rate*total_len)
print("low :",rmse_distribution_list[low_len])
high_len=int((1-rate)*total_len)
print("high :",rmse_distribution_list[high_len])

rmse_distribution_array=np.array(rmse_distribution_list)
np.save("rmse_distribution.npy",rmse_distribution_array)