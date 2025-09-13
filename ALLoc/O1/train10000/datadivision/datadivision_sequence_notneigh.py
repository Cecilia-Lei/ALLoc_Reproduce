import numpy as np
import scipy.io as scio
import hdf5storage
import random
import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
from tqdm import tqdm
from einops import rearrange,repeat
gpu_list = '0'


path=r'E:/forALLoc/ALLoc/Data/ALLoc_data_share/32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12/train10000'
os.makedirs(path+'/sequence_notneigh',exist_ok=True)

np.random.seed(1)
random.seed(1)
train_num=1000

def find_sequence(num,data,pos,pos_array,path_now):
    array_num = pos_array.shape[0]
    indices =random.sample(range(array_num), num)
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

num=64
pos_array=np.load(path+'/train'+'/pos_array.npy')
path_now=path+'/train'
for i in tqdm(range(train_num)):
    data=np.load(path + '/train' + '/data_' + str(i) + '.npy')
    pos_error=pos_array[i]
    data_sequence,pos_sequence=find_sequence(num=num,data=data,pos=pos_error,pos_array=pos_array,path_now=path_now)
    np.save(path+'/sequence_notneigh'+'/data_'+ str(i) + '.npy',data_sequence)
    np.save(path + '/sequence_notneigh' + '/data_local_' + str(i) + '.npy', pos_sequence)
np.save(path+'/sequence_notneigh'+'/data_len.npy',train_num)