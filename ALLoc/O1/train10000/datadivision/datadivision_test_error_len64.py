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
os.makedirs(path+'/sequence_test_len64',exist_ok=True)

np.random.seed(1)
random.seed(1)
test_num=2000

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


pos_array=np.load(path+'/train'+'/pos_array.npy')
path_now=path+'/train'
for i in tqdm(range(test_num)):
    test_data = np.load(path + '/test' + '/data_' + str(i) + '.npy')
    test_local = np.load(path + '/test' + '/data_local_' + str(i) + '.npy')
    data_sequence, pos_sequence = find_sequence_test(num=64, data=test_data, pos=test_local, pos_array=pos_array,
                                                     path_now=path + '/train')
    np.save(path+'/sequence_test_len64'+'/data_'+ str(i) + '.npy',data_sequence)
    np.save(path + '/sequence_test_len64' + '/data_local_' + str(i) + '.npy', pos_sequence)
np.save(path+'/sequence_test_len64'+'/data_len.npy',test_num)