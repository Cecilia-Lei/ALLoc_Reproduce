import numpy as np
import scipy.io as scio
import hdf5storage
import random
import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
from tqdm import tqdm
gpu_list = '0'


path=r'E:/forALLoc/ALLoc/Data/ALLoc_data_share/32ant_32car_3.5GHz_40MHz_R501-1400_V1_24.4.12/'

data_mat_local=hdf5storage.loadmat(path+'/mat'+'/DeepMIMO_local.mat')
data_local=data_mat_local['DeepMIMO_loc']
print(data_local.shape)

np.random.seed(1)
random.seed(1)

os.makedirs(path+'/train10000',exist_ok=True)
os.makedirs(path+'/train10000/train',exist_ok=True)
os.makedirs(path+'/train10000/test',exist_ok=True)

index=np.arange(181*900)
np.random.shuffle(index)
train_num=1000
test_num=2000


pos_array=data_local[index[:train_num]]
print(pos_array.shape)
np.save(path+'/train10000/train'+'/pos_array.npy',pos_array)
print("pos_array",pos_array)

for i in tqdm(range(train_num)):
    data_mat = hdf5storage.loadmat(path + '/mat' + '/DeepMIMO_dataset_' + str(index[i]) + '.mat')
    data = data_mat['DeepMIMO_dataset']
    np.save(path + '/train10000/train' + '/data_' + str(i) + '.npy', data)
    np.save(path + '/train10000/train' + '/data_local_' + str(i) + '.npy', data_local[index[i]])
np.save(path+'/train10000/train'+'/data_len.npy',train_num)

for i in tqdm(range(test_num)):
    data_mat = hdf5storage.loadmat(path + '/mat' + '/DeepMIMO_dataset_' + str(index[i+80000]) + '.mat')
    data = data_mat['DeepMIMO_dataset']
    np.save(path + '/train10000/test' + '/data_' + str(i) + '.npy', data)
    np.save(path + '/train10000/test' + '/data_local_' + str(i) + '.npy', data_local[index[i+80000]])
np.save(path+'/train10000/test'+'/data_len.npy',test_num)


