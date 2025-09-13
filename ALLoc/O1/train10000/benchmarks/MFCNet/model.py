import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
import math

class LocNet(nn.Module):
    def __init__(self,num_ant,num_car,hidden_size,lay=2):
        super(LocNet, self).__init__()
        self.num_ant=num_ant
        self.num_car=num_car
        self.input_size=self.num_ant*2
        self.hidden_size=hidden_size
        self.lay=lay
        self.lstmmodel=nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size, num_layers=self.lay, batch_first=True)

        self.fcmodel=nn.Sequential(
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.LeakyReLU(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.3),
            nn.Linear(self.hidden_size,2)
        )

    def forward(self,data):
        batch_size=data.shape[0]
        data=data.permute([0,2,1,3])
        data=data.reshape([batch_size,-1,self.input_size])

        out, (h_n, c_n) = self.lstmmodel(data)
        ans=self.fcmodel(out)
        return ans

class DatasetFolder_mapping(Dataset):
    def __init__(self, path):
        self.path=path
        self.len=int(np.load(self.path+'/data_len.npy'))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data=np.load(self.path+'/data_'+str(index)+'.npy')
        local=np.load(self.path+'/data_local_'+str(index)+'.npy')
        return data,local

def RMSE(input,output):
    batch_size=input.shape[0]
    rmse_batch=torch.norm(input-output,dim=1)
    rmse_average=torch.mean(rmse_batch)
    return rmse_average