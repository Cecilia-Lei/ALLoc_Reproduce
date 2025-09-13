import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
import math

class LocNet(nn.Module):
    def __init__(self,num_ant,num_car,hidden_size):
        super(LocNet, self).__init__()
        self.num_ant=num_ant
        self.num_car=num_car
        self.input_size=self.num_ant*self.num_car*2
        self.hidden_size=hidden_size
        self.convmodel=nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.hidden_size, kernel_size=(3, 3), padding=1),

            nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(in_channels=self.hidden_size,out_channels=self.hidden_size,kernel_size=(3,3),padding=1),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=self.hidden_size, out_channels=2, kernel_size=(3, 3), padding=1),
        )
        self.fcmodel=nn.Sequential(
            nn.Linear(self.input_size//(4*4*4),self.input_size//(4*4*4)),
            nn.LeakyReLU(0.3),
            nn.Linear(self.input_size//(4*4*4),2)
        )

    def forward(self,data):
        batch_size=data.shape[0]
        data=data.permute([0,3,1,2])
        ans=self.convmodel(data)
        ans=ans.reshape([batch_size,-1])
        ans=self.fcmodel(ans)
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
