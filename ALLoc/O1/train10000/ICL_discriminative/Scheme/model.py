import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
import math
from einops.layers.torch import Rearrange
from einops import rearrange,repeat

class Nlocnet(nn.Module):
    def __init__(self,ant_size,car_size,embed_dim, num_heads, hidden_dim,depth):
        super(Nlocnet, self).__init__()
        self.ant_size=ant_size
        self.car_size=car_size
        self.embed_dim=embed_dim
        self.len_upperlimit=64*2+1
        self.pos_posembedding=nn.parameter.Parameter(torch.zeros(self.len_upperlimit,embed_dim), requires_grad=True)

        self.channel_inputfc=nn.Linear(ant_size*car_size*2,embed_dim)
        self.pos_inputfc=nn.Linear(2,embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0,
            batch_first=True,
            norm_first=True
        )
        self.decoder=nn.TransformerEncoder(decoder_layer,depth)
        self.pos_outfc=nn.Linear(embed_dim,2)

    def forward(self,neigh_channel,neigh_pos,current_channel):
        batch_size=neigh_channel.shape[0]
        len_neigh=neigh_channel.shape[1]
        neigh_channel=rearrange(neigh_channel,'b l ant car c -> b l (ant car c)')
        neigh_channel=self.channel_inputfc(neigh_channel)
        current_channel = rearrange(current_channel, 'b l ant car c -> b l (ant car c)')
        current_channel = self.channel_inputfc(current_channel)
        neigh_pos=self.pos_inputfc(neigh_pos)

        seq=torch.stack([neigh_channel, neigh_pos], dim=2)
        seq=seq.reshape([batch_size,2*len_neigh,self.embed_dim])
        seq=torch.cat([seq,current_channel],dim=1)

        pos_embedding_matrix=repeat(self.pos_posembedding,'l dim -> b l dim',b=batch_size)
        seq=seq+pos_embedding_matrix[:,0:2*len_neigh+1,:]

        data_device = neigh_channel.device
        sz = seq.shape[1]
        mask = torch.full((sz, sz), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        mask = mask.to(data_device)

        seq=self.decoder(src=seq,mask=mask)
        out=self.pos_outfc(seq[:,-1:,:])
        return out

class DatasetFolder(Dataset):
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