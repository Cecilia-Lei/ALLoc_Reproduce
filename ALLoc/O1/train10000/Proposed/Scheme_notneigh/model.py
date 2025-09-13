import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
import math
from einops.layers.torch import Rearrange
from einops import rearrange,repeat

class PointwiseFeedforward_us(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super(PointwiseFeedforward_us, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerEncoderLayer_us(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer_us, self).__init__()
        self.self_attentionc = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        self.self_attentionp= nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        self.feedforwardc = PointwiseFeedforward_us(embed_dim, hidden_dim)
        self.feedforwardp = PointwiseFeedforward_us(embed_dim, hidden_dim)

        self.norm1c = nn.LayerNorm(embed_dim)
        self.norm2c = nn.LayerNorm(embed_dim)
        self.norm1p = nn.LayerNorm(embed_dim)
        self.norm2p = nn.LayerNorm(embed_dim)


    def forward(self, channel,pos, mask=None):
        channel_norm1 = self.norm1c(channel)
        pos_norm1=self.norm1p(pos)
        attn_channel,_= self.self_attentionc(channel_norm1, channel_norm1, channel_norm1,attn_mask=mask)
        attn_pos,_=self.self_attentionp(channel_norm1,channel_norm1,pos_norm1,attn_mask=mask)
        channel = channel + attn_channel
        pos=pos+attn_pos
        channel_norm2 = self.norm2p(channel)
        pos_norm2=self.norm2c(pos)
        ff_channel = self.feedforwardc(channel_norm2)
        ff_pos=self.feedforwardp(pos_norm2)
        channel = channel + ff_channel
        pos=pos+ff_pos

        return channel,pos

class Nlocnet(nn.Module):
    def __init__(self,ant_size,car_size,embed_dim, num_heads, hidden_dim,depth):
        super(Nlocnet, self).__init__()
        self.ant_size=ant_size
        self.car_size=car_size

        self.current_posembedding=nn.parameter.Parameter(torch.FloatTensor(embed_dim), requires_grad=True)
        self.current_posembedding.data.fill_(0)
        self.neigh_posembedding = nn.parameter.Parameter(torch.FloatTensor(embed_dim), requires_grad=True)
        self.neigh_posembedding.data.fill_(0)
        self.current_channelembedding = nn.parameter.Parameter(torch.FloatTensor(embed_dim), requires_grad=False)
        self.current_channelembedding.data.fill_(0)
        self.neigh_channelembedding = nn.parameter.Parameter(torch.FloatTensor(embed_dim), requires_grad=False)
        self.neigh_channelembedding.data.fill_(0)

        self.channel_inputfc=nn.Linear(ant_size*car_size*2,embed_dim)
        self.pos_inputfc=nn.Linear(2,embed_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerEncoderLayer_us(embed_dim, num_heads, hidden_dim))
        self.pos_outfc=nn.Linear(embed_dim,2)

    def forward(self,neigh_channel,neigh_pos,current_channel):
        batch_size=neigh_channel.shape[0]
        len_neigh=neigh_channel.shape[1]
        len_current=current_channel.shape[1]
        len_total=len_neigh+len_current
        channel=torch.cat((neigh_channel,current_channel),dim=1)
        current_pos = torch.mean(neigh_pos,dim=1,keepdim=False)
        current_pos=repeat(current_pos,'b p -> b k p',k=len_current)
        pos=torch.cat((neigh_pos,current_pos),dim=1)
        channel=rearrange(channel,'b l ant car c -> b l (ant car c)')
        channel=self.channel_inputfc(channel)
        pos=self.pos_inputfc(pos)
        neigh_channelembedding_matrix = repeat(self.neigh_channelembedding, 'h -> past_len h', past_len=len_neigh)
        current_channelembedding_matrix = repeat(self.current_channelembedding, 'h -> l h',l=len_current)
        channelembedding_matrix = torch.cat((neigh_channelembedding_matrix, current_channelembedding_matrix), dim=0)
        channelembedding_matrix=repeat(channelembedding_matrix,'l h -> b l h',b=batch_size)
        channel=channel+channelembedding_matrix

        neigh_posembedding_matrix = repeat(self.neigh_posembedding, 'h -> past_len h', past_len=len_neigh)
        current_posembedding_matrix = repeat(self.current_posembedding, 'h -> l h', l=len_current)
        posembedding_matrix = torch.cat((neigh_posembedding_matrix, current_posembedding_matrix), dim=0)
        posembedding_matrix = repeat(posembedding_matrix, 'l h -> b l h', b=batch_size)
        pos = pos + posembedding_matrix
        data_device = neigh_channel.device
        mask = torch.full((len_total, len_total), float(0))
        mask[:, len_neigh:len_total] = float('-inf')
        mask = mask.to(data_device)

        for layer in self.layers:
            channel,pos=layer(channel,pos,mask)

        out=self.pos_outfc(pos[:,len_neigh:len_total,:])
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