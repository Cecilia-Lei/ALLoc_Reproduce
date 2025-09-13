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

batch_size=500
epochs=100000000
steps=300000
learning_rate=1e-4
print_freq=200

ant_size=32
car_size=32
hidden_size=256

depth=8

model=Nlocnet(ant_size=ant_size,car_size=car_size,embed_dim=hidden_size,num_heads=4,hidden_dim=hidden_size*2,depth=depth)
from calflops import calculate_flops
input1=torch.randn(1,64,ant_size,car_size,2)
input2=torch.randn(1,64,2)
input3=torch.randn(1,1,ant_size,car_size,2)
inputs={
    "neigh_channel":input1,
    "neigh_pos":input2,
    "current_channel":input3
}
flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=False)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



path='/e/forALLoc/ALLoc/Data/ALLoc_data_share/32ant*32car_3.5GHz_40MHz_R501-1400_V1_24.4.12/train10000'
train_dataset = DatasetFolder(path+'/sequence')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

test_dataset=DatasetFolder(path+'/sequence_test_len64')
test_loader=torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)


class LrSchedule():
    def __init__(self,initial_lr,optimizer=None):
        super(LrSchedule,self).__init__()
        self.steps=0
        self.initial_lr=initial_lr
        self.optimizer=optimizer

    def step(self):
        self.steps+=1
        if self.steps <= 150000 :
            lr=self.initial_lr
        elif self.steps > 150000 and self.steps <= 200000:
            lr=self.initial_lr * (0.2 ** 1)
        elif self.steps > 200000 and self.steps <= 250000:
            lr=self.initial_lr * (0.2 ** 2)
        elif self.steps > 250000 and self.steps <= 300000:
            lr=self.initial_lr * (0.2 ** 3)
        cur_lr=lr
        for p in self.optimizer.param_groups:
            p['lr']=cur_lr
        return cur_lr

lr_scheduler=LrSchedule(initial_lr=learning_rate,optimizer=optimizer)

test_num=20000
total_neighbor_num=64
least_neighbor_num=16
whole_index=[a for a in range(least_neighbor_num+1,total_neighbor_num+1+1)]
found = False
for epoch in range(epochs):
    for i, (data,local) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        data=data.cuda().float()
        data=data*10000
        local=local.cuda().float()
        local=local[:,:,0:2]

        index = [b for b in range(total_neighbor_num + 1)]
        random.shuffle(index)
        index_len = random.choice(whole_index)
        index = index[:index_len]
        data_neigh=data[:,index[:-1],...]
        data_current=data[:,index[-1:],...]
        local_neigh=local[:,index[:-1],...]
        local_current=local[:,index[-1:],...]

        local_pred=model(data_neigh,local_neigh,data_current)
        MSE_loss = nn.MSELoss()(local_pred, local_current)
        loss = MSE_loss
        loss.backward()
        lr_scheduler.step()
        if lr_scheduler.steps <= steps:
            optimizer.step()
        else:
            found =True
            break
    if found ==True:
        break

    if epoch % 20 == 0:
        print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        print("epoch :",epoch)
        print("step :", lr_scheduler.steps)
        torch.save(model.state_dict(), './model' + '.pth')

        model.eval()
        sum_rmse = 0
        for i, (data, local) in enumerate(test_loader):
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
            sum_rmse+=rmse
        avg_rmse=sum_rmse/(i+1)
        print("avg_distance :",avg_rmse)

torch.save(model.state_dict(), './model' + '.pth')