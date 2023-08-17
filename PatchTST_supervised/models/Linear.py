import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.DishTS import DishTS
from layers.Revin_2 import RevIN_2
from layers.RevIN_3 import RevIN_3
import einops
import math
from torch import einsum
class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(int(self.seq_len), int(self.pred_len),bias=False)#different channel sharing a same weight
        # self.ff = nn.Sequential(nn.Linear(self.seq_len, 256),
        #                         nn.Linear(256, self.pred_len))
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.revin_layer = RevIN(configs,configs.enc_in, affine=configs.affine, subtract_last=configs.subtract_last)
       
        
        self.linear_2=nn.Linear(self.seq_len,self.seq_len)
        self.flatten = nn.Flatten(start_dim=-2)
        
        
    def forward(self, x,*args):
        # x: [Batch, Input length, Channel]
        
       
        # x,_,_=self.revin_layer(x,'norm')
        
        
        

        x=x.permute(0,2,1)

        # x=torch.fft.rfft(x)
        # topk,indices=torch.topk(torch.abs(x),10)
        # indc=torch.zeros(x.shape).to('cuda:0').scatter(-1,indices, 1)
        # x=x*indc
        # x=torch.fft.irfft(x)
        
        x=self.Linear(x)
        
       
        x=x.permute(0,2,1)
        # x=self.revin_layer(x,'denorm')
        return x # [Batch, Output length, Channel]
    

class downsampling(nn.Module):
    def __init__(self, seq_len,predict_len,scale):
        super(downsampling, self).__init__()
        self.scale=scale
        self.seq_len=seq_len
        self.patch_size=int(self.seq_len/self.scale)
        self.Linear_1=nn.Linear(self.patch_size,predict_len,bias=False)
        self.Linear_2=nn.Linear(self.patch_size,predict_len)
    def forward(self,x):
        x=x.unfold(dimension=-1, size=self.scale, step=self.scale)
        mean=torch.mean(x,dim=-1)
        
        x=0.5*self.Linear_1(mean)
        return x 