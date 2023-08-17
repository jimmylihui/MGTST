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
        self.Linear = nn.Linear(self.seq_len, self.pred_len*2)#different channel sharing a same weight
      
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.revin_layer = RevIN(configs,configs.enc_in, affine=configs.affine, subtract_last=configs.subtract_last)
       
        
        
        self.linear_2=nn.Linear(configs.seq_len,configs.pred_len*1000)
        self.active=nn.Softmax(dim=-1)
        
        
     
        
    def forward(self, x):
        x=self.revin_layer(x,'norm')
        
        
 
        x=x.permute(0,2,1)
        
       
        z=self.linear_2(x)
        z=torch.reshape(z,(z.shape[0],z.shape[1],-1,1000))

        z=self.active(z)
        weight=torch.arange(0,1,0.001).unsqueeze(-1).to("cuda:0")
        z=einsum('B D T L, L O -> B D T O',z,weight)
        z=z.squeeze(-1)

        z=z.permute(0,2,1)
        
        z=self.revin_layer(z,'denorm')
        return z # [Batch, Output length, Channel]
    