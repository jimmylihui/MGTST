import torch
import torch.nn as nn
from einops import repeat
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import math
class RevIN_3(nn.Module):
    def __init__(self,configs, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN_3, self).__init__()
        self.configs=configs
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.hidden_dim=256
        if self.affine:
            self._init_params()
        self._init_Z()
    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x
    def _init_Z(self):
        #A for input
        self.z=nn.Parameter(torch.randn(self.hidden_dim))
        self.input_mean=nn.Linear(self.configs.seq_len,1)
        self.output_mean=nn.Linear(1,self.num_features)
        self.dense=nn.Linear(self.num_features,1)
        self.active=nn.ReLU()
        self.output_layer= nn.Parameter(torch.ones(self.num_features, self.num_features))
        self.softmax=nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(0.2)
    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        
            
            
       
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()*2

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:

            # self.mean=self.input_mean(x.permute(0,2,1)).permute(0,2,1)
            self.mean=torch.mean(x, dim=1, keepdim=True)
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            res=self.dense(self.mean)
            

            res=self.output_mean(res)
            res=self.dropout(res)
            # res=self.active(res)
            x = x + res+self.mean
        return x