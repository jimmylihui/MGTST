import torch
import torch.nn as nn
from einops import repeat
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
class RevIN_2(nn.Module):
    def __init__(self,configs, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN_2, self).__init__()
        self.configs=configs
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()
        self._init_A(configs)
    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x
    def _init_A(self,configs):
        #A for input
        x=torch.arange(0,configs.seq_len)
        self.A=torch.vstack([torch.ones(len(x)),x]).T
        x_prime=torch.arange(0,configs.pred_len+configs.seq_len)
        self.A_prime=torch.vstack([torch.ones(len(x_prime)),x_prime]).T
        
    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            base_weight=tf.linalg.lstsq(self.A,x.cpu())
            
            base_weight=base_weight.numpy()
            self.base_weight=torch.from_numpy(base_weight)
            
            
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:

            mean=torch.matmul(self.A,self.base_weight).squeeze(dim=-1).to('cuda:0')
            mean_2=torch.mean(x, dim=1, keepdim=True).detach()
            x = x - mean
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
            mean=torch.matmul(self.A_prime,self.base_weight).squeeze(dim=-1).to('cuda:0')
            mean=mean[:,self.configs.seq_len:,:]
            x = x + mean
        return x
