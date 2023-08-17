import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import sys
import numpy as np
from torch import einsum
from einops import repeat
import einops

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self,configs,context_window=336, patch_len=16,stride=16,learn_pe=True,d_model=128,dropout=0):
        super(Model, self).__init__()
        self.configs=configs
        self.backbone_trend=patch_RNN_backbone(configs)
        self.backbone_seasonal=patch_RNN_backbone(configs)
      
        
        self.revin_layer = RevIN( configs.enc_in)
    def forward(self, x, batch_x_mark=0, dec=0, batch_y_mark=0, batch_y=0) :
        
        # x: [Batch, Input length, Channel]
        
        # noise_1=repeat(self.noise_1,'B T D ->(r1 B) T (r2 D)',r1=256, r2=x.shape[2])
        support=0
        x = self.revin_layer(x, 'norm')
        # seasonal_x, trend_x = self.decompsition(x)
        trend_x=torch.mean(x,dim=1,keepdim=True)
        trend_x=einops.repeat(trend_x,'h w n -> h (repeat w) n',repeat=self.configs.seq_len)
        seasonal_x=x-trend_x

        

        trend=self.backbone_trend(trend_x,support,dec)
        seasonal=self.backbone_seasonal(seasonal_x,support,dec)
        # output=seasonal+trend+noise_1
        output=seasonal+trend
        
        
        output = self.revin_layer(output, 'denorm')
       
        return output[:,:,:] # [Batch, Output length, Channel]
    



    


class patch_RNN_backbone(nn.Module):
    def __init__(self,configs,context_window=336, patch_len=16,stride=16,learn_pe=True,d_model=128,dropout=0):
        super(patch_RNN_backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.d_model=configs.dec_in
        self.d_output=configs.c_out
        # Use this line if you want to visualize the weights
        
       
        self.patch_len=configs.patch_len
        self.stride=configs.stride
        self.dropout = nn.Dropout(dropout)
        self.W=nn.Linear(self.patch_len,d_model)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.hidden_size=configs.d_model
        self.d_ff=configs.d_ff
        self.layers=configs.e_layers
        self.encoder=nn.GRU(self.hidden_size,self.hidden_size,self.layers,batch_first=True)
        self.flatten = nn.Flatten(start_dim=-2)

        patch_num = int((context_window - self.patch_len)/self.stride + 1)+1
        self.linear = nn.Linear(patch_num*configs.d_model, self.pred_len)
        self.ff = nn.Sequential(nn.Linear(self.hidden_size, self.d_ff, bias=True),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(self.d_ff, self.hidden_size, bias=True))

        
        self.W_P = nn.Linear(configs.patch_len, configs.d_model) 
        
        

        
        
    def forward(self,x,support,dec=0):
        x = x.permute(0,2,1)
        
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        n_vars = x.shape[1]
        x=self.W_P(x)
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3])) 

        # u=x

        
        #GRU encoder
        output,h1=self.encoder(u)

        
        
        
      
        # dec=dec.permute(0,2,1)
        # dec = self.padding_patch_layer(dec)
        
        # dec = dec.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # dec = self.W_P(dec)
        # dec = torch.reshape(dec, (dec.shape[0]*dec.shape[1],dec.shape[2],dec.shape[3])) 
        # output,h1=self.decoder(dec,h1)

        output=torch.reshape(output, (-1,n_vars,output.shape[-2],output.shape[-1])) 
        output=self.flatten(output)
        output=self.linear(output)
        # affine_weight=self.softmax(self.affine_weight)*256
        # output=affine_weight*output
        output=output.permute(0,2,1)
        return output

    


class patch_RNN_2(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self,configs,context_window=336, patch_len=16,stride=16,learn_pe=True,d_model=128,dropout=0):
        super(patch_RNN_2, self).__init__()
        self.seq_len = configs.enc_len
        self.pred_len = configs.dec_len
        
        self.d_model=configs.encoder_size
        self.d_output=configs.target_len
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        patch_num = int((context_window - patch_len)/stride + 1)+1
        self.patch_len=patch_len
        self.stride=stride
        self.dropout = nn.Dropout(dropout)
        self.W=nn.Linear(self.patch_len,d_model)
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
        self.hidden_size=128
        self.d_ff=256
        self.layers=4
        self.encoder=nn.GRU(self.hidden_size,self.hidden_size,self.layers,batch_first=True)
        self.flatten = nn.Flatten(start_dim=-2)
        patch_num = int((self.pred_len - patch_len)/stride + 1)+1
        self.linear = nn.Linear(patch_num*self.hidden_size, self.pred_len)
        self.ff = nn.Sequential(nn.Linear(self.hidden_size, self.d_ff, bias=True),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(self.d_ff, self.hidden_size, bias=True))


        self.dec_pos_embedding = nn.Parameter(torch.randn(1, 7, 22, self.hidden_size))
        self.decoder=nn.GRU(self.hidden_size,self.hidden_size,self.layers,batch_first=True)
        self.end_conv=nn.Conv1d(1,self.pred_len,kernel_size= self.hidden_size,bias=True)
        #AGCRN
        
        self.node_embeddings = nn.Parameter(torch.randn(22, 10), requires_grad=True)
        self.W_P = nn.Linear(self.patch_len, self.hidden_size) 
        
        self.activation=nn.GELU()
        self.w_d=nn.Linear(configs.decoder_size,configs.target_len)

        self.W_t=nn.Linear(self.seq_len,self.pred_len)
        #norm
        # self.norm_attn = nn.LayerNorm(self.hidden_size)
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.hidden_size), Transpose(1,2))
        self.norm_ff = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.hidden_size), Transpose(1,2))

        self.dense=nn.Linear(16,16)
        self.linear_hidden=nn.Linear(self.hidden_size, self.pred_len)
    def forward(self, x,dec=0) :

        # x: [Batch, Input length, Channel]
        support,x=x[:,:,:11],x[:,:,11:]
        batch_size=x.shape[0]

        # x=x[0]

        x = x.permute(0,2,1)
        
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        n_vars = x.shape[1]
        x=self.W_P(x)
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3])) 

        # u=x

        # output=self.norm_ff(u)
        #GRU encoder
        output,h1=self.encoder(u)

        # h1=self.dense(h1)
        
      
        # output=self.norm_attn(output)
        
       

        #encoder -> decoder
        support=self.W_t(support.permute(0,2,1)).permute(0,2,1)
        dec=dec+support
        dec=self.w_d(dec)
        dec=dec.permute(0,2,1)
        
        dec = self.W_t(dec)
       
        

        # output=torch.reshape(output, (-1,n_vars,output.shape[-2],output.shape[-1])) 
        # output = output.permute(0,1,3,2)  
        # output=self.flatten(output)
        # output=self.linear(output)
        # output=output+dec
        # output=output.permute(0,2,1)

        h1=torch.reshape(h1[-1,:,:],(-1,n_vars,output.shape[-1]))
        output=self.linear_hidden(h1)
        output=output+dec
        output=output.permute(0,2,1)

        

        
        return output[:,:,:],() # [Batch, Output length, Channel]
    


    

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False,custom=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.custom=custom
        self.linear=nn.Linear(256,256)
        self._init_params()
        

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x
    
    def _get_value(self):
        return self.mean,self.std
    
    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

        self.mean_weight=nn.Parameter(torch.ones(self.num_features,self.num_features))
        self.std_weight=nn.Parameter(torch.ones(self.num_features,self.num_features))
        self.mean_bias=nn.Parameter(torch.zeros(self.num_features))
        self.std_bias=nn.Parameter(torch.zeros(self.num_features))
        
        self.linear=nn.Linear(37,37)
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        # self.predict=self.linear(x.permute(0,2,1)).permute(0,2,1)
        # if self.custom:
        #     # self.mean=einsum('B a N,N M->B a M',self.mean,mean_weight)+mean_bias
        #     # self.stdev=einsum('B a N,N M->B a M',self.stdev,std_weight)+std_bias
        #     self.mean=self.mean*mean_weight+mean_bias
        #     self.stdev=self.stdev*std_weight+std_bias
    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        # if self.affine:
        #     x = x * self.affine_weight
        #     x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.custom:
            
            self.mean=self.mean*mean_weight+mean_bias
            self.stdev=self.stdev*std_weight+std_bias
        
        
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
            
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.std=torch.std(x, dim=1, keepdim=True).detach()
        return x
    

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))
    
class DishTS(nn.Module):
    def __init__(self, args,dish_init,n_series,seq_len):
        super().__init__()
        init = dish_init #'standard', 'avg' or 'uniform'
        activate = False
        n_series = n_series # number of series
        lookback = seq_len # lookback length
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(n_series, lookback, 2)/lookback)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback+torch.rand(n_series, lookback, 2)/lookback)
        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)), nn.Parameter(torch.zeros(n_series))
        self.activate = activate

    def forward(self, batch_x, mode='forward', dec_inp=None):
        if mode == 'norm':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x, dec_inp
        elif mode == 'denorm':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def preget(self, batch_x):
        x_transpose = batch_x.permute(2,0,1) 
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1,2,0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:,:1,:], theta[:,1:,:] 
        self.xil = torch.sum(torch.pow(batch_x - self.phil,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)

    def forward_process(self, batch_input):
        #print(batch_input.shape, self.phil.shape, self.xih.shape)
        temp = (batch_input - self.phil)/torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst
    
    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih
    


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x-moving_mean
        return res, moving_mean