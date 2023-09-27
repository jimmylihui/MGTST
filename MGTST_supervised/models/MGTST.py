__all__ = ['MGTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.MGTST_backbone import MGTST_backbone
import math


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        self.configs=configs
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # model
        self.decomposition = decomposition
        
        
        self.scale=configs.scale
        
        self.gate=configs.gate
        self.group=configs.group
        self.c_in_collect=[]
        # self.model=nn.ModuleList()
        # for i in range(configs.group):
        #     if (i==configs.group-1):
        #         self.c_in_collect.append(c_in-i*math.floor(c_in/configs.group))
        #     else:
        #         self.c_in_collect.append(math.floor(c_in/configs.group))
        self.model= MGTST_backbone(configs,self.scale,self.gate,c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                subtract_last=subtract_last, verbose=verbose, **kwargs)
                
        # self.channel_mixing=SVD_Block(7,7,720)
    def forward(self, x,batch_x_mark,batch_y_mark):           # x: [Batch, Input length, Channel]
       
        
        # self.output_collect=[]
        # start=0
        # for i in range(self.group):
        #     z=x[:,:,start:start+self.c_in_collect[i]]
        #     z=z.permute(0,2,1)
        #     z,attn_collect=self.model[i](z,batch_x_mark,batch_y_mark)
        #     z=z.permute(0,2,1)
        #     self.output_collect.append(z)
        #     start=start+self.c_in_collect[i]
        # z=torch.concat(self.output_collect,dim=-1)


        
        z = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        z,attn_collect,attn_gate_collect = self.model(z,batch_x_mark,batch_y_mark)
        z = z.permute(0,2,1)    # x: [Batch, Input length, Channel]
        self.attn_collect=attn_collect
        self.attn_gate_collect=attn_gate_collect
        
       
                
        return z

    def get_attn(self):
        return self.attn_collect
    
    def get_gate_attn(self):
        return self.attn_gate_collect