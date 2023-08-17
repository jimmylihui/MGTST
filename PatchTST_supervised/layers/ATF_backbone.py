__all__ = ['ATF_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from einops import repeat
#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from torch import einsum
# Cell
class ATF_backbone(nn.Module):
    def __init__(self, configs,scale,c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        self.configs=configs
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(configs,c_in, affine=affine, subtract_last=subtract_last)
        
        
        
        # Backbone 
        self.backbone=nn.ModuleList()
        self.head=nn.ModuleList()
        self.scale=scale
        self.padding_patch_layer=nn.ModuleList()
        self.identity=nn.ModuleList()
        # self.W_P=nn.ModuleList()
        padding_patch = padding_patch
        
                
        
        total_patch_num=0
        self.patch_len=patch_len
        self.stride=stride
        
        self.spatial_backbone=TSTEncoder(c_in, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        for i in range(self.scale):
            # Patching
            patch_len_stage = patch_len*(i+1)
            stride_stage = stride*(i+1)
            self.padding_patch_layer.append(nn.ReplicationPad1d((0, stride_stage)))
            patch_num = int((context_window - patch_len_stage)/stride_stage + 1)+1
            total_patch_num=total_patch_num+patch_num
            backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len_stage, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
            self.backbone.append(backbone)
            self.identity=nn.Linear(configs.seq_len,configs.seq_len)
            # self.identity.append(identity)
            # self.W_P.append(nn.Linear(patch_len_stage,d_model))
        # self.backbone = TSTiEncoder(c_in, patch_num=total_patch_num, patch_len=patch_len_stage, max_seq_len=max_seq_len,
        #                         n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
        #                         attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
        #                         attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
        #                         pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        head_nf = d_model * total_patch_num
        n_vars = c_in
        pretrain_head = pretrain_head
        head_type = head_type
        individual = individual
        self.head = Flatten_Head(individual, n_vars, head_nf, target_window, head_dropout=head_dropout)


        

        
        
        
       
        
       
    def forward(self, x):                                                                   # z: [bs x nvars x seq_len]
        # norm
        # x=torch.round(x,decimals=1)
        if self.revin: 
            x = x.permute(0,2,1)
            
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)
        
       
        # u,s,v=torch.linalg.svd(x,full_matrices=True)
        # s[:,6:]=0
        # s=torch.diag_embed(s)
        # out_us = torch.matmul(u,s)
        # x = torch.matmul(out_us,v)
        out=0
        for i in range(self.scale):
            # input=self.identity(x)
            
            z = self.padding_patch_layer[i](x)
            z = z.unfold(dimension=-1, size=self.patch_len*(i+1), step=self.stride*(i+1))                   # z: [bs x nvars x patch_num x patch_len]
            z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
       
            z = self.backbone[i](z)                                                                # z: [bs x nvars x d_model x patch_num]
            # z=self.W_P[i](z)
            if i== 0:
                out=z/self.scale
            else:
                out=torch.cat([out,z/self.scale],dim=-1)
                # out=torch.cat([out,z/self.scale],dim=-2)
            
            # x=x-input

        # spaital attention
        # out=out.permute(0,3,1,2)
        # patch_num=out.shape[1]
        # out=torch.reshape(out,(out.shape[0]*out.shape[1],out.shape[2],out.shape[3]))
        # out=self.spatial_backbone(out)
        # out=torch.reshape(out,(-1,patch_num,out.shape[1],out.shape[2]))
        # out=out.permute(0,2,1,3)
        
        # out=self.backbone(out)
        out = self.head(out)                                                                    # z: [bs x nvars x target_window] 
            
        
        # denorm
        if self.revin: 
            out = out.permute(0,2,1)
           
            out = self.revin_layer(out, 'denorm')
            out = out.permute(0,2,1)
        return out
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )
    




class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        # u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2 = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        
        return src
        




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model 
        d_v = d_model 

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model,d_model)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

        self.wbias = nn.Parameter(torch.Tensor(50, 50))
        nn.init.xavier_uniform_(self.wbias)
        self.norm_attn = nn.LayerNorm(d_model)

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        B, T, _ = Q.shape
        Q = self.to_q(Q).view(B, T, self.d_k)
        K = self.to_k(K).view(B, T, self.d_k)
        V = self.to_v(V).view(B, T, self.d_k)
        temp_wbias = self.wbias[:T, :T].unsqueeze(0) # sequences can still be variable length

        '''
        From the paper
        '''
        # Q_sig = torch.sigmoid(Q)
        # # Q_sig=Q
        # temp = torch.exp(temp_wbias) @ torch.mul(torch.exp(K), V)
        # weighted = temp / (torch.exp(temp_wbias) @ torch.exp(K))
        # Yt = torch.mul(Q_sig, weighted)

        # Yt = Yt.view(B, T, self.d_k)

        '''
        From the paper
        '''
        # weights = torch.mul(torch.softmax(K, 1), V).sum(dim=1, keepdim=True)
        # Q_sig = torch.sigmoid(Q)
        # # Q_sig=Q
        # Yt = torch.mul(Q_sig, weights)

        Q_sig=torch.sigmoid(Q)
        Yt=torch.softmax(K,1)
        Yt = torch.mul(Q_sig, Yt)

        # Yt=K
        # Yt=self.norm_attn(Yt)
        Yt = Yt.view(B, T, self.d_k)
        
        
        
        return Yt


