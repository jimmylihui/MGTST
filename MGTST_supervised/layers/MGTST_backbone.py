__all__ = ['MGTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from einops import repeat
#from collections import OrderedDict
from layers.MGTST_layers import *
from layers.RevIN import RevIN
from torch import einsum
from einops import rearrange

# Cell
class MGTST_backbone(nn.Module):
    def __init__(self, configs,scale,gate,c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
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
        
        
        self.channel_dependent=configs.channel_dependent
        # Backbone 
        self.backbone=nn.ModuleList()
        self.head=nn.ModuleList()
        self.scale=scale
        self.padding_patch_layer=nn.ModuleList()
        self.identity=nn.ModuleList()
        # self.W_P=nn.ModuleList()
        padding_patch = padding_patch
        
                
        self.group=configs.group
        total_patch_num=0
        self.patch_len=patch_len
        self.stride=stride
        self.cls_token=[]
        self.spatial_backbone=TSTEncoder(c_in, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        # self.cls_communication=nn.Linear(c_in,c_in)
        self.weight_backbone=nn.ModuleList()
        self.linear=nn.ModuleList()
        for i in range(self.scale):
            # Patching
            patch_len_stage = patch_len*(i+1)

            stride_stage = stride*(i+1)
   
            self.padding_patch_layer.append(nn.ReplicationPad1d((0, stride_stage)))
            patch_num = int((context_window - patch_len_stage)/stride_stage + 1)+1+configs.gate
            total_patch_num=total_patch_num+patch_num+self.channel_dependent-configs.gate
            backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len_stage, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
            
            if(gate==1):
                weight_backbone=GateiEncoder(c_in=1, patch_num=int(c_in/self.group), patch_len=d_model, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
            
                self.weight_backbone.append(weight_backbone)
                cls_token = nn.Parameter(torch.randn(1, c_in,1, patch_len_stage)).to('cuda:0')
                self.cls_token.append(cls_token)
                linear=nn.Linear(d_model,patch_num-configs.gate,bias=False)
                self.linear.append(linear)
            self.backbone.append(backbone)
            

            
            
            
            
        head_nf = d_model * (total_patch_num)
        n_vars = c_in
        pretrain_head = pretrain_head
        head_type = head_type
        individual = individual
        self.head = Flatten_Head(individual, n_vars, head_nf, target_window, head_dropout=head_dropout)

        # self.cls=nn.Parameter(torch.randn())


        
       
        self.gate=gate

    def forward(self, x,batch_x_mark,batch_y_mark):                                                                   # z: [bs x nvars x seq_len]
        # norm
        
        if self.revin: 
            x = x.permute(0,2,1)
            
            x,mean,std = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)
        
       
        # x = self.dropout(x + self.W_pos) 
        
        self.attn_collect=[]
        self.attn_gate_collect=[]
        out=0
        for i in range(self.scale):
            # input=self.identity(x)
            
            z = self.padding_patch_layer[i](x)
            z = z.unfold(dimension=-1, size=self.patch_len*(i+1), step=self.stride*(i+1))                   # z: [bs x nvars x patch_num x patch_len]
            
            if(self.gate==True):
                cls_tokens = repeat(self.cls_token[i], '1 n h d -> b n h d', b = z.shape[0])
                z=torch.cat((cls_tokens,z),2)
                # z=torch.cat([z,mean_n,std_n],dim=-1)
        
                z = z.permute(0,1,3,2)  
                z,attn = self.backbone[i](z)                                                                # z: [bs x nvars x d_model x patch_num]
                self.attn_collect.append(attn)
                z,cls=z[:,:,:,1:],z[:,:,:,:1]
             
                cls_number=int(cls.shape[1]/self.group)*self.group
                cls_0=cls[:,:cls_number,:,:]
                cls_1=cls[:,cls_number:,:,:]
                cls_0=torch.reshape(cls_0,(cls_0.shape[0],self.group,-1,cls_0.shape[2],cls_0.shape[3]))
                cls_0=torch.reshape(cls_0,(cls_0.shape[0]*cls_0.shape[1],cls_0.shape[2],cls_0.shape[3],cls_0.shape[4]))
                
                
                
                cls_0,attention_gate=self.weight_backbone[i](cls_0.permute(0,3,2,1))
                self.attn_gate_collect.append(attention_gate)
                cls_0=cls_0.permute(0,3,1,2)

                cls_0=torch.reshape(cls_0,(-1,self.group,cls_0.shape[1],cls_0.shape[2],cls_0.shape[3]))
                cls_0=torch.reshape(cls_0,(cls_0.shape[0],cls_0.shape[1]*cls_0.shape[2],cls_0.shape[3],cls_0.shape[4]))
                if(cls_1.shape[1]!=0):
                    cls_1,attention_gate=self.weight_backbone[i](cls_1.permute(0,3,2,1))
                
                    cls_1=cls_1.permute(0,3,1,2)
                else:
                    cls_1=cls_1.permute(0,1,3,2)
                cls=torch.cat([cls_0,cls_1],dim=1)

                cls=self.linear[i](cls)
                # cls=repeat(cls,'b n 1 p->b n d p',d=z.shape[2])
                
                
                
               
                # z=0.5*torch.sigmoid(cls)*z+0.5*z
                # original=z
                z=torch.sigmoid(cls)*z
                
            else:
                z = z.permute(0,1,3,2)  
                z,attn = self.backbone[i](z)   
                self.attn_collect.append(attn)

            
            if i== 0:
                # out=torch.cat([z/self.scale,original/self.scale],dim=-1)
                out=z/self.scale
            else:
                out=torch.cat([out,z/self.scale],dim=-1)
                
            
           

        out = self.head(out)                                                             # z: [bs x nvars x target_window] 
        
        
        # denorm
        if self.revin: 
            out = out.permute(0,2,1)
           
            out = self.revin_layer(out, 'denorm')
            out = out.permute(0,2,1)
        return out,self.attn_collect,self.attn_gate_collect
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )
    def get_attn(self):
        return self.attn_collect




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
        
        self.scale = (1 / (nf*target_window))
        self.weights_FFT = nn.Parameter(self.scale * torch.rand(nf,target_window, dtype=torch.float))
        
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
            # x = torch.einsum("bxi,io->bxo",x , self.weights_FFT)
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
        self.W_P = nn.Linear(patch_len, d_model,bias=True)        # Eq 1: projection of feature vectors onto a d-dim vector space

        self.scale = (1 / (patch_len*d_model))
        self.W_P_linear = nn.Parameter(self.scale * torch.rand(patch_len,d_model, dtype=torch.float))
        # self.W_P = spatial_encoding(patch_len,d_model,c_in,n_heads=n_heads,d_ff=d_ff)
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        

        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        # W_channel=self.W_channel.unsqueeze(-1)
        # x = self.dropout(x + self.W_pos)
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]
        # x = torch.einsum("bxhi,io->bxho",x , self.W_P_linear)

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]
        # u=self.W_pos(u)

        # Encoder
        z,attn = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]

        # z=self.W_P(z)
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        
        return z,attn    
            

class GateiEncoder(nn.Module):  #i means channel-independent
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

        

        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        
        # self.W_pos=tAPE(d_model,dropout,max_len=patch_num,scale_factor=1)
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        
       

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]

        # u=self.W_pos(u)

        # Encoder
        z,attn = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]

        # z=self.W_P(z)
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        
        return z,attn    
    
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
            for mod in self.layers: output, scores,attn = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output,attn
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
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention,q_len=q_len)

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
        # self.scale = (1 / (d_model*d_model))
        # self.ff = nn.Parameter(self.scale * torch.rand(d_model,d_model, dtype=torch.float))
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
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
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
        # src2 = torch.einsum("bxi,io->bxo",src , self.ff)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores,attn
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False,q_len=0):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa,q_len=q_len)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False,q_len=0):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

        self.seq_len=q_len
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * q_len - 1), n_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(q_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += q_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)
    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


       
       
