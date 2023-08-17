import torch
import torch.nn as nn
from einops import repeat
import torch.nn.functional as F
from torch import einsum
from layers.RevIN import RevIN
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs=configs
        self.patch_length=21
        self.stride=configs.stride
        self.d_model=configs.d_model
        self.mlpmixer=nn.ModuleList()
        for i in range(1):
            mlpmixer=MlpMixer(num_blocks=2,patch_size=self.patch_length*(i+1),tokens_hidden_dim=int(16/(i+1)),channels_hidden_dim=self.d_model,tokens_mlp_dim=int(16/(i+1)),channels_mlp_dim=self.d_model)
            self.mlpmixer.append(mlpmixer)
        nf=self.d_model*16
        self.head=Flatten_Head(nf=nf,target_window=configs.pred_len)
        self.revin_layer = RevIN(configs,configs.c_out, affine=configs.affine, subtract_last=configs.subtract_last)
    def forward(self,x):
        #concat channel and batch
        x = self.revin_layer(x, 'norm')
        nvars=x.shape[-1]
        x=x.permute(0,2,1)
       
        x=torch.reshape(x,(x.shape[0]*x.shape[1],x.shape[2]))
        #unfold into patches
        out=0
        for i in range(1):
            y=self.mlpmixer[i](x)
            if i==0:
                out=y
            else:
                out=torch.cat([out,y],dim=1)
        x=self.head(out)
        x=torch.reshape(x,(-1,nvars,x.shape[-1]))
        x=x.permute(0,2,1)
        x = self.revin_layer(x, 'denorm')
        return x



class Flatten_Head(nn.Module):
    def __init__(self,  nf, target_window, head_dropout=0):
        super().__init__()
        
       
        
        
       
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class MlpBlock(nn.Module):
    def __init__(self,input_dim,mlp_dim=512) :
        super().__init__()
        self.fc1=nn.Linear(input_dim,mlp_dim)
        self.gelu=nn.GELU()
        self.fc2=nn.Linear(mlp_dim,input_dim)
    
    def forward(self,x):
        #x: (bs,tokens,channels) or (bs,channels,tokens)
        return self.fc2(self.gelu(self.fc1(x)))



class MixerBlock(nn.Module):
    def __init__(self,tokens_mlp_dim=16,channels_mlp_dim=1024,tokens_hidden_dim=32,channels_hidden_dim=1024):
        super().__init__()
        self.ln=nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp_block=MlpBlock(tokens_mlp_dim,mlp_dim=tokens_hidden_dim)
        self.channels_mlp_block=MlpBlock(channels_mlp_dim,mlp_dim=channels_hidden_dim)

    def forward(self,x):
        """
        x: (bs,tokens,channels)
        """
        ### tokens mixing
        y=self.ln(x)
        y=y.transpose(1,2) #(bs,channels,tokens)
        y=self.tokens_mlp_block(y) #(bs,channels,tokens)
        ### channels mixing
        y=y.transpose(1,2) #(bs,tokens,channels)
        out =x+y #(bs,tokens,channels)
        y=self.ln(out) #(bs,tokens,channels)
        y=out+self.channels_mlp_block(y) #(bs,tokens,channels)
        return y

class MlpMixer(nn.Module):
    def __init__(self,num_blocks,patch_size,tokens_hidden_dim,channels_hidden_dim,tokens_mlp_dim,channels_mlp_dim):
        super().__init__()
        
        self.num_blocks=num_blocks #num of mlp layers
        self.patch_size=patch_size
        self.tokens_mlp_dim=tokens_mlp_dim
        self.channels_mlp_dim=channels_mlp_dim
        self.embd=nn.Linear(self.patch_size,channels_hidden_dim) 
        self.ln=nn.LayerNorm(channels_mlp_dim)
        self.mlp_blocks=[]
        for _ in range(num_blocks):
            self.mlp_blocks.append(MixerBlock(tokens_mlp_dim,channels_mlp_dim,tokens_hidden_dim,channels_hidden_dim).to('cuda:0'))
        

    def forward(self,x):
        x=x.unfold(dimension=-1,size=self.patch_size,step=self.patch_size)
        y=self.embd(x) # bs,channels,h,w
        bs,t,c=y.shape
        

        

        for i in range(self.num_blocks):
            y=self.mlp_blocks[i](y) # bs,tokens,channels
        y=self.ln(y) # bs,tokens,channels
        
        
        return y

