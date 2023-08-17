import torch
import torch.nn as nn
from models.Dual_attention import Dual_Attenion
class Model(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.d_model = config.d_model
        # self.task_name = config.task_name
        patch_num = int((config.seq_len - self.patch_len)/self.stride + 1)
        self.patch_num = patch_num
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num,config.d_model)*1e-2)
        self.total_token_number = self.patch_num + 1
        config.total_token_number = self.total_token_number
        # embeding layer related
        self.W_input_projection = nn.Linear(self.patch_len, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.cls = nn.Parameter(torch.randn(1,config.d_model)*1e-2)
        # mlp decoder
        self.W_out = nn.Linear((self.total_token_number)*config.d_model, config.pred_len)
        # dual attention encoder related
        self.Attentions_over_token = nn.ModuleList([Dual_Attenion(config) for i in range(config.e_layers)])
        self.Attentions_over_channel = nn.ModuleList([Dual_Attenion(config,over_channel = True) for i in range(config.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(config.d_model,config.d_model) for i in range(config.
        e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(config.dropout) for i in range(config.e_layers)])
        self.Attentions_norm = nn.ModuleList([nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model), Transpose(1,2)) for i in range(config.e_layers)])
    def forward(self, z, *args, **kwargs):
        z=z.permute(0,2,1)
        b,c,s = z.shape
        # inputs nomralization
        z_mean = torch.mean(z,dim = (-1),keepdims = True)
        z_std = torch.std(z,dim = (-1),keepdims = True)
        z = (z - z_mean)/(z_std + 1e-4)
        #transpose
        
        # tokenization
        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_embed = self.input_dropout(self.W_input_projection(zcube))+ self.W_pos_embed
        cls_token = self.cls.repeat(z_embed.shape[0],z_embed.shape[1],1,1)
        z_embed = torch.cat((cls_token,z_embed),dim = -2)
        # dual attention encoder
        inputs = z_embed
        b,c,t,h = inputs.shape
        for a_2,a_1,mlp,drop,norm in zip(self.Attentions_over_token, self.Attentions_over_channel,self.Attentions_mlp ,self.Attentions_dropout,self.Attentions_norm ):
            output_1 = a_1(inputs)
            output_2 = a_2(output_1)
            outputs = drop(mlp(output_1+output_2))+inputs
            outputs = norm(outputs.reshape(b*c,t,-1)).reshape(b,c,t,-1)
            inputs = outputs
        # mlp decoder
        z_out = self.W_out(outputs.reshape(b,c,-1))
        # denomrlaization
        z = z_out *(z_std+1e-4) + z_mean
        z=z.permute(0,2,1)
        return z


class Transpose(nn.Module):
    def __init__(self,dim1,dim2):
        super().__init__()
        self.dim1=dim1
        self.dim2=dim2
    def forward(self,x):
        return x.permute(0,self.dim2,self.dim1)