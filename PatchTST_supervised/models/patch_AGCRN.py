import torch
import torch.nn as nn
import sys
sys.path.append("/data/jiahuili/AGCRN/")
import torch.nn.functional as F
from einops import rearrange
import einops
from torch import einsum

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self,configs,context_window=336, patch_len=16,stride=16,learn_pe=True,d_model=128,dropout=0):
        super(Model, self).__init__()
        self.configs=configs
        self.backbone_trend=patch_AGCRN_backbone(configs)
        self.backbone_seasonal=patch_AGCRN_backbone(configs)
        self.backbone_noise_1=patch_AGCRN_backbone(configs)
        self.backbone_noise_2=patch_AGCRN_backbone(configs)
        # self.affine_weight = nn.Parameter(torch.ones(37))
        # self.affine_bias = nn.Parameter(torch.zeros(37))
        #decomposition
        # kernel_size=1001
        # self.decompsition = series_decomp(kernel_size)
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        
        # self.noise_1=repeat(self.noise_1,'B T D ->(r1 B) T (r2 D)',r1=256, r2=37)
      
        # self.backbone_retrival=patch_RNN_backbone(configs)

    def forward(self, x,batch_x_mark=0, dec_inp=0, batch_y_mark=0, batch_y=0) :
        # knn_set=get_k_neightbor(x,retrival_dataset)
        # x: [Batch, Input length, Channel]
        
        # noise_1=repeat(self.noise_1,'B T D ->(r1 B) T (r2 D)',r1=256, r2=x.shape[2])
        x = self.revin_layer(x, 'norm')
        # seasonal_x, trend_x = self.decompsition(x)
        trend_x=torch.mean(x,dim=1,keepdim=True)
        trend_x=einops.repeat(trend_x,'h w n -> h (repeat w) n',repeat=self.configs.seq_len)
        seasonal_x=x-trend_x

        # stdev=torch.std(seasonal_x,dim=1)
        # noise_1=noise_1[:x.shape[0],:,:]

        # noise_1=einsum('B T D, B D ->B T D',noise_1,stdev)
        # seasonal_x=seasonal_x-noise_1
        
        

        # knn_set=knn_set[:,:,11:]
        # noise_1=self.backbone_noise_1(noise_1,support,dec)
        # output=self.backbone_trend(x,support,dec,knn_set)
        trend=self.backbone_trend(trend_x,dec_inp)
        seasonal=self.backbone_seasonal(seasonal_x,dec_inp)
        # output=seasonal+trend+noise_1
        output=seasonal+trend
        # knn_set=self.backbone_retrival(knn_set[:,:,11:],knn_set[:,:,:11],dec)
        # output=output+knn_set

        output = self.revin_layer(output, 'denorm')
        
        
       
        return output # [Batch, Output length, Channel]
    
class patch_AGCRN_backbone(nn.Module):
    def __init__(self, args):
        super(patch_AGCRN_backbone, self).__init__()
        self.args=args
        self.target_len=args.pred_len
        self.decoder_len=args.dec_in
        # self.num_node = 1
        # self.input_dim = args.input_dim
        self.hidden_dim = 16
        # self.output_dim = args.output_dim
        # self.horizon = args.horizon
        self.num_layers = args.e_layers
        # self.device=args.device
        # self.default_graph = args.default_graph
        # self.enc_len=args.enc_len
        
        
        
        # self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.node_embeddings = nn.Parameter(torch.randn(self.decoder_len, 2), requires_grad=True)
        # self.time_embeddings = nn.Parameter(torch.randn(self.enc_len, args.embed_dim), requires_grad=True)
        self.encoder = AVWDCRNN(self.decoder_len, 16,self.hidden_dim, 2,
                                2, self.num_layers)
        # self.embedding=nn.Linear(11,self.hidden_dim)
        self.decoder = AVWDCRNN(self.decoder_len, 16, self.hidden_dim, 2,
                                 2, self.num_layers)
        #encoder, decoder for seasonal and trend
        self.patch_len=args.patch_len
        self.stride=args.stride

        self.project=nn.Linear(89*self.hidden_dim,self.target_len)
        self.flatten = nn.Flatten(start_dim=-2)
        # self.short=nn.Linear(args.enc_len,args.dec_len)
        
        self.dense=nn.Linear(self.hidden_dim,self.hidden_dim)
        # self.expand=nn.Linear(11,37)
    
    def forward(self, x, y=None):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        
        
        # x=self.revin_layer(x, 'norm')
        x=x.permute(0,2,1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) 
        x =rearrange(x,'b d t n-> b t d n')
        init_state = self.encoder.init_hidden(x.shape[0])
        x,h=self.encoder(x,init_state,self.node_embeddings)
        

        h=torch.stack(h,dim=0)
        h=self.dense(h)
        
        
        y=y.permute(0,2,1)
        y = y.unfold(dimension=-1, size=self.patch_len, step=self.stride) 
        y =rearrange(y,'b d t n-> b t d n')
        x,h=self.decoder(y,h,self.node_embeddings)
        
        x=rearrange(x,'b t d n-> b d t n')
        x=self.flatten(x)
        x=self.project(x)
        x=x.permute(0,2,1)
        
        

        

        



        
        return x



class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


    






class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
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
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
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
            x = x + self.mean
        return x