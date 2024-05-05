import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
from torch.distributions.normal import Normal
from .hgtmodel import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
def w_init(module, gain=1):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module

def make_mlp(dim_list):
    init_ = lambda m: w_init(m)

    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(init_(nn.Linear(dim_in, dim_out)))
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def make_mlp_default(dim_list, final_nonlinearity=True, nonlinearity="relu"):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if nonlinearity == "relu":
            layers.append(nn.ReLU())
        elif nonlinearity == "tanh":
            layers.append(nn.Tanh())

    if not final_nonlinearity:
        layers.pop()
    return nn.Sequential(*layers)

def num_params(model, only_trainable=True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = model.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)



class TransformerModel(nn.Module):
    def __init__(self, obs_shape = None, decoder_out_dim=1): 
        super(TransformerModel, self).__init__()

        self.seq_len = 5*5  


        self.d_model = 128
        #self.limb_embed = nn.Linear(2,self.d_model)   
        #self.ext_feat_fusion = "none"

        seq_len = self.seq_len
        #self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
        self.hfield_encoder = MLPObsEncoder(2)
        self.gnn = GNN(in_dim = 8, n_hid = self.d_model, n_heads = 2, n_layers = 3, dropout = 0.2,num_types = 5, num_relations =  1).to(device)
        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model+64
        self.norm      = nn.LayerNorm(self.d_model)
        # self.decoder = nn.Linear(decoder_input_dim, decoder_out_dim)
        self.decoder = make_mlp_default(
            [decoder_input_dim] + [64] + [decoder_out_dim],
            final_nonlinearity=False,
        )
        self.init_weights()

        
    def init_weights(self):
        initrange = 0.1
        #self.limb_embed.weight.data.uniform_(-initrange, initrange)
        self.decoder[-1].bias.data.zero_()
        initrange = 0.01
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self, structure, obs_env, local_obs, obs_cm_mask=None, return_attention=False):  
        batch_size, num_nodes,feature_dim = local_obs.shape
        local_obs=local_obs.permute(1,0,2)  #local_obs=(25,batch_size,feature_dim)#
        #structure_body = torch.tensor(structure[0], dtype=torch.float32).to(device)
        structure_connect = structure[1]
        structure_body = structure[0]
        structure_body_f = structure_body.flatten()   
        structure_connect = structure_connect.T   
        node_features = {'0.0': [], '1.0': [], '2.0': [], '3.0': [], '4.0': []}
        change_body = torch.Tensor(list(range(25)))  
        a = b = c = d = e = 0
        length = []
        for node_idx in range(num_nodes):
            feature_list = local_obs[node_idx].tolist()
            node_features[str(structure_body_f[node_idx])].append(feature_list)   
        for key,value in node_features.items():
            length.append(len(value))
        for i in range(25):
            if structure_body_f[i] == 0.0:
                change_body[a] = i
                a = a+1
            elif structure_body_f[i] == 1.0:
                change_body[length[0]+b] = i
                b = b+1
            elif structure_body_f[i] == 2.0:
                change_body[length[0]+length[1]+c] = i
                c = c+1
            elif structure_body_f[i] == 3.0:
                change_body[length[0]+length[1]+length[2]+d] = i
                d = d+1
            else: 
                change_body[length[0]+length[1]+length[2]+length[3]+e] = i
                e = e+1
        rechange = [None for _ in range(25)]   
        for i in range(len(change_body)):
            for a in range(len(change_body)):
                if change_body[a]==i:
                    rechange[i]=a 
        node_feature = torch.cat((torch.tensor(node_features['0.0']), # features of each node type is of shape (num_nodes, batch_size, feature_dim)
                        torch.tensor(node_features['1.0']),
                        torch.tensor(node_features['2.0']),
                        torch.tensor(node_features['3.0']),
                        torch.tensor(node_features['4.0'])), dim=0) # node_features is of shape (25, batch_size, feature_dim)
        node_type    = []
        node_dict = {}
        node_num = 0
        types = ['0.0','1.0','2.0','3.0','4.0']
        for t in types:
            node_dict[t] = [node_num, len(node_dict)]
            node_num     += len(node_features[t])
        for t in types:
            node_type    += [node_dict[t][1] for _ in range(len(node_features[t]))]  
        node_type    = torch.LongTensor(node_type).to(device)
        N=15
        W = torch.zeros(5, 25, N)  
        W[0,0:length[0], 0:length[0]] = torch.eye(length[0],length[0])
        W[1,length[0]:length[0]+length[1], 0:length[1]] = torch.eye(length[1],length[1])
        W[2,length[0]+length[1]:length[0]+length[1]+length[2], 0:length[2]] = torch.eye(length[2],length[2])
        W[3,length[0]+length[1]+length[2]:length[0]+length[1]+length[2]+length[3], 0:length[3]] = torch.eye(length[3],length[3])
        W[4,length[0]+length[1]+length[2]+length[3]:length[0]+length[1]+length[2]+length[3]+length[4], 0:length[4]] = torch.eye(length[4],length[4])
        WW = torch.unsqueeze(W, dim=0)
        WW = W.repeat(batch_size, 1, 1, 1).to(device)
        mask_matrix = W.permute(0,2,1).reshape(5*N, 25)
        mask_matrix = mask_matrix.repeat(batch_size, 1, 1).to(device)
        R = torch.full((25,25), float('-inf'))      
        attn_maskup = torch.zeros(25,25)
        attn_maskdown = torch.zeros(25,25)
        attn_maskleft = torch.zeros(25,25)
        attn_maskright = torch.zeros(25,25)
        for i in range(5):
            for j in range(5):
                if i>0:
                    attn_maskup[i*5+j][(i-1)*5+j] = 1
                if i<4:
                    attn_maskdown[i*5+j][(i+1)*5+j] = 1
                if j>0:
                    attn_maskleft[i*5+j][i*5+j-1] = 1
                if j<4:
                    attn_maskright[i*5+j][i*5+j+1] = 1
        attn_maskup = attn_maskup[change_body.tolist()]
        attn_maskup = attn_maskup[:,change_body.tolist()]                     
        attn_maskdown = attn_maskdown[change_body.tolist()]
        attn_maskdown = attn_maskdown[:,change_body.tolist()]    
        attn_maskleft = attn_maskleft[change_body.tolist()]
        attn_maskleft = attn_maskleft[:,change_body.tolist()]
        attn_maskright = attn_maskright[change_body.tolist()]
        attn_maskright = attn_maskright[:,change_body.tolist()]
        attn_mask = torch.cat((attn_maskup,attn_maskdown,attn_maskleft,attn_maskright),dim=1)        
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for row in range(5):
            for i in range(5):
                x=5*row+i
                for offset in offsets:
                    new_row = row + offset[0]
                    new_col = i + offset[1]
                    if 0 <= new_row < 5 and 0 <= new_col < 5 and structure_body[new_row][new_col] != 0:
                       R[x,5*new_row+new_col] = 0
        R_change = R[change_body.tolist()]   
        R_change = R_change[:,change_body.tolist()]  
        R_change = torch.unsqueeze(R_change, dim=0)
        R_change = R_change.repeat(batch_size, 1, 1)
        R_change = R_change.unsqueeze(1).expand(-1, 2, -1, -1) .to(device)               
        obs_rep, attention_maps = self.gnn.forward(node_feature.to(device), WW,mask_matrix,R_change,attn_mask.to(device),node_type,change_body,rechange)  
        obs_rep = obs_rep[:,:,rechange]   
        obs_rep=obs_rep.permute(2, 1, 0)  #(25,batch_size,feature_dim)#
        obs_rep = self.norm(obs_rep)
        for i in range(len(attention_maps)):
            attention_map = attention_maps[i]
            attention_map = attention_map[rechange]
            attn_list = []
            for j in range(4):
                tensor = attention_map[:,j*25:(j+1)*25]
                attention = tensor[:, rechange]
                attn_list.append(attention)
            attn = torch.cat((attn_list[0],attn_list[1],attn_list[2],attn_list[3]),dim=1)
            attention_maps[i] = attn    
        if obs_env is not None:    
            hfield_obs = self.hfield_encoder(obs_env)
            hfield_obs = hfield_obs.repeat(self.seq_len, 1)
            hfield_obs = hfield_obs.reshape(self.seq_len, batch_size, -1)
        decoder_input = torch.cat([obs_rep, hfield_obs], axis=2)   
        # (num_limbs, batch_size, J)
        output = self.decoder(decoder_input)
        # (batch_size, num_limbs, J)
        output = output.permute(1, 0, 2)
        # (batch_size, num_limbs * J)
        output = output.reshape(batch_size, -1)
        return output, attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)



class MLPObsEncoder(nn.Module):

    def __init__(self, obs_dim):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + [64,64]
        self.encoder = make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)




class ActorCritic(nn.Module):
    def __init__(self,):
        super(ActorCritic, self).__init__()
        self.seq_len = 5*5
        self.v_net = TransformerModel()        
        self.mu_net = TransformerModel()
        self.num_actions = 5 * 5 * 1
        log_std = np.log(0.9)
        self.log_std = nn.Parameter(
            log_std * torch.ones(1, self.num_actions), requires_grad=False,
        )
    

    def pre_deal_with_walker_v0(self,inputs):

        # IPython.embed()
        batch_size, plen = inputs.shape
        #print("inputs.shape:",inputs.shape)
        num_points = (plen - 2) // 2

        left = inputs[:, :2].unsqueeze(1)
        right = inputs[:, 2:].reshape(batch_size, 2, num_points).permute(0, 2, 1)
        comb = torch.concat([left, right], 1)
        
        obs = torch.nn.functional.pad(comb, pad=(0, 0, 0, 5*5 - num_points - 1, 0, 0), mode='constant', value=0)
        obs_mask = torch.nn.functional.pad(torch.zeros([batch_size, num_points]).to(inputs.device), pad=(0, 5*5 - num_points, 0, 0), mode='constant', value=1)

        obs = obs.permute([1, 0, 2])
        #obs_mask = obs_mask.permute([1, 0])
        # IPython.embed()
        obs_env = inputs[:, :2]
        return obs, obs_mask, obs_env
       
    def forward(self, structure,inputs,act=None, return_attention=False):  
        local_obs = inputs[:, 2:].reshape(inputs.shape[0], 8, 5 ** 2).permute(0, 2, 1)  
        _, _, obs_env = self.pre_deal_with_walker_v0(inputs)   
        obs_mask = torch.tensor(structure[0].flatten()==0).unsqueeze(0).repeat(inputs.shape[0],1).to(device)
        limb_vals, v_attention_maps = self.v_net(      
            structure, obs_env, local_obs
        )

        limb_vals = limb_vals * (1 - obs_mask.int())

        num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
        val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)  

        mu, mu_attention_maps = self.mu_net(       
            structure, obs_env, local_obs
        )
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)
        return val, pi, v_attention_maps


class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic 
    
            
    @torch.no_grad()
    def act_test(self, structure,inputs,action_space):
        
        val, pi, v_attention_maps = self.ac(structure,inputs)
        act = pi.sample()
        logp = pi.log_prob(act)
        act_mask_3 = torch.tensor(structure[0].flatten()==3)
        act_mask_4 = torch.tensor(structure[0].flatten()==4)
        act_mask = ((act_mask_3 + act_mask_4)==0).unsqueeze(0).repeat(inputs.shape[0],1).to(device)
        #pad_width = (0, 5*5 - action_space.shape[0], 0, 0)
        #act_mask = torch.nn.functional.pad(torch.zeros([inputs.shape[0], action_space.shape[0]]), pad=pad_width, mode='constant', value=1)
        #act_mask = act_mask.bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        
        return val, act, logp, v_attention_maps  

    @torch.no_grad()
    def get_value_test(self, structure,inputs):
        
        val, _, _ = self.ac(structure,inputs)
        return val
    
    def evaluate_actions_test(self, structure,inputs, action_space,rnn_hxs, masks, action):
        
        val, pi,_ = self.ac(structure,inputs)
        action = torch.nn.functional.pad(action, pad=(0, 5*5 - action.shape[1], 0, 0), mode='constant', value=0)
        logp = pi.log_prob(action)
        
        act_mask_3 = torch.tensor(structure[0].flatten()==3)
        act_mask_4 = torch.tensor(structure[0].flatten()==4)
        act_mask = ((act_mask_3 + act_mask_4)==0).unsqueeze(0).repeat(inputs.shape[0],1).to(device)
        #pad_width = (0, 5*5 - action_space.shape[0], 0, 0)
        #act_mask = torch.nn.functional.pad(torch.zeros([inputs.shape[0], action_space.shape[0]]), pad=pad_width, mode='constant', value=1)        
        #act_mask = act_mask.bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
                    
        entropy = pi.entropy()
        entropy[act_mask] = 0.0
        entropy = entropy.mean()

        return val, logp, entropy, torch.zeros([inputs.shape[0], 1])