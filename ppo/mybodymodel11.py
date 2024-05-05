import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .Transformer import TransformerEncoder
from .Transformer import TransformerEncoderLayerResidual

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        self.feature_size = 8
        self.d_model = 128
        #self.limb_embed = nn.Linear(2,self.d_model)   #2#
        #self.ext_feat_fusion = "none"

        #seq_len = self.seq_len
        #self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
        #self.hfield_encoder = MLPObsEncoder(2)
        self.encoder = nn.Linear(self.feature_size+2, self.d_model)
        encoder_layers = TransformerEncoderLayerResidual(
            self.d_model,
            2,
            256,
            0.5,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, 3, norm=nn.LayerNorm(self.d_model),
        )

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model

        # self.decoder = nn.Linear(decoder_input_dim, decoder_out_dim)
        self.decoder = make_mlp_default(
            [decoder_input_dim] + [] + [decoder_out_dim],
            final_nonlinearity=False,
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder[-1].bias.data.zero_()
        initrange = 0.01
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self, structure,obs_env, local_obs, obs_cm_mask=None, return_attention=False):
        batch_size, num_nodes,_ = local_obs.shape 
        local_obs=local_obs.permute(1,0,2)
        env = torch.unsqueeze(obs_env, dim=1)
        env = env.repeat(1, num_nodes, 1).to(device)
        env = env.permute(1,0,2)
        obs = torch.cat((local_obs, env), dim=2)
        encoded = self.encoder(obs)*math.sqrt(self.d_model)
        output, attention_maps = self.transformer_encoder.get_attention_maps(encoded)
        output = self.decoder(output).to(device)
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
        mlp_dims = [obs_dim] + []
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
        obs_env = inputs[:, :6]
        return obs, obs_mask, obs_env
       
    def forward(self, structure,inputs,act=None, return_attention=False):  
        
        local_obs = inputs[:, 2:].reshape(inputs.shape[0], 8, 5 ** 2).permute(0, 2, 1)    
        #_, _, obs_env = self.pre_deal_with_walker_v0(inputs)    
        obs_env = inputs[:, :2]
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
        act_mask = ((act_mask_3 + act_mask_4)==0).unsqueeze(0).repeat(inputs.shape[0],1)
        #pad_width = (0, 5*5 - action_space.shape[0], 0, 0)
        #act_mask = torch.nn.functional.pad(torch.zeros([inputs.shape[0], action_space.shape[0]]), pad=pad_width, mode='constant', value=1)
        #act_mask = act_mask.bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        
        return val, act, logp,v_attention_maps
    @torch.no_grad()
    def get_value_test(self, structure,inputs):
        
        val, _, _ = self.ac(structure,inputs)
        return val
    
    def evaluate_actions_test(self, structure,inputs, action_space,rnn_hxs, masks, action):
        
        val, pi, _ = self.ac(structure,inputs)
        action = torch.nn.functional.pad(action, pad=(0, 5*5 - action.shape[1], 0, 0), mode='constant', value=0)
        logp = pi.log_prob(action)
        
        act_mask_3 = torch.tensor(structure[0].flatten()==3)
        act_mask_4 = torch.tensor(structure[0].flatten()==4)
        act_mask = ((act_mask_3 + act_mask_4)==0).unsqueeze(0).repeat(inputs.shape[0],1)
        #pad_width = (0, 5*5 - action_space.shape[0], 0, 0)
        #act_mask = torch.nn.functional.pad(torch.zeros([inputs.shape[0], action_space.shape[0]]), pad=pad_width, mode='constant', value=1)        
        #act_mask = act_mask.bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
                    
        entropy = pi.entropy()
        entropy[act_mask] = 0.0
        entropy = entropy.mean()

        return val, logp, entropy, torch.zeros([inputs.shape[0], 1])