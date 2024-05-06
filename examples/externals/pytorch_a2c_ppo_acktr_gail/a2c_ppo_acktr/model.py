import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import copy
from torch.nn.modules import ModuleList

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

from .config_large import cfg
from .tu import *

import IPython


# evogym/examples/externals/pytorch_a2c_ppo_acktr_gail/a2c_ppo_acktr/model.py

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, body, obs_shape=None, action_space=None, env_name='Walker-v0', base=None, base_kwargs=None):
        super(Policy, self).__init__()
        # self.seq_len = cfg.MODEL.MAX_LIMBS
        self.body = body
        self.seq_len = self.body.shape[0] * self.body.shape[1]

        self.get_mask()

        # print(self.body)
        # self.action_space = action_space

        self.env_name = env_name
        
        # print('obs_shape == ', obs_shape)
        # obs_shape ==  (68,)

        ext_dim = self.pre_deal_with_env_name(None, return_sz=True)

        # self.v_net = TransformerModel(obs_shape, 1)
        self.v_net = TransformerModel(self.body, ext_dim = ext_dim)

        # self.mu_net = TransformerModel(obs_shape, 1)
        self.mu_net = TransformerModel(self.body, ext_dim = ext_dim)
        # self.num_actions = cfg.MODEL.MAX_LIMBS * 1
        self.num_actions = self.body.shape[0] * self.body.shape[1]


        if cfg.MODEL.ACTION_STD_FIXED:
            log_std = np.log(cfg.MODEL.ACTION_STD)
            self.log_std = nn.Parameter(
                log_std * torch.ones(1, self.num_actions), requires_grad=False,
            )
        else:
            self.log_std = nn.Parameter(torch.zeros(1, self.num_actions))

    def get_mask(self, ):

        # 根据body产生mask, 这个地方的mask和一般的是反过来的
        def gen_mask(type = 'obs'):
            lst = []
            for x in range(self.body.shape[0]):
                for y in range(self.body.shape[1]):
                    val = True 
                    # 体素类型不为0，不为空白
                    if type == 'obs' and self.body[x][y] != 0:
                        val = False 
                    # 体素类型为3或4，为驱动器
                    if type == 'act' and self.body[x][y] in [3, 4]:
                        val = False 
                    lst.append(val)
            return lst 

        self.obs_mask = gen_mask('obs')
        self.act_mask = gen_mask('act')

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 1

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def pre_deal_with_env_name(self, inputs, return_sz = False):

        if self.env_name == 'Walker-v0':
            return self.pre_deal_with_general(inputs, 2, return_sz = return_sz)

        if self.env_name == 'Carrier-v0':
            return self.pre_deal_with_general(inputs, 6, return_sz = return_sz)

        if self.env_name == 'Balancer-v0':
            return self.pre_deal_with_general(inputs, 1, return_sz = return_sz)

        raise Exception('unsupport env_name !', self.env_name)

    def pre_deal_with_general(self, inputs, head_sz, return_sz = False):

        if return_sz:
            return head_sz

        # IPython.embed()
        batch_size, plen = inputs.shape
        num_points = (plen - head_sz) // 8

        left = inputs[:, :head_sz].unsqueeze(1)
        right = inputs[:, head_sz:].reshape(batch_size, 8, num_points).permute(0, 2, 1)
        # comb = torch.concat([left, right], 1)
        comb = torch.concat([right], 1)

        obs = torch.nn.functional.pad(comb, pad=(0, 0, 0, self.seq_len - num_points , 0, 0), mode='constant', value=0)
        # obs_mask = torch.nn.functional.pad(torch.zeros([batch_size, num_points]).to(inputs.device), pad=(0, self.seq_len - num_points, 0, 0), mode='constant', value=1)
        obs_mask = torch.Tensor(self.obs_mask).unsqueeze(0).repeat(batch_size, 1).to(inputs.device)

        obs = obs.permute([1, 0, 2])

        obs_env = inputs[:, :head_sz]

        # IPython.embed()

        # return obs, obs_mask, None
        return obs, obs_mask, obs_env


    def act(self, inputs, rnn_hxs, masks, deterministic=False):

        # print('act')
        obs, obs_mask, obs_env = self.pre_deal_with_env_name(inputs)

        limb_vals, v_attention_maps = self.v_net(
            obs, obs_mask, obs_env
        )
        limb_vals = limb_vals * (1 - obs_mask.int())
        num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
        val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)


        mu, mu_attention_maps = self.mu_net(
            obs, obs_mask, obs_env
        )
        
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)

        act = pi.sample()
        logp = pi.log_prob(act)
        # act_mask = torch.nn.functional.pad(torch.zeros([inputs.shape[0], self.action_space.shape[0]]), pad=(0, self.seq_len - self.action_space.shape[0], 0, 0), mode='constant', value=1)
        act_mask = torch.Tensor(self.act_mask).unsqueeze(0).repeat(inputs.shape[0], 1).to(inputs.device)
        act_mask = act_mask.bool()
        logp[act_mask] = 0.0

        logp = logp.sum(-1, keepdim=True)

        # IPython.embed()
        # act = act[:, :self.action_space.shape[0]]
        act = torch.masked_select(act, ~act_mask).reshape(inputs.shape[0], -1)

        return val, act, logp, torch.zeros([inputs.shape[0], 1])

    def get_value(self, inputs, rnn_hxs, masks):

        obs, obs_mask, obs_env = self.pre_deal_with_env_name(inputs)
        limb_vals, v_attention_maps = self.v_net(
            obs, obs_mask, obs_env
        )
        limb_vals = limb_vals * (1 - obs_mask.int())
        num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
        val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)

        return val

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        # print('eval')
        # IPython.embed()


        obs, obs_mask, obs_env = self.pre_deal_with_env_name(inputs)
        limb_vals, v_attention_maps = self.v_net(
            obs, obs_mask, obs_env
        )
        limb_vals = limb_vals * (1 - obs_mask.int())
        num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
        val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)


        mu, mu_attention_maps = self.mu_net(
            obs, obs_mask, obs_env
        )
        
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)

        # print('dbg 05')
        # IPython.embed()
        action = torch.nn.functional.pad(action, pad=(0, self.seq_len - action.shape[1], 0, 0), mode='constant', value=0)
        logp = pi.log_prob(action)

        # act_mask = torch.nn.functional.pad(torch.zeros([inputs.shape[0], self.action_space.shape[0]]), pad=(0, self.seq_len - self.action_space.shape[0], 0, 0), mode='constant', value=1)
        act_mask = torch.Tensor(self.act_mask).unsqueeze(0).repeat(inputs.shape[0], 1).to(inputs.device)
        act_mask = act_mask.bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        entropy = pi.entropy()
        entropy[act_mask] = 0.0
        entropy = entropy.mean()

        return val, logp, entropy, torch.zeros([inputs.shape[0], 1])
        # return value, action_log_probs, dist_entropy, rnn_hxs



# J: Max num joints between two limbs. 1 for 2D envs, 2 for unimal
class TransformerModel(nn.Module):
    def __init__(self, body, obs_shape = None, decoder_out_dim = 1, ext_dim = 0):
        super(TransformerModel, self).__init__()

        self.model_args = cfg.MODEL.TRANSFORMER
        # self.seq_len = cfg.MODEL.MAX_LIMBS
        self.seq_len = body.shape[0] * body.shape[1] 

        # Embedding layer for per limb obs
        self.d_model = cfg.MODEL.LIMB_EMBED_SIZE
        self.limb_embed = nn.Linear(8, self.d_model)

        seq_len = self.seq_len
        self.pos_embedding = PositionalEncoding(self.d_model, seq_len)

        self.hfield_encoder = MLPObsEncoder(2)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayerResidual(
            cfg.MODEL.LIMB_EMBED_SIZE,
            self.model_args.NHEAD,
            self.model_args.DIM_FEEDFORWARD,
            self.model_args.DROPOUT,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.model_args.NLAYERS, norm=None,
        )

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model + ext_dim
        # decoder_input_dim = self.d_model


        # self.decoder = nn.Linear(decoder_input_dim, decoder_out_dim)
        self.decoder = make_mlp_default(
            [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim],
            final_nonlinearity=False,
        )
        self.init_weights()

    def init_weights(self):
        initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
        self.limb_embed.weight.data.uniform_(-initrange, initrange)
        self.decoder[-1].bias.data.zero_()
        initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self, obs, obs_mask, obs_env=None, obs_cm_mask=None, return_attention=False):
        # (num_limbs, batch_size, limb_obs_size) -> (num_limbs, batch_size, d_model)
        # IPython.embed()

        # obs.shape == torch.Size([25, 4, 1])
        # IPython.embed()
        obs_embed = self.limb_embed(obs) * math.sqrt(self.d_model)
        _, batch_size, _ = obs_embed.shape

        if obs_env is not None:
            hfield_obs = self.hfield_encoder(obs_env)
            hfield_obs = hfield_obs.repeat(self.seq_len, 1)
            hfield_obs = hfield_obs.reshape(self.seq_len, batch_size, -1)
            # hfield_obs.shape == torch.Size([36, 4, 2])

        attention_maps = None
        obs_embed = self.pos_embedding(obs_embed)

        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                obs_embed, src_key_padding_mask=obs_mask
            )
        else:
            # (num_limbs, batch_size, d_model)
            obs_embed_t = self.transformer_encoder(
                obs_embed, src_key_padding_mask=obs_mask
            )

        decoder_input = obs_embed_t
        # decoder_input.shape == torch.Size([36, 4, 128])

        if obs_env is not None:
            decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)
            # decoder_input.shape == torch.Size([36, 4, 130])


        # (num_limbs, batch_size, J)
        output = self.decoder(decoder_input)
        # (batch_size, num_limbs, J)
        output = output.permute(1, 0, 2)
        # (batch_size, num_limbs * J)
        output = output.reshape(batch_size, -1)

        # output.shape == torch.Size([4, 25])
        return output, attention_maps


class MLPObsEncoder(nn.Module):
    """Encoder for env obs like hfield."""

    def __init__(self, obs_dim):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS
        self.encoder = make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)



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


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for l in self.layers:
            output = l(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def get_attention_maps(self, src, mask=None, src_key_padding_mask=None):
        attention_maps = []
        output = src

        for l in self.layers:
            # NOTE: Shape of attention map: Batch Size x MAX_JOINTS x MAX_JOINTS
            # pytorch avgs the attention map over different heads; in case of
            # nheads > 1 code needs to change.
            output, attention_map = l(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=True
            )
            attention_maps.append(attention_map)

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_maps


class TransformerEncoderLayerResidual(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerEncoderLayerResidual, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayerResidual, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attention=False):
        src2 = self.norm1(src)
        src2, attn_weights = self.self_attn(
            src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        if return_attention:
            return src, attn_weights
        else:
            return src
