import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .Transformer import TransformerEncoder
from .Transformer import TransformerEncoderLayerResidual

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
    def __init__(self, obs_space, decoder_out_dim):
        super(TransformerModel, self).__init__()
        #嵌入层#
        self.model_args = cfg.MODEL.TRANSFORMER  # 传入transformer的具体配置参数#
        self.seq_len = 8  # 在所有环境中肢体的最大数量#

        #每个肢体的嵌入#
        limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.d_model = 128
        self.limb_embed = nn.Linear(limb_obs_size, self.d_model)
        self.ext_feat_fusion = "none"

        seq_len = self.seq_len
        self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
       
        # Transformer Encoder 编码器层#
        encoder_layers = TransformerEncoderLayerResidual(
            self.d_model,
            2,
            1024,
            0.0,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, 5, norm=None,
        )

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model

        # Task based observation encoder
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:           #将外界环境表征利用MLP提取出来?#         
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])
            

        if self.ext_feat_fusion == "late":
            decoder_input_dim += self.hfield_encoder.obs_feat_dim

        # self.decoder = nn.Linear(decoder_input_dim, decoder_out_dim)
        self.decoder = make_mlp_default(
            [decoder_input_dim] + [] + [decoder_out_dim],
            final_nonlinearity=False,
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.limb_embed.weight.data.uniform_(-initrange, initrange)
        self.decoder[-1].bias.data.zero_()
        initrange = 0.01
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, return_attention=False):
        # (num_limbs, batch_size, limb_obs_size) -> (num_limbs, batch_size, d_model)
        obs_embed = self.limb_embed(obs) * math.sqrt(self.d_model)
        _, batch_size, _ = obs_embed.shape

        if "hfield" in cfg.ENV.KEYS_TO_KEEP:    #?#
            # (batch_size, embed_size)
            hfield_obs = self.hfield_encoder(obs_env["hfield"])

        if self.ext_feat_fusion in ["late"]:
            hfield_obs = hfield_obs.repeat(self.seq_len, 1)
            hfield_obs = hfield_obs.reshape(self.seq_len, batch_size, -1)

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
        if "hfield" in cfg.ENV.KEYS_TO_KEEP and self.ext_feat_fusion == "late":   #?#
            decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)

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


class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

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
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__()
        self.seq_len = 8
        self.v_net = TransformerModel(obs_space, 1)

        self.mu_net = TransformerModel(obs_space, 2)
        self.num_actions = 8 * 2

        log_std = np.log(0.9)
        self.log_std = nn.Parameter(
            log_std * torch.ones(1, self.num_actions), requires_grad=False,
        )

    def forward(self, obs, act=None, return_attention=False):
        batch_size = 128

        obs_env = {k: obs[k] for k in cfg.ENV.KEYS_TO_KEEP}
        if "obs_padding_cm_mask" in obs:
            obs_cm_mask = obs["obs_padding_cm_mask"]
        else:
            obs_cm_mask = None
        obs, obs_mask, act_mask, _ = (
            obs["proprioceptive"],
            obs["obs_padding_mask"],
            obs["act_padding_mask"],
            obs["edges"],
        )

        obs_mask = obs_mask.bool()
        act_mask = act_mask.bool()

        obs = obs.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)

        limb_vals, v_attention_maps = self.v_net(
            obs, obs_mask, obs_env, obs_cm_mask, return_attention=return_attention
        )

        limb_vals = limb_vals * (1 - obs_mask.int())

        num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
        val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)

        mu, mu_attention_maps = self.mu_net(
            obs, obs_mask, obs_env, obs_cm_mask, return_attention=return_attention
        )
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)

        if act is not None:
            logp = pi.log_prob(act)
            logp[act_mask] = 0.0
            logp = logp.sum(-1, keepdim=True)
            entropy = pi.entropy()
            entropy[act_mask] = 0.0
            entropy = entropy.mean()
            return val, pi, logp, entropy
        else:
            if return_attention:
                return val, pi, v_attention_maps, mu_attention_maps
            else:
                return val, pi, None, None


class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic

    @torch.no_grad()
    def act(self, obs):
        val, pi, _, _ = self.ac(obs)
        act = pi.sample()
        logp = pi.log_prob(act)
        act_mask = obs["act_padding_mask"].bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs):
        val, _, _, _ = self.ac(obs)
        return val


