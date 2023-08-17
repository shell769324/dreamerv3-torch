import numpy as np
from einops import rearrange, repeat
from envs.crafter import targets

import torch
from torch import nn
from torch import distributions as torchd

import tools

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x, q2 = x
        return self.fn((self.norm(x), self.norm(q2)), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.net2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x, q2 = x
        x, q = self.net(x) + x, self.net2(q2) + q2
        print("linear", x.abs().mean(), q.abs().mean())
        return x, q


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.LogSoftmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x, qoir = x
        q2 = rearrange(qoir, 'b n (h d) -> b h n d', h=self.heads)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # b h 1 d
        dots2 = torch.matmul(q2, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn2 = self.attend(dots2)

        out = torch.matmul(attn, v)
        qout = torch.matmul(attn2, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        qout = rearrange(qout, 'b h n d -> b n (h d)')
        x, q = self.to_out(out) + x, qout + qoir
        print("attention", x.abs().mean().item(), q.abs().mean().item())
        return x, q


class MixedHead(nn.Module):
    def __init__(
        self,
        stoch_size,
        deter_size,
        embed_dim,
        attention_dim,
        shape,
        layers,
        dist="normal",
        std=1.0,
        device="cuda",
        heads=8
    ):
        super(MixedHead, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if len(self._shape) == 0:
            self._shape = (1,)
        self.layers = []
        self._dist = dist
        self._std = std
        self._device = device
        self.heads = heads
        self.embedding = nn.Embedding(len(targets), embed_dim)
        self.stoch_layer = nn.Sequential(nn.Linear(stoch_size, attention_dim * 2, bias=True),
                                          nn.GELU(),
                                          nn.Linear(attention_dim * 2, attention_dim, bias=True))
        self.deter_layer = nn.Sequential(nn.Linear(deter_size, attention_dim * 2, bias=True),
                                         nn.GELU(),
                                         nn.Linear(attention_dim * 2, attention_dim, bias=True))
        self.stoch_layer.apply(tools.weight_init)
        self.deter_layer.apply(tools.weight_init)
        self.attention_dim = attention_dim

        for index in range(layers):
            self.layers.append(PreNorm(attention_dim, Attention(attention_dim, heads=heads)))
            self.layers.append(PreNorm(attention_dim, FeedForward(attention_dim, attention_dim * 2)))
        self.layers = nn.Sequential(*self.layers)
        self.layers.apply(tools.weight_init)

        self.mean_layer = nn.Linear(attention_dim, np.prod(self._shape))
        self.mean_layer.apply(tools.weight_init)

        if self._std == "learned":
            self.std_layer = nn.Linear(self._units, np.prod(self._shape))
            self.std_layer.apply(tools.weight_init)

    def __call__(self, stoch, deter, targets_array, dtype=None):
        original = deter.shape
        stoch = stoch.reshape(-1, stoch.shape[-1])
        deter = deter.reshape(-1, deter.shape[-1])
        targets_array = targets_array.reshape(-1)
        token1 = self.stoch_layer(stoch).unsqueeze(-2)
        token2 = self.deter_layer(deter).unsqueeze(-2)
        feature = torch.cat([token1, token2], dim=-2)
        # b h 1 d
        print("before layers", feature.abs().mean())
        (_, out) = self.layers((feature, self.embedding(targets_array).unsqueeze(-2)))
        out = out.reshape(original[0], original[1], -1)

        print("before mean", out[0])
        mean = self.mean_layer(out)
        print("after mean", mean[0][:, 122:131])
        print("prob", torch.softmax(mean, -1)[0][:, 122:131])
        if self._std == "learned":
            std = self.std_layer(out)
        else:
            std = self._std
        if self._dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(
                    torchd.normal.Normal(mean, std), len(self._shape)
                )
            )
        if self._dist == "huber":
            return tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape)
                )
            )
        if self._dist == "binary":
            return tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(1/(torch.pow(torch.e, -mean) + 1)), len(self._shape)
                )
            )
        if self._dist == "twohot_symlog":
            res = tools.TwoHotDistSymlog(logits=mean, device=self._device)
            print("res mean", res.mean().squeeze(-1))
            return res
        raise NotImplementedError(self._dist)

class A2C(nn.Module):
    def __init__(
        self,
        stoch_size,
        deter_size,
        embed_dim,
        attention_dim,
        num_action,
        shape,
        layer_count,
        unimix_ratio=0.01,
        heads=8
    ):
        super(A2C, self).__init__()
        self._unimix_ratio = unimix_ratio

        self.heads = heads
        self.embedding = nn.Embedding(len(targets), embed_dim)
        self.stoch_layer = nn.Sequential(nn.Linear(stoch_size, attention_dim * 2, bias=True),
                                         nn.GELU(),
                                         nn.Linear(attention_dim * 2, attention_dim, bias=True))
        self.deter_layer = nn.Sequential(nn.Linear(deter_size, attention_dim * 2, bias=True),
                                         nn.GELU(),
                                         nn.Linear(attention_dim * 2, attention_dim, bias=True))
        self.stoch_layer.apply(tools.weight_init)
        self.deter_layer.apply(tools.weight_init)
        self.attention_dim = attention_dim
        self.layers = []
        self.shape = shape
        for index in range(layer_count):
            self.layers.append(PreNorm(attention_dim, Attention(attention_dim, heads=heads)))
            self.layers.append(PreNorm(attention_dim, FeedForward(attention_dim, attention_dim * 2)))
        self.layers = nn.Sequential(*self.layers)
        self.layers.apply(tools.weight_init)
        self._out_layer = nn.Sequential(nn.Linear(attention_dim, attention_dim, bias=True),
                                        nn.GELU(),
                                        nn.Linear(attention_dim, num_action + shape[0], bias=True))
        self._out_layer.apply(tools.weight_init)

    def __call__(self, stoch, deter, targets_array, dtype=None):
        original = deter.shape
        stoch = stoch.reshape(-1, stoch.shape[-1])
        deter = deter.reshape(-1, deter.shape[-1])
        targets_array = targets_array.reshape(-1)
        token1 = self.stoch_layer(stoch).unsqueeze(-2)
        token2 = self.deter_layer(deter).unsqueeze(-2)
        feature = torch.cat([token1, token2], dim=-2)
        # b h 1 d
        (_, out) = self.layers((feature, self.embedding(targets_array).unsqueeze(-2)))
        if len(original) == 2:
            out = out.reshape(original[0], -1)
        else:
            out = out.reshape(original[0], original[1], -1)
        x = self._out_layer(out)
        values = x[..., 0:self.shape[0]]
        actions = x[..., self.shape[0]:]
        return values, actions
