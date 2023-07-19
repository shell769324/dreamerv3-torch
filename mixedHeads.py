import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
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
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) + x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.LogSoftmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        if type(x) is tuple:
            q, k, v = x
        else:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if type(x) is not tuple:
            return self.to_out(out) + x
        return self.to_out(out)


class MixedHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        embed_dim,
        shape,
        layers,
        dist="normal",
        std=1.0,
        outscale=1.0,
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
        self.feature_layer = nn.Linear(inp_dim, embed_dim * 2 * len(targets), bias=True)
        self.feature_layer.apply(tools.weight_init)
        self.embed_dim = embed_dim

        for index in range(layers):
            if index == 0:
                self.layers.append(Attention(embed_dim, heads=heads))
            else:
                self.layers.append(PreNorm(embed_dim, Attention(embed_dim, heads=heads)))
            self.layers.append(PreNorm(embed_dim, FeedForward(embed_dim, embed_dim * 2)))
        self.layers = nn.Sequential(*self.layers)
        self.layers.apply(tools.weight_init)

        self.mean_layer = nn.Linear(embed_dim, 1)
        self.mean_layer.apply(tools.uniform_weight_init(outscale))

        if self._std == "learned":
            self.std_layer = nn.Linear(self._units, np.prod(self._shape))
            self.std_layer.apply(tools.uniform_weight_init(outscale))

    def __call__(self, features, targets_array, dtype=None):
        print("Embedding cosine-similarity")
        for i in range(len(targets)):
            if i == 0:
                print(targets, end=" ")
                print("")
            for j in range(len(targets)):
                print(nn.functional.cosine_similarity(self.embedding(torch.IntTensor([i]).to(self._device)), self.embedding(torch.IntTensor([j]).to(self._device))).item(), end=" ")
            print("")

        original = features.shape
        features = features.reshape(-1, features.shape[-1])
        targets_array = targets_array.reshape(-1)
        kv = self.feature_layer(features).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b (n h d) -> b h n d', n=len(targets), h=self.heads), kv)
        q = self.embedding(targets_array).reshape(-1, self.heads, self.embed_dim // self.heads)
        q = repeat(q, 'b h d -> b h n d', n=len(targets))
        out = self.layers((q, k, v))
        out = out.mean(dim=1)
        out = out.reshape(original[0], original[1], -1)
        print("out mean", torch.mean(out, dim=(1, 2)))
        print("out var",  torch.var(out, dim=(1, 2)))

        mean = self.mean_layer(out)
        print("mean comp", mean[:, 0], targets_array.reshape(original[0], original[1])[:, 0])
        print("mean var comp", torch.var(mean))
        logits = 1/(torch.pow(torch.e, -mean) + 1)
        print("logits", logits[:, 0])
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
                    torchd.bernoulli.Bernoulli(logits=1/(torch.pow(torch.e, -mean) + 1)), len(self._shape)
                )
            )
        if self._dist == "twohot_symlog":
            return tools.TwoHotDistSymlog(logits=mean, device=self._device)
        raise NotImplementedError(self._dist)
