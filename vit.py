import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


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
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        print("qkv", qkv[0].shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        print("q k v", q.shape, k.shape, v.shape)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        print("dots", dots.shape)

        attn = self.attend(dots)
        print("attn softmax", attn.shape)

        out = torch.matmul(attn, v)
        print("value weighted", out.shape)
        out = rearrange(out, 'b h n d -> b n (h d)')
        print("post value weighted rearrange", out.shape)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim) if i < depth - 1 else nn.Sequential(nn.Linear(dim, output_dim)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            print("post attn", x.shape)
            x = ff(x) + x
            print("post feed", x.shape)
        return x

    def __call__(self, x):
        return self.forward(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, emb_dropout = 0., output_dim = 512):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Segmented: 1 16 16*16*3
        # 16*16*3 dim matrix mult
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Concat with cls and then mult
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, output_dim)

        self.pool = pool

    def forward(self, obs):
        print("initial", obs["image"].shape)
        x = obs["image"].reshape((-1,) + tuple(obs["image"].shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        print("post permute", x.shape)
        x = self.to_patch_embedding(x)
        print("post patch embedding", x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        print("post cls cat", x.shape)
        x += self.pos_embedding[:, :(n + 1)]
        print("post cls cat embedding", x.shape)
        x = self.dropout(x)

        x = self.transformer(x)

        print("post transformer", x.shape)
        shape = list(obs["image"].shape[:-3]) + [x.shape[-1]]
        print("final shape", shape)
        return x.reshape(shape)

    def __call__(self, obs):
        return self.forward(obs)