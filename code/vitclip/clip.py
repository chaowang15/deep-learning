"""Minimal CLIP (Radford et al. 2021): an image tower (ViT) and a text tower
(causal Transformer), each projected to a shared embedding space, trained with
a symmetric contrastive (InfoNCE) loss. Layout follows openai/CLIP model.py,
so the parameter counts of the released models can be reproduced
(see check_clip.py).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import Block


class QuickGELU(nn.Module):          # x * sigmoid(1.702 x), used by the OpenAI checkpoints
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Tower(nn.Module):
    def __init__(self, dim, depth, heads, act=QuickGELU):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim, heads, 4.0, act, eps=1e-5) for _ in range(depth))

    def forward(self, x, mask=None):
        for b in self.blocks:
            x = b(x, mask)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, img=224, patch=32, in_ch=3, dim=768, depth=12, heads=12, embed=512):
        super().__init__()
        n = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, stride=patch, bias=False)
        self.cls = nn.Parameter(torch.randn(dim) * dim ** -0.5)
        self.pos = nn.Parameter(torch.randn(n + 1, dim) * dim ** -0.5)
        self.ln_pre = nn.LayerNorm(dim)
        self.tower = Tower(dim, depth, heads)
        self.ln_post = nn.LayerNorm(dim)
        self.proj = nn.Parameter(torch.randn(dim, embed) * dim ** -0.5)

    def forward(self, x):                                       # (B, C, H, W) -> (B, E)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)      # (B, N, D)
        x = torch.cat([self.cls.expand(len(x), 1, -1), x], 1) + self.pos
        x = self.tower(self.ln_pre(x))
        return self.ln_post(x[:, 0]) @ self.proj                # [CLS] -> shared space


class TextEncoder(nn.Module):
    def __init__(self, vocab=49408, ctx=77, dim=512, depth=12, heads=8, embed=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(ctx, dim) * 0.01)
        self.tower = Tower(dim, depth, heads)
        self.ln_final = nn.LayerNorm(dim)
        self.proj = nn.Parameter(torch.randn(dim, embed) * dim ** -0.5)
        # causal mask: token i attends only to tokens <= i (GPT-style)
        self.register_buffer("mask", torch.ones(ctx, ctx, dtype=torch.bool).tril(), persistent=False)

    def forward(self, ids):                                     # (B, L) int -> (B, E)
        L = ids.shape[1]
        x = self.tok(ids) + self.pos[:L]
        x = self.ln_final(self.tower(x, self.mask[:L, :L]))
        # feature at the [EOT] token; in the BPE vocab EOT has the largest id
        eot = ids.argmax(-1)
        return x[torch.arange(len(x)), eot] @ self.proj


class CLIP(nn.Module):
    def __init__(self, image: ImageEncoder, text: TextEncoder, init_temp=0.07):
        super().__init__()
        self.image, self.text = image, text
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temp)))   # learned 1/tau

    def forward(self, imgs, ids):
        u = F.normalize(self.image(imgs), dim=-1)               # (B, E), unit length
        v = F.normalize(self.text(ids), dim=-1)
        s = self.logit_scale.exp().clamp(max=100)               # paper clips 1/tau at 100
        return s * u @ v.t()                                    # (B, B): logits[i, j] = s * cos(u_i, v_j)


def clip_loss(logits):
    """Symmetric InfoNCE: row i should pick text i, column j should pick image j."""
    y = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y)) / 2


# OpenAI release configs: image (img, patch, width, layers, heads), text (width, layers, heads), embed
CONFIGS = {
    "ViT-B/32": ((224, 32, 768, 12, 12), (512, 12, 8), 512),
    "ViT-B/16": ((224, 16, 768, 12, 12), (512, 12, 8), 512),
    "ViT-L/14": ((224, 14, 1024, 24, 16), (768, 12, 12), 768),
    "ViT-L/14@336": ((336, 14, 1024, 24, 16), (768, 12, 12), 768),
}


def build(name):
    (img, p, w, L, h), (tw, tL, th), e = CONFIGS[name]
    return CLIP(ImageEncoder(img, p, 3, w, L, h, e), TextEncoder(49408, 77, tw, tL, th, e))
