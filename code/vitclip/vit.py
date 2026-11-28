"""Minimal ViT (Dosovitskiy et al. 2021), written to match torchvision's vit_b_16
layer for layer so that weights can be copied across and outputs compared
(see check_vit.py).

Shapes: B = batch, N = number of patches = (H/p)(W/p), D = hidden width.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)          # torchvision: in_proj_weight / in_proj_bias
        self.proj = nn.Linear(dim, dim)             # torchvision: out_proj

    def forward(self, x, mask=None):                # x: (B, N, D)
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # softmax(qk^T/sqrt(d)) v
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """Pre-LN Transformer block: x + MSA(LN(x)), then x + MLP(LN(x))."""
    def __init__(self, dim, heads, mlp_ratio=4.0, act=nn.GELU, eps=1e-6):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads)
        self.ln2 = nn.LayerNorm(dim, eps=eps)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), act(), nn.Linear(hid, dim))

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))


class ViT(nn.Module):
    def __init__(self, img=224, patch=16, in_ch=3, dim=768, depth=12, heads=12,
                 mlp_ratio=4.0, n_classes=1000):
        super().__init__()
        self.patch = patch
        self.n = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, stride=patch)   # = Linear on flattened patches
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, self.n + 1, dim) * 0.02)  # learned, 1-D
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio) for _ in range(depth))
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, n_classes)

    def tokens(self, x):                              # (B, C, H, W) -> (B, 1+N, D)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)              # (B, N, D)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        return x + self.pos

    def forward(self, x, return_tokens=False):
        x = self.tokens(x)
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        return x if return_tokens else self.head(x[:, 0])              # classify from [CLS]


def interpolate_pos(pos, new_grid):
    """Resize learned position embeddings (1, 1+g*g, D) to a new grid, as done
    when fine-tuning at higher resolution (ViT paper sec 3.2)."""
    cls, grid = pos[:, :1], pos[:, 1:]
    g = int(grid.shape[1] ** 0.5)
    grid = grid.reshape(1, g, g, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    return torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)], 1)


CONFIGS = {  # name: (patch, dim, depth, heads, mlp)
    "ViT-Ti/16": (16, 192, 12, 3), "ViT-S/16": (16, 384, 12, 6),
    "ViT-B/16": (16, 768, 12, 12), "ViT-B/32": (32, 768, 12, 12),
    "ViT-L/16": (16, 1024, 24, 16), "ViT-H/14": (14, 1280, 32, 16),
}


def load_torchvision(m, tv):
    """Copy a torchvision VisionTransformer's weights into our ViT."""
    sd = tv.state_dict()
    out = {"patch_embed.weight": sd["conv_proj.weight"], "patch_embed.bias": sd["conv_proj.bias"],
           "cls": sd["class_token"], "pos": sd["encoder.pos_embedding"],
           "ln.weight": sd["encoder.ln.weight"], "ln.bias": sd["encoder.ln.bias"],
           "head.weight": sd["heads.head.weight"], "head.bias": sd["heads.head.bias"]}
    for i in range(len(m.blocks)):
        s = f"encoder.layers.encoder_layer_{i}."
        d = f"blocks.{i}."
        pairs = {"ln1": "ln_1", "ln2": "ln_2", "attn.qkv": "self_attention.in_proj",
                 "attn.proj": "self_attention.out_proj", "mlp.0": "mlp.0", "mlp.2": "mlp.3"}
        for a, b in pairs.items():
            if a == "attn.qkv":
                out[d + "attn.qkv.weight"] = sd[s + b + "_weight"]
                out[d + "attn.qkv.bias"] = sd[s + b + "_bias"]
            else:
                out[d + a + ".weight"] = sd[s + b + ".weight"]
                out[d + a + ".bias"] = sd[s + b + ".bias"]
    m.load_state_dict(out)
    return m
