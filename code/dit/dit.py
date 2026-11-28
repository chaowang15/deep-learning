"""DiT (Peebles & Xie, 2023) written from scratch, following the official facebookresearch/DiT code.

Shapes: z_t [B, C, H, W] -> tokens [B, N, D] with N = (H/p)(W/p) -> output [B, C_out, H, W].
cond_mode selects the block design from the paper's ablation: "adaln_zero" (default), "adaln", "in_context", "cross_attn".
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    """adaLN: x * (1 + scale) + shift.  x: [B, N, D], shift/scale: [B, D] (same for every token)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------- embeddings
class TimestepEmbedder(nn.Module):
    """t [B] (float or int) -> sinusoidal features [B, 256] -> MLP -> [B, D]."""
    def __init__(self, D, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, D), nn.SiLU(), nn.Linear(D, D))

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)  # [half]
        args = t.float()[:, None] * freqs[None]                                                 # [B, half]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)                            # [B, dim]

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


class LabelEmbedder(nn.Module):
    """y [B] -> [B, D]. One extra row (id = num_classes) is the null label used for CFG."""
    def __init__(self, num_classes, D, dropout_prob=0.1):
        super().__init__()
        self.table = nn.Embedding(num_classes + 1, D)
        self.num_classes, self.dropout_prob = num_classes, dropout_prob

    def forward(self, y, train):
        if train and self.dropout_prob > 0:                       # drop labels -> null id, for CFG
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        return self.table(y)


def get_2d_sincos_pos_embed(D, grid_size):
    """Fixed 2-D sin-cos position embedding: [grid*grid, D]. Half of D encodes the row, half the column."""
    def emb_1d(d, pos):                                            # pos [M] -> [M, d]
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    g0, g1 = np.meshgrid(np.arange(grid_size, dtype=np.float64), np.arange(grid_size, dtype=np.float64), indexing="xy")
    # official code: grid[0] (column index w) goes to the first half, grid[1] (row index h) to the second
    return np.concatenate([emb_1d(D // 2, g0), emb_1d(D // 2, g1)], axis=1)


# ---------------------------------------------------------------- blocks
class Attention(nn.Module):
    """Multi-head self-attention (or cross-attention when ctx is given). x: [B, N, D]."""
    def __init__(self, D, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(D, D)
        self.kv = nn.Linear(D, 2 * D)
        self.proj = nn.Linear(D, D)

    def forward(self, x, ctx=None):
        B, N, D = x.shape
        ctx = x if ctx is None else ctx
        q = self.q(x).view(B, N, self.h, D // self.h).transpose(1, 2)               # [B, h, N, d]
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)                               # softmax(qk^T/sqrt d) v
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class Mlp(nn.Module):
    def __init__(self, D, ratio=4.0):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(D, int(D * ratio)), nn.Linear(int(D * ratio), D)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class DiTBlock(nn.Module):
    """One DiT block. c: [B, D] = t_emb + y_emb (adaLN variants) or the cond tokens (cross_attn)."""
    def __init__(self, D, heads, cond_mode="adaln_zero"):
        super().__init__()
        self.cond_mode = cond_mode
        ada = cond_mode in ("adaln", "adaln_zero")
        self.norm1 = nn.LayerNorm(D, elementwise_affine=not ada, eps=1e-6)  # adaLN supplies γ, β itself
        self.attn = Attention(D, heads)
        self.norm2 = nn.LayerNorm(D, elementwise_affine=not ada, eps=1e-6)
        self.mlp = Mlp(D)
        if ada:
            n = 6 if cond_mode == "adaln_zero" else 4                        # (β1, γ1, α1, β2, γ2, α2) or no α
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, n * D))
        if cond_mode == "cross_attn":
            self.norm_x = nn.LayerNorm(D, eps=1e-6)
            self.cross = Attention(D, heads)

    def forward(self, x, c):
        if self.cond_mode == "adaln_zero":
            shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
            x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))
        elif self.cond_mode == "adaln":
            shift1, scale1, shift2, scale2 = self.adaLN_modulation(c).chunk(4, dim=1)
            x = x + self.attn(modulate(self.norm1(x), shift1, scale1))
            x = x + self.mlp(modulate(self.norm2(x), shift2, scale2))
        elif self.cond_mode == "cross_attn":                                # c: [B, 2, D] tokens (t, y)
            x = x + self.attn(self.norm1(x))
            x = x + self.cross(self.norm_x(x), c)
            x = x + self.mlp(self.norm2(x))
        else:                                                               # in_context: cond already in x
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """adaLN (shift, scale) + Linear: [B, N, D] -> [B, N, p*p*C_out]."""
    def __init__(self, D, p, out_ch, ada=True):
        super().__init__()
        self.ada = ada
        self.norm = nn.LayerNorm(D, elementwise_affine=not ada, eps=1e-6)
        self.linear = nn.Linear(D, p * p * out_ch)
        if ada:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))

    def forward(self, x, c):
        if self.ada:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            return self.linear(modulate(self.norm(x), shift, scale))
        return self.linear(self.norm(x))


# ---------------------------------------------------------------- model
class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16,
                 num_classes=1000, class_dropout_prob=0.1, learn_sigma=True, cond_mode="adaln_zero"):
        super().__init__()
        self.p, self.C = patch_size, in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels       # ε (and Σ)
        self.cond_mode = cond_mode
        D = hidden_size
        self.x_embedder = nn.Conv2d(in_channels, D, kernel_size=patch_size, stride=patch_size)  # patchify
        self.t_embedder = TimestepEmbedder(D)
        self.y_embedder = LabelEmbedder(num_classes, D, class_dropout_prob)
        self.grid = input_size // patch_size
        self.num_patches = self.grid ** 2                                           # N = (H/p)(W/p)
        self.register_buffer("pos_embed", torch.from_numpy(
            get_2d_sincos_pos_embed(D, self.grid)).float().unsqueeze(0))          # [1, N, D], fixed
        self.blocks = nn.ModuleList([DiTBlock(D, num_heads, cond_mode) for _ in range(depth)])
        self.final_layer = FinalLayer(D, patch_size, self.out_channels, ada=cond_mode in ("adaln", "adaln_zero"))
        self.initialize_weights()

    def initialize_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        w = self.x_embedder.weight.data                                             # patchify = a Linear on p*p*C
        nn.init.xavier_uniform_(w.view(w.shape[0], -1)); nn.init.zeros_(self.x_embedder.bias)
        nn.init.normal_(self.y_embedder.table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.cond_mode == "adaln_zero":                                          # the "Zero" in adaLN-Zero
            for blk in self.blocks:
                nn.init.zeros_(blk.adaLN_modulation[-1].weight)
                nn.init.zeros_(blk.adaLN_modulation[-1].bias)
        if self.final_layer.ada:
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)                              # output starts at 0
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x):
        """[B, N, p*p*C_out] -> [B, C_out, H, W]."""
        B, p, c, h = x.shape[0], self.p, self.out_channels, self.grid
        x = x.reshape(B, h, h, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)                                       # [B, C, h, p, w, p]
        return x.reshape(B, c, h * p, h * p)

    def forward(self, z_t, t, y):
        x = self.x_embedder(z_t).flatten(2).transpose(1, 2) + self.pos_embed        # [B, N, D]
        t_emb = self.t_embedder(t)                                                   # [B, D]
        y_emb = self.y_embedder(y, self.training)                                    # [B, D]
        if self.cond_mode in ("adaln", "adaln_zero"):
            c = t_emb + y_emb                                                        # [B, D]
        elif self.cond_mode == "cross_attn":
            c = torch.stack([t_emb, y_emb], dim=1)                                   # [B, 2, D]
        else:                                                                        # in_context
            x = torch.cat([torch.stack([t_emb, y_emb], dim=1), x], dim=1)            # [B, 2+N, D]
            c = None
        for blk in self.blocks:
            x = blk(x, c)
        if self.cond_mode == "in_context":
            x = x[:, 2:]                                                             # drop the 2 cond tokens
        x = self.final_layer(x, t_emb + y_emb)                                       # [B, N, p*p*C_out]
        return self.unpatchify(x)                                                    # [B, C_out, H, W]


CONFIGS = {  # name: (depth, hidden, heads), from the paper's Table 1
    "S": (12, 384, 6), "B": (12, 768, 12), "L": (24, 1024, 16), "XL": (28, 1152, 16)}


def DiT_model(name, patch_size, **kw):
    d, D, h = CONFIGS[name]
    return DiT(patch_size=patch_size, depth=d, hidden_size=D, num_heads=h, **kw)
