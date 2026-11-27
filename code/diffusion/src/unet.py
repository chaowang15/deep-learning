"""Stage 02 (notebooks/02_noise_prediction.ipynb): small time-conditioned U-Net that predicts noise."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.time_embedding import sinusoidal_timestep_embedding


class TimeResBlock(nn.Module):
    """Residual block conditioned on a timestep embedding.

    x:        [B, in_ch, H, W]
    time_emb: [B, time_dim]
    returns:  [B, out_ch, H, W]

    h = conv1(x) + W_t * emb      (time bias broadcast over H, W)
    out = act(conv2(act(h)) + skip(x))
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        # Residual add needs equal channels: 1x1 conv when in_ch != out_ch.
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, time_emb):
        h = self.conv1(x)                                   # [B, out, H, W]
        time_bias = self.time_proj(time_emb)[:, :, None, None]  # [B, out] -> [B, out, 1, 1]
        h = h + time_bias                                   # broadcast over H, W
        h = F.silu(h)
        h = self.conv2(h)                                   # [B, out, H, W]
        # Note: many diffusion ResBlocks return h + skip(x) without the final SiLU.
        return F.silu(h + self.skip(x))


class SpatialSelfAttention(nn.Module):
    """Self-attention over the H*W spatial positions of a feature map.

    x: [B, C, H, W] -> tokens [B, H*W, C] -> attention -> [B, C, H, W]
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V, with Q = K = V = norm(tokens).
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)               # [B, C, H*W] -> [B, H*W, C]
        q = self.norm(tokens)
        attended, _ = self.attention(q, q, q, need_weights=False)
        tokens = tokens + attended                          # residual in token space
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class TinyUNet(nn.Module):
    """One-down / one-up U-Net that predicts epsilon_theta(x_t, t).

    x_t: [B, C, H, W] (H, W even);  t: [B] long  ->  pred_noise: [B, C, H, W]
    """

    def __init__(self, in_channels=1, base_channels=32, time_dim=128, use_attention=False):
        super().__init__()
        self.time_dim = time_dim
        c = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.encoder = TimeResBlock(in_channels, c, time_dim)
        self.down = nn.Conv2d(c, 2 * c, kernel_size=4, stride=2, padding=1)        # H -> H/2
        self.bottleneck = TimeResBlock(2 * c, 2 * c, time_dim)
        self.attention = SpatialSelfAttention(2 * c) if use_attention else nn.Identity()
        self.up = nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1)  # H/2 -> H
        self.decoder = TimeResBlock(2 * c, c, time_dim)     # input is concat [up, skip]
        self.output = nn.Conv2d(c, in_channels, kernel_size=3, padding=1)

    def forward(self, x_t, t):
        time_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))  # [B, time_dim]
        skip = self.encoder(x_t, time_emb)                  # [B, c, H, W]
        h = self.down(skip)                                 # [B, 2c, H/2, W/2]
        h = self.bottleneck(h, time_emb)                    # [B, 2c, H/2, W/2]
        h = self.attention(h)                               # [B, 2c, H/2, W/2]
        h = self.up(h)                                      # [B, c, H, W]
        h = torch.cat([h, skip], dim=1)                     # [B, 2c, H, W]
        h = self.decoder(h, time_emb)                       # [B, c, H, W]
        return self.output(h)                               # [B, C, H, W]


class ClassConditionedTinyUNet(TinyUNet):
    """TinyUNet + class-label conditioning, with a learned "null" label for CFG.

    labels: [B] long in [0, num_classes); id num_classes = null (unconditional).
    condition_emb = time_emb + class_emb, injected wherever time_emb was.
    """

    def __init__(self, num_classes, in_channels=1, base_channels=32, time_dim=128, use_attention=False):
        super().__init__(in_channels, base_channels, time_dim, use_attention)
        self.num_classes = num_classes
        self.null_class_id = num_classes
        self.class_embedding = nn.Embedding(num_classes + 1, time_dim)

    def forward(self, x_t, t, labels=None):
        if labels is None:
            labels = torch.full((x_t.shape[0],), self.null_class_id, device=x_t.device, dtype=torch.long)
        time_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))
        cond = time_emb + self.class_embedding(labels)      # [B, time_dim]
        skip = self.encoder(x_t, cond)
        h = self.attention(self.bottleneck(self.down(skip), cond))
        h = torch.cat([self.up(h), skip], dim=1)
        return self.output(self.decoder(h, cond))
