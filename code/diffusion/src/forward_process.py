"""Stage 01 (notebooks/01_forward_process.ipynb): q_sample, the forward noising formula."""

import torch


def q_sample(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, noise=None):
    #
    # x0: [B, C, H, W]
    # t: [B], 0-based code index (paper timestep = t + 1)
    #
    # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

    if noise is None:
        noise = torch.randn_like(x0)  # same shape/device/dtype as x0

    t = t.to(device=x0.device, dtype=torch.long)
    sqrt_alpha_bars = sqrt_alpha_bars.to(device=x0.device, dtype=x0.dtype)
    sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.to(
        device=x0.device, dtype=x0.dtype
    )

    # [B] => [B, 1, 1, 1]
    sqrt_alpha_bar_t = sqrt_alpha_bars[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

    xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    return xt, noise
