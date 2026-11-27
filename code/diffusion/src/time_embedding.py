"""Stage 02 (notebooks/02_noise_prediction.ipynb): sinusoidal timestep embedding."""

import math

import torch


def sinusoidal_timestep_embedding(t, time_dim):
    """Encode integer timesteps as sin/cos features.

    t:        [B] long, 0-based timestep index (same index used by q_sample).
    time_dim: even int >= 4, embedding length (not the number of steps T).
    returns:  [B, time_dim]

    D = time_dim / 2
    omega_i = exp(-log(10000) * i / (D - 1)),  i = 0..D-1
    emb(t) = [sin(t*omega_0..D-1), cos(t*omega_0..D-1)]
    """
    if time_dim < 4 or time_dim % 2 != 0:
        raise ValueError("time_dim must be an even integer >= 4")

    half_dim = time_dim // 2                                        # D
    # omega_i, [D]; highest frequency 1 (i=0), lowest 1/10000 (i=D-1)
    omegas = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=t.device) / (half_dim - 1)
    )
    # angles[b, i] = t[b] * omega_i:  [B, 1] * [1, D] -> [B, D]
    angles = t.float().unsqueeze(1) * omegas.unsqueeze(0)
    # [B, D] ++ [B, D] -> [B, 2D] = [B, time_dim]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
