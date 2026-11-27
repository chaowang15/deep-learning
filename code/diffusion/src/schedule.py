"""Stage 01 (notebooks/01_forward_process.ipynb): noise schedule beta / alpha / alpha_bar."""

import torch


def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    # Index convention: code index i = 0..T-1 is paper timestep t = i + 1.
    # x0 (clean image) is paper t = 0 and has no entry in these tables.
    #
    # beta = 1e-4 .. 0.02
    # alpha = 1 - beta
    # alpha_bars[i] = alphas[0] * alphas[1] * ... * alphas[i]   (cumprod includes i)
    # sqrt_alpha_bars[i] = sqrt(alpha_bars[i])
    #
    # q_sample: x_t = sqrt_alpha_bars[t] * x0 + sqrt_one_minus_alpha_bars[t] * noise

    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    # p_sample:
    # x_{t-1} = mu_theta(x_t, t) + sigma_t * noise
    # mu_theta(x_t, t) = 1 / sqrt(alpha_t) * (x_t - pred_noise * beta_t / sqrt(1 - alpha_bar_t))
    #  = sqrt_recip_alphas * (x_t - pred_noise * beta_t / sqrt_one_minus_alpha_bars)
    # sigma_t = sqrt(beta_tilde_t)
    # beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) = posterior_variance

    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # NOTE: alpha_bars_prev 表示 alpha_bar_{t-1}；当 t=0 时，前一刻的 clean
    # image 保留比例应为 1.0。torch.cat 需要 tensor，所以用 torch.ones(1)。
    # In short: prepend alpha_bar_{-1}=1.0 for the posterior variance formula.
    alpha_bars_prev = torch.cat(
        [torch.ones(1, device=device), alpha_bars[:-1]])
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    # All tensors are with size [T]
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }
