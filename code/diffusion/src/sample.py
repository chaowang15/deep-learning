"""Stage 04-05 (notebooks/04_sampling.ipynb, 05_ddim_and_cfg.ipynb): DDPM / DDIM sampling and CFG."""

import torch


def predict_noise(model, x_t, t, labels=None, guidance_scale=None):
    """eps_theta(x_t, t[, y]).  With guidance_scale w (CFG):
    eps = eps_uncond + w * (eps_cond - eps_uncond)
    """
    if labels is None:
        return model(x_t, t)
    if guidance_scale is None:
        return model(x_t, t, labels)
    eps_uncond = model(x_t, t, torch.full_like(labels, model.null_class_id))
    eps_cond = model(x_t, t, labels)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


@torch.no_grad()
def p_sample(model, x_t, t_index, sched, labels=None, guidance_scale=None):
    """One DDPM reverse step x_t -> x_{t-1}. t_index is a Python int (0-based).

    mu = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta)
    x_{t-1} = mu + sqrt(beta_tilde_t) * z   (no z at the last step)
    """
    t = torch.full((x_t.shape[0],), t_index, device=x_t.device, dtype=torch.long)  # [B]
    pred_noise = predict_noise(model, x_t, t, labels, guidance_scale)

    # Same t for the whole batch -> scalar coefficients broadcast directly.
    beta_t = sched["betas"][t_index]
    sqrt_recip_alpha_t = sched["sqrt_recip_alphas"][t_index]
    sqrt_one_minus_alpha_bar_t = sched["sqrt_one_minus_alpha_bars"][t_index]

    mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise)
    if t_index == 0:
        return mean
    z = torch.randn_like(x_t)
    return mean + torch.sqrt(sched["posterior_variance"][t_index]) * z


@torch.no_grad()
def sample_ddpm(model, shape, sched, device, labels=None, guidance_scale=None):
    """x_T ~ N(0, I), then T reverse steps: x_T -> ... -> x_0."""
    model.eval()
    sched = {k: v.to(device) for k, v in sched.items()}
    x_t = torch.randn(shape, device=device)
    for t_index in reversed(range(sched["betas"].shape[0])):
        x_t = p_sample(model, x_t, t_index, sched, labels, guidance_scale)
    return x_t


@torch.no_grad()
def ddim_step(model, x_t, t_index, prev_index, sched, eta=0.0, labels=None, guidance_scale=None):
    """DDIM jump from timestep t to an earlier selected timestep s (prev_index; -1 = clean).

    x0_hat = (x_t - sqrt(1 - abar_t) eps) / sqrt(abar_t)
    sigma^2 = eta^2 * (1 - abar_s)/(1 - abar_t) * (1 - abar_t/abar_s)
    x_s = sqrt(abar_s) x0_hat + sqrt(1 - abar_s - sigma^2) eps + sigma z
    """
    t = torch.full((x_t.shape[0],), t_index, device=x_t.device, dtype=torch.long)
    pred_noise = predict_noise(model, x_t, t, labels, guidance_scale)
    abar_t = sched["alpha_bars"][t_index]
    abar_s = sched["alpha_bars"][prev_index] if prev_index >= 0 else abar_t.new_tensor(1.0)

    x0_hat = (x_t - torch.sqrt(1 - abar_t) * pred_noise) / torch.sqrt(abar_t)
    sigma2 = eta**2 * (1 - abar_s) / (1 - abar_t) * (1 - abar_t / abar_s)
    dir_coef = torch.sqrt(torch.clamp(1 - abar_s - sigma2, min=0.0))
    return torch.sqrt(abar_s) * x0_hat + dir_coef * pred_noise + torch.sqrt(sigma2) * torch.randn_like(x_t)


@torch.no_grad()
def sample_ddim(model, shape, sched, device, num_steps=50, eta=0.0, labels=None, guidance_scale=None):
    """Visit only num_steps timesteps, e.g. 299, 293, ..., 0, then jump to clean."""
    model.eval()
    sched = {k: v.to(device) for k, v in sched.items()}
    T = sched["betas"].shape[0]
    steps = torch.linspace(T - 1, 0, num_steps).round().long().tolist()
    x_t = torch.randn(shape, device=device)
    for i, t_index in enumerate(steps):
        prev_index = steps[i + 1] if i + 1 < len(steps) else -1
        x_t = ddim_step(model, x_t, t_index, prev_index, sched, eta, labels, guidance_scale)
    return x_t
