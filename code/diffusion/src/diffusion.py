"""Stage 03 (notebooks/03_ddpm_training.ipynb): DDPM training objective, plus epsilon / v / x0 targets."""

import torch
import torch.nn.functional as F

from src.forward_process import q_sample


def drop_labels_for_cfg(labels, null_class_id, drop_prob=0.1):
    """Replace each label by the null id with probability drop_prob (CFG training)."""
    drop = torch.rand(labels.shape, device=labels.device) < drop_prob
    return torch.where(drop, torch.full_like(labels, null_class_id), labels)


def ddpm_training_step(model, optimizer, x0, sched, labels=None, cfg_drop_prob=0.1):
    """One optimizer update on one batch.

    t ~ Uniform{0..T-1} (0-based index),  eps ~ N(0, I)
    x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps
    L = || eps - eps_theta(x_t, t) ||^2
    """
    model.train()
    T = sched["betas"].shape[0]
    t = torch.randint(0, T, (x0.shape[0],), device=x0.device)              # [B]
    xt, noise = q_sample(x0, t, sched["sqrt_alpha_bars"], sched["sqrt_one_minus_alpha_bars"])
    if labels is None:
        pred_noise = model(xt, t)
    else:
        pred_noise = model(xt, t, drop_labels_for_cfg(labels, model.null_class_id, cfg_drop_prob))
    loss = F.mse_loss(pred_noise, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach()


# ---- Prediction targets: epsilon / v / x0 -----------------------------------
# a_t = sqrt(alpha_bar_t), s_t = sqrt(1 - alpha_bar_t), x_t = a_t x0 + s_t eps
# v_t = a_t eps - s_t x0


def _coef(table, t, like):
    return table.to(like.device, like.dtype)[t].view(-1, 1, 1, 1)


def training_target(x0, noise, t, sched, prediction_type):
    a_t = _coef(sched["sqrt_alpha_bars"], t, x0)
    s_t = _coef(sched["sqrt_one_minus_alpha_bars"], t, x0)
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "x0":
        return x0
    if prediction_type == "v":
        return a_t * noise - s_t * x0
    raise ValueError(prediction_type)


def output_to_x0(x_t, out, t, sched, prediction_type):
    """Turn the network output into the current estimate x0_hat."""
    a_t = _coef(sched["sqrt_alpha_bars"], t, x_t)
    s_t = _coef(sched["sqrt_one_minus_alpha_bars"], t, x_t)
    if prediction_type == "epsilon":
        return (x_t - s_t * out) / a_t
    if prediction_type == "x0":
        return out
    if prediction_type == "v":
        return a_t * x_t - s_t * out
    raise ValueError(prediction_type)
