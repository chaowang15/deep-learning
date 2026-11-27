"""Sanity checks for src/schedule.py (Stage 01). Run: python -m pytest -q"""

import torch

from src.schedule import make_beta_schedule


def test_shapes_and_ranges():
    T = 300
    s = make_beta_schedule(T)
    for name, values in s.items():
        assert values.shape == (T,), name
    # alpha_bar is a cumulative product of numbers in (0, 1): it must keep shrinking.
    assert torch.all(s["alpha_bars"][1:] < s["alpha_bars"][:-1])
    assert 0.0 < s["alpha_bars"][-1] < s["alpha_bars"][0] < 1.0


def test_signal_plus_noise_variance_is_one():
    # sqrt(a)^2 + sqrt(1-a)^2 == 1, which is why x_t keeps unit variance.
    s = make_beta_schedule(300)
    total = s["sqrt_alpha_bars"] ** 2 + s["sqrt_one_minus_alpha_bars"] ** 2
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_posterior_variance_starts_at_zero():
    # alpha_bar_{-1} = 1, so beta_tilde_0 = 0: the last reverse step adds no noise.
    s = make_beta_schedule(300)
    assert s["posterior_variance"][0].item() == 0.0
    assert torch.all(s["posterior_variance"][1:] <= s["betas"][1:])
