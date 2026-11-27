"""Sanity checks for src/forward_process.py (Stage 01)."""

import torch

from src.forward_process import q_sample
from src.schedule import make_beta_schedule


def _setup(T=300):
    s = make_beta_schedule(T)
    return s["sqrt_alpha_bars"], s["sqrt_one_minus_alpha_bars"]


def test_shape_grayscale_and_rgb():
    a, b = _setup()
    for shape in [(4, 1, 28, 28), (4, 3, 32, 32)]:
        x0 = torch.randn(shape)
        t = torch.randint(0, 300, (shape[0],))
        xt, noise = q_sample(x0, t, a, b)
        assert xt.shape == noise.shape == x0.shape


def test_matches_formula_with_fixed_noise():
    a, b = _setup()
    x0 = torch.randn(2, 1, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 299])
    xt, _ = q_sample(x0, t, a, b, noise=noise)
    for i in range(2):
        expected = a[t[i]] * x0[i] + b[t[i]] * noise[i]
        assert torch.allclose(xt[i], expected)


def test_last_step_is_almost_pure_noise():
    a, b = _setup()
    x0 = torch.ones(512, 1, 8, 8)  # a constant "image"
    t = torch.full((512,), 299)
    xt, _ = q_sample(x0, t, a, b)
    # With T=300 and linear betas, sqrt(alpha_bar_299) is about 0.2:
    # the image is mostly noise, so the mean drops and the std is close to 1.
    assert xt.mean().item() < 0.3
    assert 0.9 < xt.std().item() < 1.1
