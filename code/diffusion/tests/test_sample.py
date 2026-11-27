import torch
import pytest

from src.schedule import make_beta_schedule
from src.sample import ddim_timesteps, ddim_step, sample_ddim, ddim_invert

T = 300


class OracleNoise(torch.nn.Module):
    """Knows the clean image x0, so it returns the exact noise eps = (x_t - sqrt(abar_t) x0) / sqrt(1 - abar_t)."""

    def __init__(self, x0, alpha_bars):
        super().__init__()
        self.x0, self.alpha_bars = x0, alpha_bars

    def forward(self, x_t, t):
        abar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return (x_t - torch.sqrt(abar) * self.x0) / torch.sqrt(1 - abar)


def test_timesteps_start_end_unique():
    for n in [1, 2, 10, 50, 300]:
        steps = ddim_timesteps(T, n)
        assert len(steps) == n and len(set(steps)) == n
        assert steps[0] == T - 1 and steps[-1] == (0 if n > 1 else T - 1)
    with pytest.raises(ValueError):
        ddim_timesteps(T, T + 1)


def test_eta1_single_step_matches_ddpm_variance():
    sched = make_beta_schedule(T)
    ab = sched["alpha_bars"]
    for t in [1, 50, 299]:
        s = t - 1
        sigma2 = (1 - ab[s]) / (1 - ab[t]) * (1 - ab[t] / ab[s])
        assert torch.allclose(sigma2, sched["posterior_variance"][t], rtol=1e-4)


@pytest.mark.parametrize("num_steps", [1, 5, 50])
def test_perfect_noise_prediction_recovers_x0_in_any_number_of_steps(num_steps):
    sched = make_beta_schedule(T)
    x0 = torch.rand(2, 1, 8, 8) * 2 - 1
    model = OracleNoise(x0, sched["alpha_bars"])
    out = sample_ddim(model, x0.shape, sched, "cpu", num_steps=num_steps)
    assert torch.allclose(out, x0, atol=1e-4)


def test_eta0_is_deterministic_and_inversion_round_trips():
    sched = make_beta_schedule(T)
    x0 = torch.rand(2, 1, 8, 8) * 2 - 1
    model = OracleNoise(x0, sched["alpha_bars"])
    x_T = ddim_invert(model, x0, sched, num_steps=20)
    a = sample_ddim(model, x0.shape, sched, "cpu", num_steps=20, x_T=x_T)
    b = sample_ddim(model, x0.shape, sched, "cpu", num_steps=20, x_T=x_T)
    assert torch.equal(a, b)
    assert torch.allclose(a, x0, atol=1e-4)


def test_clip_x0_keeps_output_in_range_at_last_step():
    sched = make_beta_schedule(T)
    x_t = torch.randn(2, 1, 8, 8) * 5
    model = lambda x, t: torch.zeros_like(x)  # eps = 0 -> x0_hat = x_t / sqrt(abar) is large
    out = ddim_step(model, x_t, 0, -1, sched, clip_x0=True)
    assert out.abs().max() <= 1 + 1e-6
