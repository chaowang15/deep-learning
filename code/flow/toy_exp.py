"""2-D flow matching / rectified flow experiment for the flow-matching post (CPU, ~2 min).

Convention (same as the DDPM / DDIM posts and SD3): t=0 is data x0, t=1 is noise eps,
x_t = (1 - t) x0 + t eps, target velocity u = dx_t/dt = eps - x0, sampling integrates t: 1 -> 0.
"""
import json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

torch.manual_seed(0)
OUT = Path(__file__).resolve().parents[2] / "assets" / "flow-matching"
OUT.mkdir(parents=True, exist_ok=True)


def sample_data(n):
    """8 Gaussians on a ring of radius 3, std 0.15  -> [n, 2]."""
    k = torch.randint(0, 8, (n,))
    ang = k.float() * (2 * math.pi / 8)
    centers = 3.0 * torch.stack([ang.cos(), ang.sin()], 1)
    return centers + 0.15 * torch.randn(n, 2)


class VelocityMLP(nn.Module):
    """v_theta(x_t, t): [B, 2], [B] -> [B, 2]."""
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 16, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.register_buffer("freq", torch.arange(1, 9).float() * math.pi)

    def forward(self, x, t):
        tf = t[:, None] * self.freq                                  # [B, 8]
        return self.net(torch.cat([x, tf.sin(), tf.cos()], 1))


def fm_loss(model, x0, eps):
    """Conditional flow matching loss with the linear (rectified flow) path."""
    t = torch.rand(x0.shape[0])                                       # t ~ U(0, 1)
    tt = t[:, None]
    x_t = (1 - tt) * x0 + tt * eps                                    # x_t = (1-t) x0 + t eps
    u = eps - x0                                                      # u = eps - x0
    return ((model(x_t, t) - u) ** 2).mean()


@torch.no_grad()
def sample_euler(model, eps, num_steps, return_path=False):
    """Euler from t=1 (noise) to t=0 (data): x_{t-dt} = x_t - dt * v_theta(x_t, t)."""
    x = eps.clone()
    ts = torch.linspace(1.0, 0.0, num_steps + 1)
    path = [x.clone()]
    for i in range(num_steps):
        t, t_next = ts[i], ts[i + 1]
        v = model(x, t.expand(x.shape[0]))
        x = x + (t_next - t) * v                                      # t_next - t < 0
        path.append(x.clone())
    return (x, torch.stack(path)) if return_path else x


def train(model, pair_fn, steps=6000, bs=512, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x0, eps = pair_fn(bs)
        loss = fm_loss(model, x0, eps)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def mode_stats(x):
    """Fraction of samples within 0.5 of a mode, and number of modes covered (>=2% of samples)."""
    ang = torch.arange(8).float() * (2 * math.pi / 8)
    centers = 3.0 * torch.stack([ang.cos(), ang.sin()], 1)
    d = torch.cdist(x, centers)
    near = d.min(1).values < 0.5
    idx = d.argmin(1)[near]
    counts = torch.bincount(idx, minlength=8).float() / len(x)
    return near.float().mean().item(), int((counts > 0.02).sum())


@torch.no_grad()
def straightness(model, eps, n=100):
    """S = E_t || v(x_t, t) - (x_1 - x_0) ||^2 along the model's own ODE path (0 = perfectly straight)."""
    x0, path = sample_euler(model, eps, n, return_path=True)
    disp = eps - x0                                                   # x_1 - x_0 for each trajectory
    ts = torch.linspace(1.0, 0.0, n + 1)
    s = 0.0
    for i in range(n):
        v = model(path[i], ts[i].expand(len(eps)))
        s += ((v - disp) ** 2).sum(1).mean().item() / n
    return s


if __name__ == "__main__":
    t0 = time.time()
    rf1 = train(VelocityMLP(), lambda b: (sample_data(b), torch.randn(b, 2)))
    print("1-RF trained", round(time.time() - t0, 1), "s")

    # Reflow: build pairs (x0_hat, eps) with the 1-RF ODE, then train again on those fixed pairs.
    eps_pool = torch.randn(50000, 2)
    x0_pool = sample_euler(rf1, eps_pool, 100)
    def reflow_pairs(b):
        i = torch.randint(0, len(eps_pool), (b,))
        return x0_pool[i], eps_pool[i]
    rf2 = train(VelocityMLP(), reflow_pairs)
    print("2-RF trained", round(time.time() - t0, 1), "s")

    torch.manual_seed(1)
    eps_test = torch.randn(4000, 2)
    res = {}
    for name, m in [("1-RF", rf1), ("2-RF", rf2)]:
        ref = sample_euler(m, eps_test, 100)
        row = {"straightness": straightness(m, eps_test[:1000])}
        for n in [1, 2, 4, 8, 100]:
            x = sample_euler(m, eps_test, n)
            near, modes = mode_stats(x)
            row[f"N{n}"] = {"near": near, "modes": modes, "mse_to_100": ((x - ref) ** 2).sum(1).mean().item()}
        res[name] = row
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / "toy_results.json", "w"), indent=1)
    torch.save({"rf1": rf1.state_dict(), "rf2": rf2.state_dict()}, "/tmp/flow_toy.pt")
    print("total", round(time.time() - t0, 1), "s")
