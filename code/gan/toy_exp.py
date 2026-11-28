"""2-D ring of 8 Gaussians: saturating GAN vs non-saturating GAN vs WGAN-GP."""
import json, math, time, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parents[2] / "assets" / "generative-models-2-gan"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False

centers = torch.tensor([[2 * math.cos(2 * math.pi * i / 8), 2 * math.sin(2 * math.pi * i / 8)] for i in range(8)])
def real_batch(n): return centers[torch.randint(0, 8, (n,))] + 0.05 * torch.randn(n, 2)
def mlp(i, o): return nn.Sequential(nn.Linear(i, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, o))

def modes_covered(x):    # a mode counts if >= 2% of samples fall within 0.3 of its centre
    d = torch.cdist(x, centers); near = d.min(1).values < 0.3; k = d.argmin(1)[near]
    cnt = torch.bincount(k, minlength=8).float() / len(x)
    return int((cnt > 0.02).sum()), round(near.float().mean().item(), 3)

def train(kind, iters=6000, bs=256, snaps=(500, 2000, 6000), n_d=1, lr=1e-4):
    torch.manual_seed(0)
    G, D = mlp(8, 2), mlp(2, 1)
    if kind == "wgan-gp":
        oG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.9)); oD = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.9)); n_critic = 5
    else:
        oG = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999)); n_critic = n_d
    zfix = torch.randn(2000, 8); out = {}
    for it in range(1, iters + 1):
        for _ in range(n_critic):
            x, fake = real_batch(bs), G(torch.randn(bs, 8)).detach()
            if kind == "wgan-gp":
                eps = torch.rand(bs, 1); xh = (eps * x + (1 - eps) * fake).requires_grad_(True)
                g = torch.autograd.grad(D(xh).sum(), xh, create_graph=True)[0]
                lossD = D(fake).mean() - D(x).mean() + 10 * ((g.norm(dim=1) - 1) ** 2).mean()
            else:
                lr, lf = D(x), D(fake)
                lossD = F.binary_cross_entropy_with_logits(lr, torch.ones_like(lr)) + F.binary_cross_entropy_with_logits(lf, torch.zeros_like(lf))
            oD.zero_grad(); lossD.backward(); oD.step()
        s = D(G(torch.randn(bs, 8)))
        if kind == "wgan-gp": lossG = -s.mean()
        elif kind == "saturating": lossG = -F.binary_cross_entropy_with_logits(s, torch.zeros_like(s))   # min log(1 - D(G(z)))
        else: lossG = F.binary_cross_entropy_with_logits(s, torch.ones_like(s))                           # min -log D(G(z))
        oG.zero_grad(); lossG.backward(); oG.step()
        if it in snaps:
            with torch.no_grad(): out[it] = G(zfix)
    return out

if __name__ == "__main__":
    t0 = time.time(); kinds = [("saturating", "原始 loss：min log(1−D(G(z)))"), ("nonsat", "non-saturating：max log D(G(z))"), ("wgan-gp", "WGAN-GP")]
    res, snaps = {}, {}
    for k, _ in kinds:
        snaps[k] = train(k, n_d=5); res[k] = {it: modes_covered(x) for it, x in snaps[k].items()}; print(k, res[k], flush=True)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9.4))
    real = real_batch(2000)
    for r, (k, title) in enumerate(kinds):
        for c, it in enumerate([500, 2000, 6000]):
            ax = axes[r, c]; x = snaps[k][it]
            ax.scatter(real[:, 0], real[:, 1], s=2, c="#98a2b3", alpha=.4); ax.scatter(x[:, 0], x[:, 1], s=2, c="#d92d20", alpha=.5)
            ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            m, f = res[k][it]
            ax.set_title(f"第 {it} 步：覆盖 {m}/8 个模式", fontsize=10)
            if c == 0: ax.set_ylabel(title, fontsize=10)
    plt.tight_layout(); plt.savefig(OUT / "toy_8gaussians.png", dpi=140, bbox_inches="tight")
    res["seconds"] = round(time.time() - t0); json.dump(res, open(OUT / "toy_results.json", "w"), indent=1); print("done", res["seconds"])
