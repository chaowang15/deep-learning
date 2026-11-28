"""Figures for the FashionMNIST flow matching model (needs /tmp/flow_fmnist.pt from fmnist_fm.py).

fmnist_steps.png   : Euler with N = 1, 2, 5, 10, 20, 50 steps, same eps per row, CFG w = 3
fmnist_solvers.png : Euler vs Heun, squared error to a 200-step Heun reference vs number of network calls
fmnist_curve.png   : training loss
Also checks numerically that one Euler step on the linear path equals one DDIM (eta = 0) step.
"""
import json, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, torch
from fmnist_fm import FlowUNet, ROOT

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
OUT = ROOT.parent / "assets" / "flow-matching"
torch.set_num_threads(4)
model = FlowUNet(num_classes=10, in_channels=1, base_channels=48, time_dim=128)
model.load_state_dict(torch.load("/tmp/flow_fmnist.pt")["ema"]); model.eval()


@torch.no_grad()
def velocity(x, t, labels, w):
    """CFG on the velocity: v = v_uncond + w (v_cond - v_uncond)."""
    B = x.shape[0]
    tb = torch.full((B,), float(t))
    null = torch.full_like(labels, model.null_class_id)
    v = model(torch.cat([x, x]), torch.cat([tb, tb]), torch.cat([labels, null]))
    v_c, v_u = v.chunk(2)
    return v_u + w * (v_c - v_u)


@torch.no_grad()
def sample(eps, labels, num_steps, w=3.0, solver="euler"):
    x = eps.clone()
    ts = torch.linspace(1.0, 0.0, num_steps + 1)
    for i in range(num_steps):
        t, t_next = ts[i], ts[i + 1]
        dt = t_next - t                                   # negative
        v = velocity(x, t, labels, w)
        if solver == "euler" or t_next == 0:              # last Heun step falls back to Euler
            x = x + dt * v
        else:                                             # Heun: average slope at both ends
            x_pred = x + dt * v
            v_next = velocity(x_pred, t_next, labels, w)
            x = x + dt * 0.5 * (v + v_next)
    return x


def nfe(n, solver):
    return n if solver == "euler" else 2 * n - 1


if __name__ == "__main__":
    t0 = time.time()
    res = {}
    # 1. steps grid
    classes = [1, 7, 8, 9, 3, 2]          # trouser, sneaker, bag, ankle boot, dress, pullover
    torch.manual_seed(0)
    eps = torch.randn(len(classes), 1, 28, 28)
    labels = torch.tensor(classes)
    cols = [1, 2, 5, 10, 20, 50]
    imgs = [sample(eps, labels, n) for n in cols]
    fig, axs = plt.subplots(len(classes), len(cols), figsize=(len(cols) * 1.25, len(classes) * 1.3))
    for c, n in enumerate(cols):
        for r in range(len(classes)):
            ax = axs[r, c]; ax.imshow(((imgs[c][r, 0] + 1) / 2).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(f"{n} 步", fontsize=10)
    plt.tight_layout(pad=0.3); plt.savefig(OUT / "fmnist_steps.png", dpi=130, bbox_inches="tight"); plt.close()

    # 2. solver convergence
    torch.manual_seed(1)
    eps = torch.randn(64, 1, 28, 28); labels = torch.arange(64) % 10
    ref = sample(eps, labels, 200, solver="heun")
    conv = {"euler": [], "heun": []}
    for solver, ns in [("euler", [1, 2, 3, 5, 10, 20, 50]), ("heun", [2, 3, 5, 10, 20])]:
        for n in ns:
            x = sample(eps, labels, n, solver=solver)
            conv[solver].append((nfe(n, solver), ((x - ref) ** 2).mean().item()))
    res["convergence"] = conv
    plt.figure(figsize=(5.6, 4))
    for solver, name, c in [("euler", "Euler（一阶）", "#2a6fdb"), ("heun", "Heun（二阶）", "#e0662b")]:
        a = np.array(conv[solver]); plt.loglog(a[:, 0], a[:, 1], "o-", color=c, label=name)
    plt.xlabel("网络调用次数（CFG 的两路算一次）"); plt.ylabel("和 200 步 Heun 结果的均方误差")
    plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "fmnist_solvers.png", dpi=130); plt.close()

    # 3. loss curve
    log = json.load(open(OUT / "fmnist_train.json"))
    a = np.array(log["loss"])
    k = 5; sm = np.convolve(a[:, 1], np.ones(k) / k, mode="valid")
    plt.figure(figsize=(5.6, 3.4)); plt.plot(a[:, 0], a[:, 1], color="#d0d5dd", lw=1)
    plt.plot(a[k - 1:, 0], sm, color="#2a6fdb"); plt.xlabel("训练步数"); plt.ylabel("Flow Matching 损失")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(OUT / "fmnist_curve.png", dpi=130); plt.close()

    # 4. Euler step on the linear path == DDIM (eta=0) step with a_t = 1 - t, s_t = t
    x_t = torch.randn(8, 1, 28, 28); lab = torch.arange(8); t, s = 0.7, 0.4
    v = velocity(x_t, t, lab, 1.0)
    euler = x_t + (s - t) * v
    x0_hat, eps_hat = x_t - t * v, x_t + (1 - t) * v      # convert v -> x0_hat, eps_hat
    ddim = (1 - s) * x0_hat + s * eps_hat                  # DDIM: a_s x0_hat + s_s eps_hat
    res["euler_vs_ddim_maxabs"] = (euler - ddim).abs().max().item()
    res["seconds"] = time.time() - t0
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / "fmnist_results.json", "w"), indent=1)
