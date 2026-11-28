"""Compare the DiT with the flow-matching post's Tiny U-Net (same loss, same sampler) and draw the figures.

Needs /tmp/dit_main.pt, /tmp/dit_abl_*.pt (train_dit.py), /tmp/flow_fmnist.pt (code/flow/fmnist_fm.py), /tmp/fmnist_clf.pt.
Outputs: dit_samples.png, dit_vs_unet.png, dit_ablation.png, dit_attention.png, eval_results.json
"""
import json, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, torch
from train_dit import make_model, OUT, ROOT, TIME_SCALE
from classifier import Classifier, load_fmnist
sys.path.insert(0, str(ROOT / "flow")); sys.path.insert(0, str(ROOT / "diffusion"))
from fmnist_fm import FlowUNet  # noqa: E402

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Zen Hei"]
torch.set_num_threads(4)


def load_dit(tag):
    ck = torch.load(f"/tmp/dit_{tag}.pt")
    m = make_model(ck["mode"], *ck["arch"]); m.load_state_dict(ck["ema"]); return m.eval()


class DiTWrap(torch.nn.Module):            # same call signature as FlowUNet: (x_t, t in [0,1], labels)
    def __init__(self, m):
        super().__init__(); self.m = m; self.null_class_id = 10
    def forward(self, x, t, y):
        return self.m(x, t * TIME_SCALE, y)


@torch.no_grad()
def sample(model, eps, labels, n=50, w=3.0):
    x = eps.clone(); ts = torch.linspace(1.0, 0.0, n + 1)
    null = torch.full_like(labels, model.null_class_id)
    for i in range(n):
        tb = torch.full((len(x),), float(ts[i]))
        v = model(torch.cat([x, x]), torch.cat([tb, tb]), torch.cat([labels, null]))
        v_c, v_u = v.chunk(2)
        x = x + (ts[i + 1] - ts[i]) * (v_u + w * (v_c - v_u))
    return x.clamp(-1, 1)


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # tr(sqrt(S1 S2)) = sum of sqrt of the eigenvalues of S1 S2 (they are real and >= 0)
    tr_covmean = np.sqrt(np.clip(np.linalg.eigvals(s1 @ s2).real, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


if __name__ == "__main__":
    t0 = time.time(); res = {}
    clf = Classifier(); clf.load_state_dict(torch.load("/tmp/fmnist_clf.pt")); clf.eval()
    xt, yt = load_fmnist("test"); xt = xt * 2 - 1
    with torch.no_grad():
        ref_feat = clf.features(xt).numpy()
    unet = FlowUNet(num_classes=10, in_channels=1, base_channels=48, time_dim=128)
    unet.load_state_dict(torch.load("/tmp/flow_fmnist.pt")["ema"]); unet.eval()
    models = {"U-Net": unet, "DiT p=4": DiTWrap(load_dit("main")), "DiT p=2": DiTWrap(load_dit("p2"))}
    cache_path = OUT / "eval_results.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    cache = {k.replace("DiT_", "DiT p=4_"): v for k, v in cache.items()}
    res["params"] = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    torch.manual_seed(0)
    eps = torch.randn(1000, 1, 28, 28); labels = torch.arange(1000) % 10
    from torch.utils.flop_counter import FlopCounterMode
    res["gmacs"] = {}
    for name, m in models.items():
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(eps[:1], torch.tensor([0.5]), labels[:1])
        res["gmacs"][name] = fc.get_total_flops() / 2 / 1e9
    for name, m in models.items():
        for w in [1.0, 3.0]:
            key = f"{name}_w{w:g}"
            if key in cache and w != 3.0:
                res[key] = cache[key]; continue
            ts = time.time()
            xs = torch.cat([sample(m, eps[i:i + 500], labels[i:i + 500], 50, w) for i in range(0, 1000, 500)])
            with torch.no_grad():
                acc = (clf(xs).argmax(1) == labels).float().mean().item()
                fd = frechet(clf.features(xs).numpy(), ref_feat)
            res[f"{name}_w{w:g}"] = {"acc": acc, "fd": fd, "sample_seconds": time.time() - ts}
            print(name, w, res[f"{name}_w{w:g}"], flush=True)
            if w == 3.0:
                res[f"{name}_grid"] = xs[:80]
    # sample grids: rows = classes, same eps for U-Net and DiT
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, name in zip(axs, models):
        g = res.pop(f"{name}_grid").view(8, 10, 28, 28).permute(1, 2, 0, 3).reshape(280, 224)
        ax.imshow((g + 1) / 2, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}（{res['params'][name] / 1e6:.2f}M 参数）", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / "dit_vs_unet.png", dpi=120, bbox_inches="tight"); plt.close()

    # loss curves: main run vs U-Net from the flow post; ablation (first 400 steps)
    def curve(path):
        a = np.array(json.load(open(path))["loss"], dtype=float); return a
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8))
    u = curve(ROOT.parent / "assets" / "flow-matching" / "fmnist_train.json"); d = curve(OUT / "train_main.json")
    d2 = curve(OUT / "train_p2.json")
    k = 4
    for a, name, c in [(u, "U-Net（0.46M）", "#98a2b3"), (d, "DiT p=4（4.2M）", "#2a6fdb"), (d2, "DiT p=2（1.9M）", "#e0662b")]:
        axs[0].plot(a[k - 1:, 0], np.convolve(a[:, 1], np.ones(k) / k, "valid"), color=c, label=name)
    axs[0].set_ylim(0.2, 0.6); axs[0].set_xlabel("训练步数"); axs[0].set_ylabel("Flow Matching 损失"); axs[0].legend(); axs[0].grid(alpha=0.3)
    axs[0].set_title("同样的损失和 batch，训练 2500 步")
    abl = {"adaLN-Zero": d[d[:, 0] <= 400]}
    for m in ["adaln", "in_context", "cross_attn"]:
        abl[{"adaln": "adaLN", "in_context": "In-context", "cross_attn": "Cross-attention"}[m]] = curve(OUT / f"train_abl_{m}.json")
    res["ablation_last100_mean"] = {}
    for (name, a), c in zip(abl.items(), ["#2a6fdb", "#12b76a", "#e0662b", "#9e77ed"]):
        axs[1].plot(a[k - 1:, 0], np.convolve(a[:, 1], np.ones(k) / k, "valid"), color=c, label=name)
        res["ablation_last100_mean"][name] = float(a[a[:, 0] > 300, 1].mean())
    axs[1].set_ylim(0.2, 0.8); axs[1].set_xlabel("训练步数"); axs[1].legend(); axs[1].grid(alpha=0.3)
    axs[1].set_title("四种条件注入方式（前 400 步）")
    plt.tight_layout(); plt.savefig(OUT / "dit_curves.png", dpi=130); plt.close()
    # gate magnitudes: how far the adaLN-Zero gates moved away from 0 during training
    m = load_dit("p2")
    with torch.no_grad():
        c = m.t_embedder(torch.tensor([500.0] * 10)) + m.y_embedder(torch.arange(10), False)
        gates = []
        for blk in m.blocks:
            _, _, g1, _, _, g2 = blk.adaLN_modulation(c).chunk(6, dim=1)       # α1, α2: [10, D]
            gates.append([float(g1.abs().mean()), float(g2.abs().mean())])
        res["gate_abs_mean_per_block"] = gates                                   # all 0 at initialization
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / "eval_results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items()}, indent=1))
