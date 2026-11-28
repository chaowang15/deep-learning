"""Same-step comparison at 7500 steps: U-Net (train_unet_cont.py) vs DiT p=4 (train_dit.py ... main_cont).

Adds "U-Net (7500)_w1/_w3", "DiT p=4 (7500)_w1" and per-t MSE of the U-Net at 7500 to eval_results.json,
draws dit_vs_unet_7500.png (samples) and dit_curves_7500.png (training loss of both runs, steps 2500-7500).
"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, torch
from eval_figs import DiTWrap, load_dit, sample, frechet, Classifier, load_fmnist, FlowUNet, OUT

torch.set_num_threads(4)
res = json.load(open(OUT / "eval_results.json"))
clf = Classifier(); clf.load_state_dict(torch.load("/tmp/fmnist_clf.pt")); clf.eval()
xt, yt = load_fmnist("test"); xt = xt * 2 - 1
with torch.no_grad():
    ref_feat = clf.features(xt).numpy()
unet = FlowUNet(num_classes=10, in_channels=1, base_channels=48, time_dim=128)
unet.load_state_dict(torch.load("/tmp/unet_cont.pt")["ema"]); unet.eval()
dit = DiTWrap(load_dit("main_cont"))
models = {"U-Net (7500)": unet, "DiT p=4 (7500)": dit}

torch.manual_seed(0)                                     # same test images / eps as eval_longer.py
x0, y = xt[:500], yt[:500]; e = torch.randn_like(x0)
for t in [0.02, 0.05, 0.1, 0.3, 0.5, 0.8, 0.95]:
    x_t = (1 - t) * x0 + t * e
    with torch.no_grad():
        res["per_t_mse"][str(t)]["U-Net (7500)"] = float(((unet(x_t, torch.full((500,), t), y) - (e - x0)) ** 2).mean())
print({k: round(v["U-Net (7500)"], 3) for k, v in res["per_t_mse"].items()}, flush=True)

torch.manual_seed(0)                                     # same eps / labels as eval_figs.py
eps = torch.randn(1000, 1, 28, 28); labels = torch.arange(1000) % 10
grids = {}
for name, m in models.items():
    for w in [1.0, 3.0]:
        key = f"{name}_w{w:g}"
        if key in res and w != 3.0 or (key in res and name.startswith("DiT")):
            continue
        xs = torch.cat([sample(m, eps[i:i + 500], labels[i:i + 500], 50, w) for i in range(0, 1000, 500)])
        with torch.no_grad():
            res[key] = {"acc": float((clf(xs).argmax(1) == labels).float().mean()),
                        "fd": frechet(clf.features(xs).numpy(), ref_feat)}
        print(key, res[key], flush=True)
    grids[name] = sample(m, eps[:80], labels[:80], 50, 3.0)

fig, axs = plt.subplots(1, 2, figsize=(10, 5.6))
for ax, name in zip(axs, models):
    g = grids[name].view(8, 10, 28, 28).permute(1, 2, 0, 3).reshape(280, 224)
    ax.imshow((g + 1) / 2, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(name.replace(" (7500)", "，7500 步"), fontsize=12)
plt.tight_layout(); plt.savefig(OUT / "dit_vs_unet_7500.png", dpi=120, bbox_inches="tight"); plt.close()

def curve(path, offset):
    a = np.array(json.load(open(path))["loss"], dtype=float); a[:, 0] += offset; return a
k = 8
plt.figure(figsize=(6.4, 3.6))
for path, name, c in [(OUT / "train_unet_cont.json", "U-Net（0.46M）", "#98a2b3"), (OUT / "train_main_cont.json", "DiT p=4（4.2M）", "#2a6fdb")]:
    a = curve(path, 2500)
    plt.plot(a[k - 1:, 0], np.convolve(a[:, 1], np.ones(k) / k, "valid"), color=c, label=name)
    res.setdefault("cont_loss_last500_mean", {})[name] = float(a[a[:, 0] > 7000, 1].mean())
plt.xlabel("训练步数"); plt.ylabel("Flow Matching 损失"); plt.legend(); plt.grid(alpha=0.3)
plt.title("第 2500–7500 步（同样的优化器和学习率）"); plt.tight_layout()
plt.savefig(OUT / "dit_curves_7500.png", dpi=130); plt.close()
json.dump(res, open(OUT / "eval_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if "7500" in k or k == "cont_loss_last500_mean"}, indent=1))
