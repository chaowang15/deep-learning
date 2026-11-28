"""Evaluate the DiT p=4 trained 3x longer (2500 + 5000 steps) and the per-t velocity error of all models.

Needs /tmp/dit_main.pt (2500 steps), /tmp/dit_main_cont.pt (+5000 steps), /tmp/dit_p2.pt, /tmp/flow_fmnist.pt, /tmp/fmnist_clf.pt.
Adds "DiT p=4 (7500)_w3" and "per_t_mse" to eval_results.json and draws dit_longer.png.
"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from eval_figs import (DiTWrap, load_dit, sample, frechet, Classifier, load_fmnist, FlowUNet, OUT, plt as _plt)

torch.set_num_threads(4)
res = json.load(open(OUT / "eval_results.json"))
clf = Classifier(); clf.load_state_dict(torch.load("/tmp/fmnist_clf.pt")); clf.eval()
xt, yt = load_fmnist("test"); xt = xt * 2 - 1
with torch.no_grad():
    ref_feat = clf.features(xt).numpy()
unet = FlowUNet(num_classes=10, in_channels=1, base_channels=48, time_dim=128)
unet.load_state_dict(torch.load("/tmp/flow_fmnist.pt")["ema"]); unet.eval()
models = {"U-Net": unet, "DiT p=4": DiTWrap(load_dit("main")), "DiT p=2": DiTWrap(load_dit("p2")),
          "DiT p=4 (7500)": DiTWrap(load_dit("main_cont"))}

# per-t velocity MSE on 500 test images (same eps for every model)
torch.manual_seed(0)
x0, y = xt[:500], yt[:500]; eps = torch.randn_like(x0)
res["per_t_mse"] = {}
for t in [0.02, 0.05, 0.1, 0.3, 0.5, 0.8, 0.95]:
    x_t = (1 - t) * x0 + t * eps
    with torch.no_grad():
        res["per_t_mse"][str(t)] = {n: float(((m(x_t, torch.full((500,), t), y) - (eps - x0)) ** 2).mean())
                                    for n, m in models.items()}
    print(t, res["per_t_mse"][str(t)], flush=True)

# samples of the longer-trained DiT, same eps / labels as eval_figs.py
torch.manual_seed(0)
eps = torch.randn(1000, 1, 28, 28); labels = torch.arange(1000) % 10
m = models["DiT p=4 (7500)"]
xs = torch.cat([sample(m, eps[i:i + 500], labels[i:i + 500], 50, 3.0) for i in range(0, 1000, 500)])
with torch.no_grad():
    res["DiT p=4 (7500)_w3"] = {"acc": float((clf(xs).argmax(1) == labels).float().mean()),
                                "fd": frechet(clf.features(xs).numpy(), ref_feat)}
print(res["DiT p=4 (7500)_w3"], flush=True)
x25 = torch.cat([sample(models["DiT p=4"], eps[:80], labels[:80], 50, 3.0)])
fig, axs = plt.subplots(1, 2, figsize=(10, 5.6))
for ax, xs_, title in [(axs[0], x25, "DiT p=4，2500 步"), (axs[1], xs[:80], "DiT p=4，7500 步")]:
    g = xs_.view(8, 10, 28, 28).permute(1, 2, 0, 3).reshape(280, 224)
    ax.imshow((g + 1) / 2, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=12)
plt.tight_layout(); plt.savefig(OUT / "dit_longer.png", dpi=120, bbox_inches="tight"); plt.close()
json.dump(res, open(OUT / "eval_results.json", "w"), indent=1)
