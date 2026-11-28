import torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parents[2] / "assets" / "generative-models-2-gan"
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
ck = {k: torch.load(f"/tmp/gan_{k}.pt") for k in ["mlp", "dcgan"]}

def grid(x, n=8):  # [64,1,28,28] in [-1,1]
    x = (x[:n * 4] + 1) / 2
    return x[:, 0].view(4, n, 28, 28).permute(0, 2, 1, 3).reshape(4 * 28, n * 28)

fig, axes = plt.subplots(2, 3, figsize=(12, 4.6))
for r, (k, name) in enumerate([("mlp", "MLP GAN"), ("dcgan", "DCGAN")]):
    for c, ep in enumerate([1, 3, 10]):
        ax = axes[r, c]; ax.imshow(grid(ck[k]["snaps"][ep]), cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}，第 {ep} 个 epoch", fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "fmnist_samples.png", dpi=140, bbox_inches="tight"); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, (k, name) in zip(axes, [("mlp", "MLP GAN"), ("dcgan", "DCGAN")]):
    log = ck[k]["log"]; s = [l["step"] for l in log]
    sm = lambda v: np.convolve(v, np.ones(5) / 5, mode="valid")
    s = s[2:-2]
    ax.plot(s, sm([l["Dx"] for l in log]), label="D(x)：真实样本", color="#2a6fdb")
    ax.plot(s, sm([l["DGz"] for l in log]), label="D(G(z))：生成样本", color="#d92d20")
    ax.axhline(0.5, color="#98a2b3", ls="--", lw=1); ax.set_title(name); ax.set_xlabel("训练步数"); ax.set_ylim(0, 1)
axes[0].legend(fontsize=9); axes[0].set_ylabel("判别器输出（batch 平均）")
plt.tight_layout(); plt.savefig(OUT / "fmnist_curves.png", dpi=140, bbox_inches="tight"); print("ok")
