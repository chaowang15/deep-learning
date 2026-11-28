"""Figures for the VAE section (needs /tmp/vae_*.pt from vae_exp.py)."""
import torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_fmnist, OUT
from vae_exp import VAE

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
xte, yte = load_fmnist("test")
torch.manual_seed(1)
idx = torch.arange(0, 10000, 1000)[:10] + 3
x = xte[idx]


def row(imgs):           # [n, 784] -> [28, 28n]
    return imgs.view(-1, 28, 28).permute(1, 0, 2).reshape(28, -1)


def load(name, z=20):
    m = VAE(z); m.load_state_dict(torch.load(f"/tmp/vae_{name}.pt")); m.eval(); return m


rows, labels = [x.view(-1, 784)], ["测试图像"]
with torch.no_grad():
    zs = torch.randn(10, 20)
    for name, tag in [("bce_sum", "BCE(sum) + KL"), ("mse_mean", "MSE(mean) + KL")]:
        m = load(name)
        mu, logvar = m.encode(x.view(-1, 784))
        rows.append(torch.sigmoid(m.decode_logits(mu))); labels.append(f"重建：{tag}")
        rows.append(torch.sigmoid(m.decode_logits(zs))); labels.append(f"从 N(0,I) 采样：{tag}")
fig, axes = plt.subplots(len(rows), 1, figsize=(8, 4.6))
for ax, r, lab in zip(axes, rows, labels):
    ax.imshow(row(r), cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "vae_recon_samples.png", dpi=150, bbox_inches="tight"); plt.close()

# 2-D latent: encoder means coloured by class, and decoder outputs on a grid of N(0,1) quantiles
names = ["T恤", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]
m = VAE(2); m.load_state_dict(torch.load("/tmp/vae_z2.pt")); m.eval()
with torch.no_grad():
    mu, _ = m.encode(xte.view(-1, 784))
    n = 15
    q = torch.linspace(0.03, 0.97, n)
    g = torch.erfinv(2 * q - 1) * 2 ** 0.5                       # standard-normal quantiles
    zz = torch.stack(torch.meshgrid(g, g.flip(0), indexing="xy"), -1).view(-1, 2)
    imgs = torch.sigmoid(m.decode_logits(zz)).view(n, n, 28, 28).permute(0, 2, 1, 3).reshape(n * 28, n * 28)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.2))
sc = a.scatter(mu[:, 0], mu[:, 1], c=yte, cmap="tab10", s=2, alpha=0.6)
a.set_xlim(-3.5, 3.5); a.set_ylim(-3.5, 3.5); a.set_aspect("equal"); a.set_title("测试集的编码均值 μ(x)（按类别着色）")
handles = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab10(i / 9), label=names[i]) for i in range(10)]
a.legend(handles=handles, fontsize=7, loc="upper right", markerscale=0.8)
b.imshow(imgs, cmap="gray"); b.set_xticks([]); b.set_yticks([]); b.set_title("在 z 平面网格上解码（网格取 N(0,1) 分位点）")
plt.tight_layout(); plt.savefig(OUT / "vae_latent2d.png", dpi=150, bbox_inches="tight"); plt.close()
print("ok")
