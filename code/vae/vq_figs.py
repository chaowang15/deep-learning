"""Figures for the VQ-VAE section (needs /tmp/vqvae.pt, /tmp/vq_prior.pt, /tmp/vae_bce_sum.pt)."""
import torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_fmnist, OUT
from vae_exp import VAE
from vqvae_exp import VQVAE, TinyGPT

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
xte, _ = load_fmnist("test")
vae = VAE(20); vae.load_state_dict(torch.load("/tmp/vae_bce_sum.pt")); vae.eval()
vq = VQVAE(64); vq.load_state_dict(torch.load("/tmp/vqvae.pt")); vq.eval()
prior = TinyGPT(64); prior.load_state_dict(torch.load("/tmp/vq_prior.pt")); prior.eval()


def grid(imgs, nrow):    # [N, 1, 28, 28] -> image
    n = imgs.shape[0]; ncol = n // nrow
    return imgs[:, 0].view(nrow, ncol, 28, 28).permute(0, 2, 1, 3).reshape(nrow * 28, ncol * 28)


def show(rows, labels, path, h):
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, h))
    for ax, r, lab in zip(axes, rows, labels):
        ax.imshow(r, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


x = xte[torch.arange(10) * 997 + 11]
with torch.no_grad():
    mu, _ = vae.encode(x.view(-1, 784))
    r_vae = torch.sigmoid(vae.decode_logits(mu)).view(-1, 1, 28, 28)
    logits, _, idx = vq(x)
    r_vq = torch.sigmoid(logits)
show([grid(x, 1), grid(r_vae, 1), grid(r_vq, 1)], ["测试图像", "VAE 重建（d = 20）", "VQ-VAE 重建（7×7 个索引）"],
     OUT / "vq_recon.png", 3.0)

torch.manual_seed(0)
with torch.no_grad():
    s_prior = vq.decode_indices(prior.sample(20).view(-1, 7, 7))
    s_unif = vq.decode_indices(torch.randint(0, 64, (10, 7, 7)))
show([grid(s_prior, 2), grid(s_unif, 1)], ["学到的先验\n（Transformer）", "均匀随机索引"], OUT / "vq_samples.png", 3.6)
print("ok")
