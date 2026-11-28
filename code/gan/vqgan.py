"""Minimal VQGAN-style training step: VQ-VAE + PatchGAN discriminator (hinge) + adaptive weight.
Reuses the VQ-VAE from code/vae. Perceptual (LPIPS) loss is omitted (needs pretrained VGG weights)."""
import sys, time, json, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vae"))
from vqvae_exp import VQVAE, load_fmnist


class PatchDiscriminator(nn.Module):
    """Outputs a grid of real/fake logits, one per receptive-field patch (no final pooling)."""
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2),                                  # 28 -> 14
            nn.Conv2d(ch, 2 * ch, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * ch), nn.LeakyReLU(0.2),   # 14 -> 7
            nn.Conv2d(2 * ch, 1, 3, 1, 1))                                                 # [B, 1, 7, 7] logits
    def forward(self, x): return self.net(x)


def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (F.relu(1 - logits_real).mean() + F.relu(1 + logits_fake).mean())


def adaptive_weight(rec_loss, g_loss, last_layer, delta=1e-4):
    """lambda = ||grad_{G_L} L_rec|| / (||grad_{G_L} L_GAN|| + delta), G_L = last decoder layer weight."""
    g_rec = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    g_gan = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    return (g_rec.norm() / (g_gan.norm() + delta)).clamp(0, 1e4).detach()


def train_step(model, disc, opt_ae, opt_d, x, use_gan, disc_weight=0.5):
    # ---- autoencoder (generator) step
    logits, vq_loss, _ = model(x)
    x_rec = torch.sigmoid(logits)
    rec_loss = (x_rec - x).abs().mean()                      # L1 (VQGAN: L1 + LPIPS)
    loss = rec_loss + vq_loss
    lam = torch.tensor(0.)
    if use_gan:
        g_loss = -disc(x_rec * 2 - 1).mean()                 # generator hinge loss
        lam = adaptive_weight(rec_loss, g_loss, model.decoder[-1].weight)
        loss = loss + disc_weight * lam * g_loss
    opt_ae.zero_grad(); loss.backward(); opt_ae.step()
    # ---- discriminator step
    d_loss = torch.tensor(0.)
    if use_gan:
        d_loss = hinge_d_loss(disc(x * 2 - 1), disc(x_rec.detach() * 2 - 1))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
    return rec_loss.item(), lam.item(), d_loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    xtr, _ = load_fmnist("train")
    model = VQVAE(64); model.load_state_dict(torch.load("/tmp/vqvae.pt"))   # warm start = "disc_start" phase done
    disc = PatchDiscriminator()
    opt_ae = torch.optim.Adam(model.parameters(), 1e-4, betas=(0.5, 0.9)); opt_d = torch.optim.Adam(disc.parameters(), 1e-4, betas=(0.5, 0.9))
    log = []; t0 = time.time()
    for it in range(int(sys.argv[1]) if len(sys.argv) > 1 else 50):
        x = xtr[torch.randint(0, len(xtr), (64,))]
        r, lam, d = train_step(model, disc, opt_ae, opt_d, x, use_gan=True)
        if it % 10 == 0: log.append(dict(it=it, l1=round(r, 4), lam=round(lam, 3), d_loss=round(d, 3))); print(log[-1], flush=True)
    print("seconds", round(time.time() - t0))
