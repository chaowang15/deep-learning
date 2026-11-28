"""Gaussian decoder with fixed variance sigma^2: loss = ||x - x_hat||^2 / (2 sigma^2) + KL (per image, summed)."""
import json, torch, torch.nn.functional as F
from common import load_fmnist, OUT
from vae_exp import VAE, kl_per_dim

xtr, _ = load_fmnist("train"); xte, _ = load_fmnist("test")
res = []
for s2 in [0.01, 0.1, 0.5, 2.0, 392.0]:
    torch.manual_seed(0); m = VAE(20); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(10):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            x = xtr[perm[i:i + 128]].view(-1, 784)
            logits, mu, logvar = m(x)
            sse = (torch.sigmoid(logits) - x).pow(2).sum(1)
            loss = (sse / (2 * s2) + kl_per_dim(mu, logvar).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mu, logvar = m.encode(xte.view(-1, 784))
        sse = (torch.sigmoid(m.decode_logits(mu)) - xte.view(-1, 784)).pow(2).sum(1).mean().item()
        kl = kl_per_dim(mu, logvar)
        r = dict(sigma2=s2, beta=2 * s2, sse=round(sse, 2), kl=round(kl.sum(1).mean().item(), 2), active=int((kl.mean(0) > 0.01).sum()))
    res.append(r); print(r, flush=True)
json.dump(res, open(OUT / "vae_sigma_results.json", "w"), indent=1)
