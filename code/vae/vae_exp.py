"""MLP VAE on FashionMNIST: correct loss scaling vs. MSE(mean) + KL, and a 2-D latent VAE."""
import json, time, torch, torch.nn as nn, torch.nn.functional as F
from common import load_fmnist, OUT

torch.manual_seed(0)
xtr, ytr = load_fmnist("train"); xte, yte = load_fmnist("test")


class VAE(nn.Module):
    def __init__(self, z_dim=20, h=400):
        super().__init__()
        self.fc1 = nn.Linear(784, h)
        self.fc_mu = nn.Linear(h, z_dim)
        self.fc_logvar = nn.Linear(h, z_dim)
        self.fc3 = nn.Linear(z_dim, h)
        self.fc4 = nn.Linear(h, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode_logits(self, z):
        return self.fc4(F.relu(self.fc3(z)))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode_logits(z), mu, logvar


def kl_per_dim(mu, logvar):          # [B, d]
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


def loss_fn(kind, logits, x, mu, logvar):
    x = x.view(-1, 784); B = x.shape[0]
    kl = kl_per_dim(mu, logvar).sum()
    if kind == "bce_sum":            # -ELBO, Bernoulli likelihood
        rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    elif kind == "mse_mean":         # common mistake: per-pixel mean MSE + summed KL
        rec = F.mse_loss(torch.sigmoid(logits), x, reduction="mean") * B
    return (rec + kl) / B, rec / B, kl / B


def train(kind, z_dim=20, epochs=15, bs=128):
    torch.manual_seed(0)
    m = VAE(z_dim); opt = torch.optim.Adam(m.parameters(), 1e-3)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            x = xtr[perm[i:i + bs]]
            logits, mu, logvar = m(x)
            loss, _, _ = loss_fn(kind, logits, x, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            logits, mu, logvar = m(xte)
            _, rec, kl = loss_fn("bce_sum", logits, xte, mu, logvar)   # evaluate everything with true -ELBO terms
            active = int((kl_per_dim(mu, logvar).mean(0) > 0.01).sum())
        hist.append(dict(epoch=ep + 1, bce=round(rec.item(), 2), kl=round(kl.item(), 2), active=active))
        print(kind, z_dim, hist[-1], flush=True)
    return m, hist


if __name__ == "__main__":
    t0 = time.time(); res = {}
    models = {}
    for kind in ["bce_sum", "mse_mean"]:
        models[kind], res[kind] = train(kind)
        torch.save(models[kind].state_dict(), f"/tmp/vae_{kind}.pt")
    m2, res["bce_sum_z2"] = train("bce_sum", z_dim=2)
    torch.save(m2.state_dict(), "/tmp/vae_z2.pt")
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(OUT / "vae_results.json", "w"), indent=1)
    print("done", res["seconds"])
