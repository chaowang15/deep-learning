"""MLP GAN and DCGAN on FashionMNIST (non-saturating loss), with D(x)/D(G(z)) curves."""
import json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "code" / "data" / "FashionMNIST" / "raw"
OUT = ROOT / "assets" / "generative-models-2-gan"; OUT.mkdir(parents=True, exist_ok=True)

with open(DATA / "train-images-idx3-ubyte", "rb") as f:
    X = torch.from_numpy(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 127.5 - 1)  # [-1, 1]


class MLPGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        def block(i, o, bn=True):
            return [nn.Linear(i, o)] + ([nn.BatchNorm1d(o)] if bn else []) + [nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*block(z_dim, 128, bn=False), *block(128, 256), *block(256, 512), *block(512, 1024),
                                 nn.Linear(1024, 784), nn.Tanh())
    def forward(self, z): return self.net(z).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))
    def forward(self, x): return self.net(x)          # logits


class DCGenerator(nn.Module):
    def __init__(self, z_dim=100, ch=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(z_dim, 2 * ch * 7 * 7, bias=False), nn.BatchNorm1d(2 * ch * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),   # 7 -> 14
            nn.ConvTranspose2d(ch, 1, 4, 2, 1), nn.Tanh())                                       # 14 -> 28
        self.ch = ch
    def forward(self, z): return self.net(self.fc(z).view(-1, 2 * self.ch, 7, 7))


class DCDiscriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2),                                        # 28 -> 14
            nn.Conv2d(ch, 2 * ch, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * ch), nn.LeakyReLU(0.2),  # 14 -> 7
            nn.Flatten(), nn.Linear(2 * ch * 7 * 7, 1))
    def forward(self, x): return self.net(x)          # logits


def weights_init(m):                                   # DCGAN paper: N(0, 0.02)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)): nn.init.normal_(m.weight, 0, 0.02)
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)): nn.init.normal_(m.weight, 1, 0.02); nn.init.zeros_(m.bias)


def train(name, G, D, epochs, bs=128, z_dim=100):
    torch.manual_seed(0); G.apply(weights_init); D.apply(weights_init)
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    zfix = torch.randn(64, z_dim); log = []; snaps = {}
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X) - bs + 1, bs):
            x = X[perm[i:i + bs]]
            # ---- D step: max log D(x) + log(1 - D(G(z)))
            fake = G(torch.randn(bs, z_dim))
            lr_, lf_ = D(x), D(fake.detach())
            lossD = F.binary_cross_entropy_with_logits(lr_, torch.ones_like(lr_)) + F.binary_cross_entropy_with_logits(lf_, torch.zeros_like(lf_))
            oD.zero_grad(); lossD.backward(); oD.step()
            # ---- G step (non-saturating): max log D(G(z))
            lg = D(fake)
            lossG = F.binary_cross_entropy_with_logits(lg, torch.ones_like(lg))
            oG.zero_grad(); lossG.backward(); oG.step()
            step += 1
            if step % 50 == 0:
                log.append(dict(step=step, lossD=round(lossD.item(), 3), lossG=round(lossG.item(), 3),
                                Dx=round(torch.sigmoid(lr_).mean().item(), 3), DGz=round(torch.sigmoid(lf_).mean().item(), 3)))
        G.eval()
        with torch.no_grad(): snaps[ep + 1] = G(zfix)
        G.train(); print(name, ep + 1, log[-1], flush=True)
    torch.save({"G": G.state_dict(), "snaps": snaps, "log": log}, f"/tmp/gan_{name}.pt")
    return log


if __name__ == "__main__":
    t0 = time.time(); res = {}
    res["mlp"] = train("mlp", MLPGenerator(), MLPDiscriminator(), epochs=10)
    res["dcgan"] = train("dcgan", DCGenerator(), DCDiscriminator(), epochs=10)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(OUT / "fmnist_gan_log.json", "w")); print("done", res["seconds"])
