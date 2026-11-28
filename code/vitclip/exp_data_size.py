"""Experiment 1: tiny ViT vs a small CNN as the amount of training data grows
(1k / 6k / 60k FashionMNIST images). Same optimizer, steps, and augmentation
for both models. Shows the inductive-bias effect from the ViT paper at toy scale.

usage: python3 exp_data_size.py [steps]       (~2-3 min per run on 4 CPU cores)
writes assets/vit-clip/data_size.json and /tmp/vit_60000.pt (used by figs.py)
"""
import json, math, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "vae"))
from common import load_fmnist  # noqa: E402
from vit import ViT  # noqa: E402


class SmallCNN(nn.Module):
    """Two conv blocks + MLP head; built-in locality and translation equivariance."""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Flatten(), nn.Linear(64 * 49, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.f(x)


def tiny_vit():   # 28x28, patch 4 -> 7x7 = 49 tokens (+[CLS]), width 64, 6 blocks
    return ViT(img=28, patch=4, in_ch=1, dim=64, depth=6, heads=4, n_classes=10)


def augment(x):   # random shift by up to 2 px (pad + crop) and horizontal flip
    B = len(x)
    xp = F.pad(x, (2, 2, 2, 2))
    i, j = torch.randint(0, 5, (2,))
    x = xp[:, :, i:i + 28, j:j + 28]
    flip = torch.rand(B) < 0.5
    x[flip] = x[flip].flip(-1)
    return x


def train(model, x, y, steps, bs=128, lr=1e-3, warm=100):
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) *
                                              0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    model.train()
    for s in range(steps):
        j = torch.randint(0, len(x), (bs,))
        loss = F.cross_entropy(model(augment(x[j])), y[j])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    return sum((model(x[i:i + 1000]).argmax(1) == y[i:i + 1000]).sum().item()
               for i in range(0, len(x), 1000)) / len(x)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    x, y = load_fmnist("train"); x = x * 2 - 1
    xt, yt = load_fmnist("test"); xt = xt * 2 - 1
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(x), generator=g)
    res = {"steps": steps, "runs": []}
    for n in (1000, 6000, 60000):
        idx = perm[:n]
        for name, make in (("CNN", SmallCNN), ("ViT", tiny_vit)):
            torch.manual_seed(0)
            m = make(); t0 = time.time()
            last = train(m, x[idx], y[idx], steps)
            r = {"model": name, "n": n, "params": sum(p.numel() for p in m.parameters()),
                 "train_acc": evaluate(m, x[idx][:5000], y[idx][:5000]),
                 "test_acc": evaluate(m, xt, yt), "last_loss": last, "sec": round(time.time() - t0)}
            print(r, flush=True); res["runs"].append(r)
            if name == "ViT" and n == 60000:
                torch.save(m.state_dict(), "/tmp/vit_60000.pt")
    out = ROOT / "assets" / "vit-clip"; out.mkdir(parents=True, exist_ok=True)
    (out / "data_size.json").write_text(json.dumps(res, indent=1))
