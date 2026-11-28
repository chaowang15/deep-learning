"""Small FashionMNIST CNN classifier, used only to score generated samples (accuracy + Frechet distance of its features)."""
import sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vae"))
from common import load_fmnist  # noqa: E402


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)
        self.fc1, self.fc2 = nn.Linear(64 * 7 * 7, 128), nn.Linear(128, 10)

    def features(self, x):                                   # x in [-1, 1], [B, 1, 28, 28] -> [B, 128]
        h = F.max_pool2d(F.relu(self.c1(x)), 2)
        h = F.max_pool2d(F.relu(self.c2(h)), 2)
        return F.relu(self.fc1(h.flatten(1)))

    def forward(self, x):
        return self.fc2(self.features(x))


if __name__ == "__main__":
    torch.manual_seed(0); torch.set_num_threads(1)
    x, y = load_fmnist("train"); x = x * 2 - 1
    xt, yt = load_fmnist("test"); xt = xt * 2 - 1
    m = Classifier(); opt = torch.optim.Adam(m.parameters(), 1e-3); t0 = time.time()
    for ep in range(3):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            j = perm[i:i + 256]
            loss = F.cross_entropy(m(x[j]), y[j]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (m(xt).argmax(1) == yt).float().mean().item()
        print("epoch", ep, "test acc", round(acc, 4), round(time.time() - t0), "s", flush=True)
    torch.save(m.state_dict(), "/tmp/fmnist_clf.pt")
