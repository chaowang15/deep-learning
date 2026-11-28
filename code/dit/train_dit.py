"""Train a small class-conditional DiT on FashionMNIST with the flow matching loss (same loss as the flow-matching post).

usage: python3 train_dit.py <cond_mode> <steps> <tag> [patch] [hidden] [depth] [heads]
"""
import copy, json, os, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vae"))
from common import load_fmnist                                   # noqa: E402
from dit import DiT                                              # noqa: E402

OUT = ROOT.parent / "assets" / "dit"
TIME_SCALE = 1000.0


def make_model(cond_mode="adaln_zero", patch=4, hidden=192, depth=6, heads=3):
    return DiT(input_size=28, patch_size=patch, in_channels=1, hidden_size=hidden, depth=depth, num_heads=heads,
               num_classes=10, class_dropout_prob=0.1, learn_sigma=False, cond_mode=cond_mode)


def fm_loss(model, x0, y):
    """Rectified-flow loss: x_t = (1-t) x0 + t eps, target eps - x0. Label dropout happens inside the model."""
    B = x0.shape[0]
    eps = torch.randn_like(x0)
    t = torch.rand(B)
    tt = t.view(B, 1, 1, 1)
    x_t = (1 - tt) * x0 + tt * eps
    return F.mse_loss(model(x_t, t * TIME_SCALE, y), eps - x0)


if __name__ == "__main__":
    mode, steps, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    arch = [int(a) for a in sys.argv[4:8]] if len(sys.argv) > 4 else [4, 192, 6, 3]
    torch.manual_seed(0); torch.set_num_threads(4)
    x_all, y_all = load_fmnist("train")
    x_all = x_all * 2 - 1
    model = make_model(mode, *arch)
    ema = copy.deepcopy(model).requires_grad_(False)
    resume = os.environ.get("RESUME")                         # continue from a checkpoint (optimizer restarts)
    if resume:
        ck = torch.load(resume); model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(x_all), (128,))
        loss = fm_loss(model, x_all[idx], y_all[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - 0.999)
        if step % 25 == 0:
            log.append((step, loss.item()))
        if step % 250 == 0:
            print(tag, step, round(loss.item(), 4), round(time.time() - t0), "s", flush=True)
    torch.save({"ema": ema.state_dict(), "model": model.state_dict(), "arch": arch, "mode": mode}, f"/tmp/dit_{tag}.pt")
    json.dump({"loss": log, "seconds": time.time() - t0, "params": sum(p.numel() for p in model.parameters()),
               "mode": mode, "arch": arch}, open(OUT / f"train_{tag}.json", "w"))
    print(tag, "done", round(time.time() - t0), "s", flush=True)
