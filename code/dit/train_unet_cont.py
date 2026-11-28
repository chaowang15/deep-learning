"""Continue the flow-matching post's Tiny U-Net from 2500 to 7500 steps with the same protocol as the DiT continuation
(AdamW lr 3e-4 after 100 warmup steps, batch 128, grad clip 1.0, EMA 0.999), for a same-step comparison with the DiT.

usage: python3 train_unet_cont.py 5000      (needs /tmp/flow_fmnist.pt from code/flow/fmnist_fm.py)
"""
import copy, json, sys, time
import torch
from train_dit import ROOT, OUT
sys.path.insert(0, str(ROOT / "flow")); sys.path.insert(0, str(ROOT / "diffusion")); sys.path.insert(0, str(ROOT / "vae"))
from fmnist_fm import FlowUNet, fm_loss          # noqa: E402  (same loss, labels dropped 10%)
from common import load_fmnist                    # noqa: E402

steps = int(sys.argv[1])
torch.manual_seed(1); torch.set_num_threads(4)
x_all, y_all = load_fmnist("train"); x_all = x_all * 2 - 1
model = FlowUNet(num_classes=10, in_channels=1, base_channels=48, time_dim=128)
ck = torch.load("/tmp/flow_fmnist.pt"); model.load_state_dict(ck["model"])
ema = copy.deepcopy(model).requires_grad_(False); ema.load_state_dict(ck["ema"])
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
log, t0 = [], time.time()
for step in range(1, steps + 1):
    idx = torch.randint(0, len(x_all), (128,))
    loss = fm_loss(model, x_all[idx], y_all[idx])
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - 0.999)
    if step % 25 == 0:
        log.append((step, loss.item()))
    if step % 250 == 0:
        print("unet_cont", step, round(loss.item(), 4), round(time.time() - t0), "s", flush=True)
torch.save({"ema": ema.state_dict(), "model": model.state_dict()}, "/tmp/unet_cont.pt")
json.dump({"loss": log, "seconds": time.time() - t0}, open(OUT / "train_unet_cont.json", "w"))
print("done", round(time.time() - t0), "s")
