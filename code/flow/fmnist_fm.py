"""Class-conditional flow matching on FashionMNIST with the same Tiny U-Net as the DDPM post (CPU, ~19 min).

x_t = (1 - t) x0 + t eps,  target u = eps - x0,  t ~ U(0, 1);  labels dropped 10% of the time for CFG.
"""
import copy, json, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diffusion"))
sys.path.insert(0, str(ROOT / "vae"))
from src.unet import ClassConditionedTinyUNet          # noqa: E402
from common import load_fmnist                         # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(4)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
BS, LR, P_DROP, TIME_SCALE = 128, 1e-3, 0.1, 1000.0


class FlowUNet(ClassConditionedTinyUNet):
    """v_theta(x_t, t, y). t is continuous in [0, 1]; scaled by 1000 before the sinusoidal embedding."""
    def forward(self, x_t, t, labels=None):
        return super().forward(x_t, t * TIME_SCALE, labels)


def fm_loss(model, x0, labels):
    B = x0.shape[0]
    eps = torch.randn_like(x0)                                        # eps ~ N(0, I)
    t = torch.rand(B)                                                 # t ~ U(0, 1)
    tt = t.view(B, 1, 1, 1)
    x_t = (1 - tt) * x0 + tt * eps                                    # x_t = (1-t) x0 + t eps
    u = eps - x0                                                      # target velocity
    drop = torch.rand(B) < P_DROP
    labels = torch.where(drop, torch.full_like(labels, model.null_class_id), labels)
    return F.mse_loss(model(x_t, t, labels), u)


if __name__ == "__main__":
    x_all, y_all = load_fmnist("train")
    x_all = x_all * 2 - 1                                             # [0,1] -> [-1,1]
    model = FlowUNet(num_classes=10, in_channels=1, base_channels=48, time_dim=128)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05)
    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(x_all), (BS,))
        loss = fm_loss(model, x_all[idx], y_all[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - 0.999)
        if step % 50 == 0:
            log.append((step, loss.item()))
        if step % 250 == 0:
            print(step, round(loss.item(), 4), round(time.time() - t0), "s", flush=True)
    torch.save({"ema": ema.state_dict(), "model": model.state_dict()}, "/tmp/flow_fmnist.pt")
    json.dump({"loss": log, "seconds": time.time() - t0, "params": sum(p.numel() for p in model.parameters())},
              open(ROOT.parent / "assets" / "flow-matching" / "fmnist_train.json", "w"))
    print("done", round(time.time() - t0), "s")
