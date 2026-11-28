"""Parameter count and Gflops of every paper config (256x256 images -> 32x32x4 latents)."""
import json, torch
from torch.utils.flop_counter import FlopCounterMode
from dit import DiT_model
from pathlib import Path
res = {}
for name in ["S", "B", "L", "XL"]:
    for p in [2, 4, 8]:
        m = DiT_model(name, p).eval()
        params = sum(q.numel() for q in m.parameters())
        z, t, y = torch.randn(1, 4, 32, 32), torch.tensor([500]), torch.tensor([1])
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            out = m(z, t, y)
        res[f"DiT-{name}/{p}"] = {"params_M": params / 1e6, "gmacs": fc.get_total_flops() / 2 / 1e9, "out": list(out.shape)}
        print(f"DiT-{name}/{p}", res[f"DiT-{name}/{p}"], flush=True)
json.dump(res, open(Path(__file__).resolve().parents[2] / "assets" / "dit" / "paper_configs.json", "w"), indent=1)
