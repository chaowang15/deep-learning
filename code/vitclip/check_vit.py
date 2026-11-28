"""Check our ViT against torchvision: copy random-init weights, compare logits,
and print parameter counts of the paper configs."""
import torch, torchvision
from vit import ViT, CONFIGS, load_torchvision, interpolate_pos

torch.manual_seed(0)
tv = torchvision.models.vit_b_16(weights=None).eval()
m = load_torchvision(ViT(), tv).eval()
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    a, b = tv(x), m(x)
print("max |ours - torchvision| =", (a - b).abs().max().item())
for name, (p, d, L, h) in CONFIGS.items():
    n = sum(q.numel() for q in ViT(224, p, 3, d, L, h).parameters())
    print(f"{name:10s} {n/1e6:7.1f}M")
pos = interpolate_pos(m.pos.data, 24)   # 224/16=14 -> 384/16=24
print("pos 14x14 -> 24x24:", tuple(m.pos.shape), "->", tuple(pos.shape))
