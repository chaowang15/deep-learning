"""Parameter counts of the released CLIP configs, plus a loss sanity check."""
import math, torch
from clip import build, clip_loss, CONFIGS

for name in CONFIGS:
    m = build(name)
    n = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
    print(f"{name:13s} total {n(m):6.1f}M  image {n(m.image):6.1f}M  text {n(m.text):5.1f}M")
# with random features the loss should be about log(B)
torch.manual_seed(0)
B = 256
print("random-logit loss", round(clip_loss(torch.randn(B, B) * 0.01).item(), 3), "log B =", round(math.log(B), 3))
m = build("ViT-B/32").eval()
with torch.no_grad():
    ids = torch.randint(1, 49000, (4, 77)); ids[:, 10] = 49407     # put EOT at position 10
    print("logits", tuple(m(torch.randn(4, 3, 224, 224), ids).shape), "scale", round(m.logit_scale.exp().item(), 2))
