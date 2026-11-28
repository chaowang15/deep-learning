"""Figures for the ViT + CLIP post (run after exp_data_size.py and exp_clip.py)."""
import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "vae"))
from common import load_fmnist  # noqa: E402
from exp_data_size import tiny_vit  # noqa: E402
import exp_clip as E  # noqa: E402
plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Zen Hei"]
OUT = ROOT / "assets" / "vit-clip"
C_CNN, C_VIT = "#d9822b", "#2b6cb0"

# 1) data size
r = json.loads((OUT / "data_size.json").read_text())
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for name, c in (("CNN", C_CNN), ("ViT", C_VIT)):
    rs = [q for q in r["runs"] if q["model"] == name]
    ns, acc = [q["n"] for q in rs], [100 * q["test_acc"] for q in rs]
    ax.plot(ns, acc, "o-", color=c, lw=2, label=f"{name}（{rs[0]['params']/1e3:.0f}K 参数）")
    for n_, a in zip(ns, acc):
        ax.annotate(f"{a:.1f}", (n_, a), textcoords="offset points", xytext=(0, 7 if name == "CNN" else -14),
                    ha="center", fontsize=9, color=c)
ax.set_ylim(min(q["test_acc"] for q in r["runs"]) * 100 - 2, max(q["test_acc"] for q in r["runs"]) * 100 + 1.5)
ax.set_xscale("log"); ax.set_xticks([1000, 6000, 60000]); ax.set_xticklabels(["1k", "6k", "60k"])
ax.set_xlabel("训练图像数（对数轴）"); ax.set_ylabel("测试准确率 (%)")
ax.set_title(f"数据量 vs 准确率（每个模型都训练 {r['steps']} 步）", fontsize=11)
ax.grid(alpha=0.3); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "data_size.png", dpi=150); plt.close(fig)

# 2) learned position-embedding similarity (7x7 grid of 7x7 maps)
m = tiny_vit(); m.load_state_dict(torch.load("/tmp/vit_60000.pt"))
pos = F.normalize(m.pos.data[0, 1:], dim=-1)          # (49, D), drop [CLS]
sim = (pos @ pos.t()).reshape(49, 7, 7)
fig, axes = plt.subplots(7, 7, figsize=(5.6, 5.6))
for i, a in enumerate(axes.flat):
    a.imshow(sim[i], cmap="viridis", vmin=-1, vmax=1); a.set_xticks([]); a.set_yticks([])
fig.suptitle("每格：第 (行, 列) 个 patch 的位置编码与所有位置的余弦相似度", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "pos_embed_sim.png", dpi=150); plt.close(fig)
S = pos @ pos.t(); rr, cc = torch.arange(49) // 7, torch.arange(49) % 7
dr, dc = (rr[:, None] - rr[None]).abs(), (cc[:, None] - cc[None]).abs()
stats = {"neighbors": S[dr + dc == 1].mean().item(),
         "mirror_same_row": S[(dr == 0) & (cc[:, None] == 6 - cc[None]) & (cc[:, None] != 3)].mean().item(),
         "same_row": S[(dr == 0) & (dc > 0)].mean().item(), "same_col": S[(dc == 0) & (dr > 0)].mean().item(),
         "other": S[(dr > 0) & (dc > 0)].mean().item()}
(OUT / "pos_embed_stats.json").write_text(json.dumps(stats, indent=1)); print(stats)

# 3) CLIP training curves
c = json.loads((OUT / "clip_results.json").read_text())
st = [q["step"] for q in c["log"]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
a1.plot(st, [q["loss"] for q in c["log"]], color=C_VIT, lw=1.5, label="对比损失")
B, k = c["batch"], c["avg_same_class_in_batch"]
a1.axhline(math.log(B), ls="--", color="gray", lw=1); a1.text(st[-1], math.log(B) + 0.08, f"log B = {math.log(B):.2f}（随机）", ha="right", fontsize=8.5)
# floor if same-class items are indistinguishable except for the dark/light word (see the post, sec 4.3):
# image->text: 0.75k captions fit equally well; text->image: k images (no adjective) or k/2 (with one)
floor = 0.5 * math.log(0.75 * k) + 0.25 * (math.log(k) + math.log(k / 2))
a1.axhline(floor, ls=":", color="#c53030", lw=1.2)
a1.text(st[-1], floor - 0.32, f"≈ {floor:.2f}：同类图文无法区分时的损失下限", ha="right", fontsize=8.5, color="#c53030")
a1.set_xlabel("步数"); a1.set_ylabel("loss"); a1.grid(alpha=0.3); a1.set_ylim(0, math.log(B) + 0.6)
a2.plot(st, [q["scale"] for q in c["log"]], color=C_CNN, lw=1.5)
a2.set_xlabel("步数"); a2.set_ylabel("1/τ = exp(logit_scale)"); a2.set_title("可学习温度", fontsize=10); a2.grid(alpha=0.3)
a1.set_title(f"batch = {B}，每张图平均有 k = {k:.1f} 张同类图（含自己）", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "clip_curves.png", dpi=150); plt.close(fig)

# 4) text -> image retrieval on the test set
mc = E.tiny_clip(); mc.load_state_dict(torch.load("/tmp/clip_tiny.pt")); mc.eval()
xt, yt = load_fmnist("test")
u = E.encode_images(mc, xt * 2 - 1)
queries = ["a photo of a dark bag", "a photo of a light bag", "a photo of a sneaker",
           "a photo of a dark dress", "a photo of a light dress", "a photo of an ankle boot"]
w = E.encode_texts(mc, queries)
top = (w @ u.t()).topk(8, dim=1).indices
fig, axes = plt.subplots(len(queries), 8, figsize=(8, 1.15 * len(queries) + 0.3))
for qi, q in enumerate(queries):
    for k_ in range(8):
        a = axes[qi, k_]; a.imshow(xt[top[qi, k_], 0], cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
    axes[qi, 0].set_ylabel(q.replace("a photo of ", ""), rotation=0, ha="right", va="center", fontsize=9)
fig.suptitle("文本检索图像：每行一个文本查询，前 8 张最相似的测试图", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "clip_retrieval.png", dpi=150); plt.close(fig)

# 5) image x class-prompt similarity for one test image per class
idx = torch.stack([(yt == cl).nonzero()[0, 0] for cl in range(10)])
W = E.encode_texts(mc, [E.an(f"a photo of a {n}") for n in E.CLASSES])
with torch.no_grad():
    S = (u[idx] @ W.t()) * mc.logit_scale.exp().clamp(max=100)
P = S.softmax(1)
fig, a1 = plt.subplots(figsize=(6.6, 5.4))
a1.imshow(P, cmap="Blues", vmin=0, vmax=1, extent=(-0.5, 9.5, 9.5, -0.5))
a1.imshow(torch.cat([xt[i, 0] for i in idx], 0), cmap="gray", extent=(-1.7, -0.7, 9.5, -0.5))
a1.set_xlim(-1.75, 9.5)
for i in range(10):
    for j in range(10):
        a1.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if P[i, j] > 0.5 else "black")
a1.set_xticks(range(10)); a1.set_xticklabels(E.CLASSES, rotation=40, ha="right", fontsize=8.5); a1.set_yticks([])
a1.set_title("softmax(cos / τ)：每行一张测试图，每列一个提示 “a photo of a …”", fontsize=9.5)
a1.spines[:].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "clip_zeroshot_matrix.png", dpi=150); plt.close(fig)
print("figs done")
