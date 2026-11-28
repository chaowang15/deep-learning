"""Figures for the 2-D experiment: toy_paths.png, toy_steps.png (needs /tmp/flow_toy.pt from toy_exp.py)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from toy_exp import VelocityMLP, sample_data, sample_euler, OUT

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.set_num_threads(1)
ck = torch.load("/tmp/flow_toy.pt")
rf1, rf2 = VelocityMLP(), VelocityMLP()
rf1.load_state_dict(ck["rf1"]); rf2.load_state_dict(ck["rf2"])

torch.manual_seed(3)
eps = torch.randn(300, 2)
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6))
x0r = sample_data(300)
ax = axs[0]
for i in range(120):
    ax.plot([eps[i, 0], x0r[i, 0]], [eps[i, 1], x0r[i, 1]], color="#98a2b3", lw=0.6, alpha=0.7)
ax.scatter(eps[:120, 0], eps[:120, 1], s=6, c="#2a6fdb", zorder=3, label="噪声 ε（t=1）")
ax.scatter(x0r[:120, 0], x0r[:120, 1], s=6, c="#e0662b", zorder=3, label="数据 x₀（t=0）")
ax.set_title("(a) 训练用的条件路径：随机配对后连直线\n直线之间大量交叉")
for ax, m, title in [(axs[1], rf1, "(b) 1-Rectified Flow 的 ODE 轨迹\n交叉处被平均，轨迹弯曲"),
                     (axs[2], rf2, "(c) Reflow 之后（2-Rectified Flow）\n轨迹几乎是直线，不再交叉")]:
    x0, path = sample_euler(m, eps, 100, return_path=True)
    for i in range(120):
        ax.plot(path[:, i, 0], path[:, i, 1], color="#98a2b3", lw=0.6, alpha=0.8)
    ax.scatter(eps[:120, 0], eps[:120, 1], s=6, c="#2a6fdb", zorder=3)
    ax.scatter(x0[:120, 0], x0[:120, 1], s=6, c="#e0662b", zorder=3)
    ax.set_title(title)
for ax in axs:
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
axs[0].legend(loc="lower left", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "toy_paths.png", dpi=130, bbox_inches="tight"); plt.close()

torch.manual_seed(1)
eps = torch.randn(3000, 2)
steps = [1, 2, 4, 100]
fig, axs = plt.subplots(2, 4, figsize=(12, 6.3))
for r, (m, name) in enumerate([(rf1, "1-RF"), (rf2, "2-RF（Reflow）")]):
    for c, n in enumerate(steps):
        x = sample_euler(m, eps, n)
        ax = axs[r, c]
        ax.scatter(x[:, 0], x[:, 1], s=1.5, c="#e0662b", alpha=0.5)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(f"N = {n} 步")
        if c == 0: ax.set_ylabel(name, fontsize=12)
plt.tight_layout(); plt.savefig(OUT / "toy_steps.png", dpi=130); plt.close()
print("ok")
