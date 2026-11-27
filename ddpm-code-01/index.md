---
date: 2026-11-26 16:30:00 -0800
keywords: >-
  DDPM from scratch, PyTorch, dataset normalization, linear beta schedule, alpha_bar, q_sample, tensor shapes.
summary: >-
  DDPM 从零实现系列第一篇：对照自己写的代码，讲清楚数据如何归一化到 [-1, 1]、线性 beta schedule 算出哪些量、q_sample 如何用一个闭式公式把 x_0 一步加噪到 x_t。每个模块给出输入输出和 tensor shape，附运行结果图、自检测试和面试要点。
layout: note
title: "DDPM 代码解读（一）：数据、噪声调度与前向加噪"
description: "DDPM from scratch, part 1: dataset normalization, the linear beta schedule, and the forward noising process q_sample, with tensor shapes and results."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig svg .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;fill:#475467}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
</style>

# DDPM 代码解读（一）：数据、噪声调度与前向加噪

这是"从零实现 DDPM"系列的第一篇。代码是我自己一步步手写的学习代码，放在本博客仓库的 [`code/diffusion/`](https://github.com/chaowang15/deep-learning/tree/main/code/diffusion)。公式推导见之前的笔记 [Diffusion Models 扩散模型学习](/deep-learning/diffusion-study/)，这里只讲**代码怎么对应公式**，以及每一步的输入输出和 tensor shape。

## 0. 整体安排：一个阶段一个 notebook

整个 DDPM pipeline 故意拆成几个阶段。每个阶段有一个 notebook 作为学习入口，实现放在 `src/`，`tests/` 做快速自检。这样每个阶段都能单独讲清楚，也方便面试前复习。

| 阶段 | Notebook | 实现 | 本文 |
|---|---|---|---|
| 00 数据 | `00_check_dataset.ipynb` | `datasets.py` | ✅ |
| 01 前向过程 | `01_forward_process.ipynb` | `schedule.py`, `forward_process.py` | ✅ |
| 02 噪声预测网络 | `02_noise_prediction.ipynb` | `time_embedding.py`, `unet.py` | 下一篇 |
| 03 DDPM 训练 | `03_ddpm_training.ipynb` | `diffusion.py`, `train.py` | |
| 04 采样 | `04_sampling.ipynb` | `sample.py` | |

本文覆盖阶段 00 和 01：**还不涉及任何神经网络**。前向加噪是一个固定的、没有参数的过程，但它决定了后面训练时"模型的输入是什么、要预测的目标是什么"。

## 1. 原理速览

DDPM 的前向过程每一步往图像里加一点高斯噪声：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big)
$$

记 $$\alpha_t = 1-\beta_t$$，$$\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$$。把多步高斯叠在一起，可以得到一个**闭式解**，从 $$x_0$$ 一步跳到任意 $$x_t$$：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\qquad \epsilon \sim \mathcal{N}(0, I)
$$

这一行公式就是本文代码的核心。它有两个直接后果：

- **训练时不用一步步加噪**：随机抽一个 $$t$$，直接算出 $$x_t$$，非常快。
- **训练目标就是 $$\epsilon$$**：模型看到 $$(x_t, t)$$，去预测当初加进去的那份噪声 $$\epsilon$$。所以 `q_sample` 要把 `noise` 一起返回。

两个系数的平方和恒等于 1：$$(\sqrt{\bar\alpha_t})^2 + (\sqrt{1-\bar\alpha_t})^2 = 1$$。如果 $$x_0$$ 的方差接近 1，那么 $$x_t$$ 的方差在每个时刻都保持在 1 附近，不会爆掉或塌缩。这也是数据要先归一化到 $$[-1, 1]$$ 的原因之一。

## 2. 模块关系图

<figure class="ddpm-fig">
<svg viewBox="0 0 760 330" role="img" aria-label="DDPM 阶段 00-01 模块关系图">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker>
  </defs>
  <!-- datasets -->
  <rect x="20" y="30" width="190" height="78" rx="10" fill="#eef4ff" stroke="#2a6fdb"/>
  <text x="115" y="56" text-anchor="middle">datasets.py</text>
  <text x="115" y="76" text-anchor="middle" class="sm">MNIST / CIFAR-10 等</text>
  <text x="115" y="95" text-anchor="middle" class="sm">像素 [0,1] → [-1,1]</text>
  <!-- schedule -->
  <rect x="20" y="200" width="190" height="78" rx="10" fill="#eef4ff" stroke="#2a6fdb"/>
  <text x="115" y="226" text-anchor="middle">schedule.py</text>
  <text x="115" y="246" text-anchor="middle" class="sm">make_beta_schedule(T)</text>
  <text x="115" y="265" text-anchor="middle" class="sm">β, α, ᾱ, √ᾱ, √(1−ᾱ) …</text>
  <!-- t sampler -->
  <rect x="250" y="125" width="130" height="50" rx="10" fill="#f8f9fa" stroke="#98a2b3"/>
  <text x="315" y="148" text-anchor="middle">随机 timestep t</text>
  <text x="315" y="166" text-anchor="middle" class="sm">randint(0, T)</text>
  <!-- q_sample -->
  <rect x="420" y="105" width="170" height="90" rx="10" fill="#fff4ec" stroke="#e0662b"/>
  <text x="505" y="133" text-anchor="middle">forward_process.py</text>
  <text x="505" y="153" text-anchor="middle" class="mono">q_sample(x0, t, …)</text>
  <text x="505" y="176" text-anchor="middle" class="sm">x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε</text>
  <!-- outputs -->
  <rect x="630" y="70" width="110" height="46" rx="10" fill="#fff" stroke="#475467"/>
  <text x="685" y="98" text-anchor="middle">x_t</text>
  <rect x="630" y="185" width="110" height="46" rx="10" fill="#fff" stroke="#475467"/>
  <text x="685" y="213" text-anchor="middle">ε（训练目标）</text>
  <!-- edges -->
  <path d="M210,69 C330,69 330,120 418,135" fill="none" stroke="#475467" marker-end="url(#ar)"/>
  <text x="300" y="62" class="mono">x0 [B,C,H,W]</text>
  <path d="M210,239 C330,239 340,190 418,170" fill="none" stroke="#475467" marker-end="url(#ar)"/>
  <text x="228" y="296" class="mono">√ᾱ, √(1−ᾱ) [T]</text>
  <path d="M380,150 L418,150" fill="none" stroke="#475467" marker-end="url(#ar)"/>
  <text x="399" y="195" text-anchor="middle" class="mono">t [B]</text>
  <path d="M590,135 L628,98" fill="none" stroke="#475467" marker-end="url(#ar)"/>
  <path d="M590,170 L628,205" fill="none" stroke="#475467" marker-end="url(#ar)"/>
  <text x="685" y="134" text-anchor="middle" class="mono">[B,C,H,W]</text>
  <text x="685" y="250" text-anchor="middle" class="mono">[B,C,H,W]</text>
  <!-- future -->
  <text x="685" y="300" text-anchor="middle" class="sm">→ 阶段 02：U-Net(x_t, t) 预测 ε</text>
</svg>
<figcaption>阶段 00–01 的数据流。B 为 batch，C/H/W 为通道和尺寸，T 为总步数（本项目默认 300）。</figcaption>
</figure>

## 3. 逐块解读

### 3.1 数据：`datasets.py`

源码：[`src/datasets.py`](https://github.com/chaowang15/deep-learning/blob/main/code/diffusion/src/datasets.py)

| | 内容 |
|---|---|
| 输入 | `DatasetConfig(name, data_dir, image_size, channels, batch_size, ...)` |
| 输出 | `DataLoader`，每个 batch 是 `images [B, C, H, W]`、`labels [B]` |
| 取值范围 | `images` 在 $$[-1, 1]$$ |
| shape | MNIST / FashionMNIST：`[B, 1, 28, 28]`；CIFAR-10：`[B, 3, 32, 32]` |

核心只有归一化这一步：

```python
def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),                     # uint8 → float，[0, 1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0, 1] → [-1, 1]
    ])

def denormalize(images):
    return (images.clamp(-1.0, 1.0) + 1.0) / 2.0  # 画图前再变回 [0, 1]
```

为什么是 $$[-1, 1]$$：噪声 $$\epsilon$$ 是均值 0、方差 1 的高斯，数据也以 0 为中心、尺度相近，两者按 $$\sqrt{\bar\alpha_t}$$ 和 $$\sqrt{1-\bar\alpha_t}$$ 混合时才"势均力敌"。采样阶段最后也会把输出 clamp 到 $$[-1, 1]$$ 再变回图片。

注意灰度图也**保留通道维** `C=1`。后面所有代码都按 `[B, C, H, W]` 写，同一套代码对灰度和 RGB 都能用。

### 3.2 噪声调度：`schedule.py`

源码：[`src/schedule.py`](https://github.com/chaowang15/deep-learning/blob/main/code/diffusion/src/schedule.py)

输入是总步数 `T` 和 $$\beta$$ 的起止值，输出一个字典，**每一项都是 shape `[T]` 的一维 tensor**，下标就是 timestep。

```python
betas = torch.linspace(beta_start, beta_end, T)       # β_t，线性从 1e-4 到 0.02
alphas = 1.0 - betas                                   # α_t
alpha_bars = torch.cumprod(alphas, dim=0)              # ᾱ_t = α_0·α_1·…·α_t
sqrt_alpha_bars = torch.sqrt(alpha_bars)               # x0 的系数
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)  # 噪声的系数

# 下面两项是给阶段 04（反向采样）准备的
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])   # ᾱ_{t-1}，约定 ᾱ_{-1}=1
posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)  # β̃_t
```

| 键 | 含义 | 用在哪 | t=0 | t=299 |
|---|---|---|---|---|
| `betas` | 每步加噪强度 $$\beta_t$$ | 采样 | 0.0001 | 0.02 |
| `alpha_bars` | 累计保留的信号比例 $$\bar\alpha_t$$ | 推导 | 0.9999 | 0.0481 |
| `sqrt_alpha_bars` | $$x_0$$ 系数 | `q_sample` | 0.99995 | 0.219 |
| `sqrt_one_minus_alpha_bars` | 噪声系数 | `q_sample`、采样 | 0.010 | 0.976 |
| `sqrt_recip_alphas` | $$1/\sqrt{\alpha_t}$$ | 采样均值 | 1.00005 | 1.010 |
| `posterior_variance` | $$\tilde\beta_t$$ | 采样方差 | 0 | 0.020 |

两个容易写错的细节：

- `torch.cumprod` 是累乘，所以 $$\bar\alpha_t$$ 单调下降。代码里 timestep 从 0 开始编号，和论文从 1 开始差一位，只是记号不同。
- `alpha_bars_prev` 在最前面补 1：$$t=0$$ 时"前一步"就是干净图像，信号保留比例为 1。这让 $$\tilde\beta_0 = 0$$，也就是采样的最后一步不再加噪声。

### 3.3 前向加噪：`forward_process.py`

源码：[`src/forward_process.py`](https://github.com/chaowang15/deep-learning/blob/main/code/diffusion/src/forward_process.py)

| 参数 | shape | 说明 |
|---|---|---|
| `x0` | `[B, C, H, W]` | 干净图像，$$[-1, 1]$$ |
| `t` | `[B]` | **每张图各自的** timestep |
| `sqrt_alpha_bars`、`sqrt_one_minus_alpha_bars` | `[T]` | 来自 schedule |
| `noise`（可选） | `[B, C, H, W]` | 不传就随机生成 |
| 返回 `xt, noise` | 都是 `[B, C, H, W]` | `noise` 就是训练目标 |

```python
def q_sample(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    # 用 t 做索引取出每张图自己的系数：[T] → [B]，再变成 [B, 1, 1, 1]
    sqrt_alpha_bar_t = sqrt_alpha_bars[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

    # [B,1,1,1] 与 [B,C,H,W] 广播相乘：同一张图的所有像素用同一个系数
    xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    return xt, noise
```

关键在 `.view(-1, 1, 1, 1)`。`sqrt_alpha_bars[t]` 是 `[B]`，直接和 `[B, C, H, W]` 相乘会按**最后一维**对齐而报错（或者更糟，碰巧 shape 对上而算错）。变成 `[B, 1, 1, 1]` 后，广播规则让第 i 张图的所有像素都乘上第 i 个系数。

完整代码里还把 `t`、系数搬到和 `x0` 相同的 device 和 dtype 上，这样 schedule 在 CPU 上建好，图片在 GPU / MPS 上也能直接用。

### 3.4 可视化里的 batch 小技巧：`repeat_interleave` vs `repeat`

[`01_forward_process.ipynb`](https://github.com/chaowang15/deep-learning/blob/main/code/diffusion/notebooks/01_forward_process.ipynb) 想把 3 张图各自在 6 个 timestep 下的加噪结果排成一个网格，一次 `q_sample` 算完：

```python
selected_steps = torch.tensor([0, 25, 50, 100, 200, T - 1])     # [6]
x0_batch = x0.repeat_interleave(len(selected_steps), dim=0)      # [3,1,28,28] → [18,1,28,28]
t_batch = selected_steps.repeat(num_images)                      # [6] → [18]
xt, _ = q_sample(x0_batch, t_batch, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)
```

- `repeat_interleave` 是**逐元素**复制：`[img0, img1, img2]` → `[img0×6, img1×6, img2×6]`。
- `repeat` 是**整体**复制：`[t0…t5]` → `[t0…t5, t0…t5, t0…t5]`。

两者拼起来，第 k 个元素正好是"第 k//6 张图在第 k%6 个 timestep"。`make_grid(..., nrow=6)` 再按行排开，就是一张图一行。

## 4. 运行与结果

在 Mac 上（Windows 见代码目录的 README）：

```bash
cd code/diffusion
python -m pip install -r requirements.txt
python scripts/check_env.py      # Apple Silicon 上应显示 device: mps
python -m pytest -q              # 6 passed
jupyter notebook                 # 打开 notebooks/00、01
```

**噪声调度曲线**（T=300，线性 $$\beta$$）：

![schedule](/deep-learning/assets/ddpm-code-01/schedule.png)

| t | 0 | 25 | 50 | 100 | 200 | 299 |
|---|---|---|---|---|---|---|
| $$\sqrt{\bar\alpha_t}$$（图像） | 1.000 | 0.988 | 0.956 | 0.841 | 0.506 | 0.219 |
| $$\sqrt{1-\bar\alpha_t}$$（噪声） | 0.010 | 0.155 | 0.294 | 0.542 | 0.863 | 0.976 |

**FashionMNIST 前向加噪**（每行一张图，从左到右 t 增大）：

![forward noising on FashionMNIST](/deep-learning/assets/ddpm-code-01/forward_fashion_mnist.png)

几个观察：

1. 前 50 步图像几乎不变，噪声主要在中后段才快速占上风，和右边曲线的形状一致。
2. **T=300 时最后一步并不是纯噪声**：$$\sqrt{\bar\alpha_{299}} \approx 0.22$$，仔细看 t=299 仍能隐约看到轮廓。DDPM 原论文用 T=1000、同样的 $$\beta$$ 范围，最后 $$\sqrt{\bar\alpha_{999}} \approx 0.006$$，才接近纯噪声。这里选 300 是为了在小数据集上训练、采样更快；代价是采样起点 $$x_T \sim \mathcal{N}(0, I)$$ 和真实的 $$x_T$$ 分布有偏差。之后在阶段 04 可以对比 T=300 和 T=1000 的生成效果，或者换成 cosine schedule。
3. 同一套代码对 `[B,1,28,28]` 和 `[B,3,32,32]` 都成立，notebook 里也跑了 CIFAR-10。

## 5. 自检测试

[`tests/`](https://github.com/chaowang15/deep-learning/tree/main/code/diffusion/tests) 里放了几条不需要下载数据、一秒内跑完的检查，改完 `src/` 后跑 `python -m pytest -q`：

- schedule 每一项 shape 都是 `[T]`，$$\bar\alpha_t$$ 单调下降；
- $$(\sqrt{\bar\alpha_t})^2 + (\sqrt{1-\bar\alpha_t})^2 = 1$$；
- $$\tilde\beta_0 = 0$$ 且 $$\tilde\beta_t \le \beta_t$$；
- `q_sample` 对灰度和 RGB 输出 shape 正确，固定 `noise` 时与公式逐元素一致；
- 最后一步大部分是噪声（std 接近 1）。

## 6. 面试要点

- **为什么可以一步从 $$x_0$$ 得到 $$x_t$$？** 每步都是线性高斯，高斯的线性组合仍是高斯，均值系数累乘成 $$\sqrt{\bar\alpha_t}$$，方差合并成 $$1-\bar\alpha_t$$。
- **前向过程有可学习参数吗？** 没有，$$\beta_t$$ 是预先定好的。学习只发生在反向过程（预测噪声的网络）。
- **为什么数据要归一化到 $$[-1, 1]$$？** 和标准高斯噪声的尺度匹配，保证 $$x_t$$ 方差稳定在 1 左右；也让 $$x_T$$ 接近 $$\mathcal{N}(0, I)$$，采样才有合理起点。
- **`t` 为什么是 `[B]` 而不是一个标量？** 训练时 batch 里每张图独立随机抽 timestep，让一次前向就覆盖多个噪声水平，梯度方差更小。
- **T 取多大有什么影响？** T 太小，最后一步残留信号多，采样起点和训练分布不一致；T 大则采样步数多、慢（DDIM 等方法就是为了解决这个）。

## 下一步

阶段 02：写 sinusoidal timestep embedding 和一个小 U-Net，输入 `(x_t, t)`，输出和 `x_t` 同 shape 的噪声预测，先在一个小 batch 上确认 MSE loss 能下降。
