---
date: 2026-11-26 16:30:00 -0800
keywords: >-
  DDPM from scratch, PyTorch, dataset normalization, linear beta schedule, alpha_bar, q_sample, tensor shapes.
summary: >-
  DDPM 从零实现系列第一篇：对照自己写的代码，讲清楚数据如何归一化到 [-1, 1]、线性 beta schedule 算出哪些量、q_sample 如何用一个闭式公式把 x_0 一步加噪到 x_t。每个模块逐项对照公式和代码、标出 tensor shape，并总结广播、下标、device、pickle 等实现上的易错点，附运行结果、自检测试和面试要点。
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

每个模块按同样的顺序讲：**公式 → 公式和代码逐项对应 → 核心代码 → 易错点**。易错点里标了"代码注释"的，是我写代码时在注释里记下的坑；其余是这次整理时补充的。

### 3.1 数据：`datasets.py`

源码：[`src/datasets.py`](https://github.com/chaowang15/deep-learning/blob/main/code/diffusion/src/datasets.py)

**公式。** 把像素从 $$[0, 1]$$ 线性映射到 $$[-1, 1]$$，显示时再映射回去：

$$
x = 2u - 1,\quad u \in [0, 1] \qquad\Longleftrightarrow\qquad u = \frac{x + 1}{2}
$$

| 公式 | 代码 | shape |
|---|---|---|
| 原始像素 $$\in \{0,\dots,255\}$$ → $$u \in [0,1]$$ | `transforms.ToTensor()`（内部除以 255，并把 HWC 转成 CHW） | `[C, H, W]` |
| $$x = 2u - 1$$ | `scale_to_minus_one_one(x)`：`x * 2.0 - 1.0` | `[C, H, W]` |
| 一个 batch 的 $$x_0$$ | `DataLoader` 输出的 `images` | `[B, C, H, W]` |
| $$u = (x+1)/2$$ | `denormalize(images)`：`(images.clamp(-1, 1) + 1) / 2` | 不变 |

```python
def scale_to_minus_one_one(x):
    return x * 2.0 - 1.0          # [0, 1] → [-1, 1]

def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),      # uint8 HWC → float CHW，[0, 1]
        scale_to_minus_one_one,
    ])

def denormalize(images):
    return (images.clamp(-1.0, 1.0) + 1.0) / 2.0  # 画图前再变回 [0, 1]
```

为什么是 $$[-1, 1]$$：噪声 $$\epsilon$$ 均值 0、方差 1，数据也以 0 为中心、尺度相近，两者按 $$\sqrt{\bar\alpha_t}$$ 和 $$\sqrt{1-\bar\alpha_t}$$ 混合时才"势均力敌"。

> **易错点**
>
> 1. **灰度图也要保留通道维 `C=1`**（代码注释）。MNIST 一个 batch 是 `[B, 1, 28, 28]` 而不是 `[B, 28, 28]`。后面所有代码都按 `[B, C, H, W]` 写，灰度和 RGB 才能共用一套代码；U-Net 的第一层卷积也需要这个 `C`。
> 2. **不要在 transform 里写 `lambda`。** 最初的写法是 `transforms.Lambda(lambda x: x * 2.0 - 1.0)`。在 macOS（和 Windows）上，`DataLoader` 的 `num_workers > 0` 会用 spawn 方式启动子进程，需要 pickle 整个 transform，而 lambda 无法 pickle，会直接报 `Can't get local object 'build_transform.<locals>.<lambda>'`。notebook 里用的是 `num_workers=0` 所以没暴露，但 config 里的 `num_workers: 2` 在 Mac 上训练时就会崩。这次整理已改成模块级函数，并加了测试。
> 3. **`denormalize` 里先 `clamp`。** 现在输入都在 $$[-1, 1]$$ 内，看起来多余；但到了采样阶段，模型输出可能略微越界，不 clamp 的话 `imshow` 会报 warning 或者颜色失真。
> 4. **相对路径按项目目录解析，而不是当前工作目录。** `data_dir: ../data` 是相对 `code/diffusion/` 的；notebook 的工作目录是 `notebooks/`，如果直接用 `Path("../data")` 就会指到错误的位置。代码里用 `PROJECT_DIR / root` 解决了这个问题。

### 3.2 噪声调度：`schedule.py`

源码：[`src/schedule.py`](https://github.com/chaowang15/deep-learning/blob/main/code/diffusion/src/schedule.py)

**公式。** 线性 schedule 和它派生出的量：

$$
\beta_t = \beta_{\text{start}} + \frac{t}{T-1}(\beta_{\text{end}} - \beta_{\text{start}}),\qquad
\alpha_t = 1 - \beta_t,\qquad
\bar\alpha_t = \prod_{s=0}^{t} \alpha_s
$$

为阶段 04（反向采样）提前准备的两个量，来自后验 $$q(x_{t-1} \mid x_t, x_0)$$：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\Big),\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
$$

| 公式 | 代码 | shape |
|---|---|---|
| $$\beta_t$$ | `betas = torch.linspace(beta_start, beta_end, T)` | `[T]` |
| $$\alpha_t = 1 - \beta_t$$ | `alphas = 1.0 - betas` | `[T]` |
| $$\bar\alpha_t = \prod_{s \le t} \alpha_s$$ | `alpha_bars = torch.cumprod(alphas, dim=0)` | `[T]` |
| $$\sqrt{\bar\alpha_t}$$（$$x_0$$ 的系数） | `sqrt_alpha_bars` | `[T]` |
| $$\sqrt{1-\bar\alpha_t}$$（噪声的系数） | `sqrt_one_minus_alpha_bars` | `[T]` |
| $$1/\sqrt{\alpha_t}$$（$$\mu_\theta$$ 最外面的系数） | `sqrt_recip_alphas = torch.sqrt(1.0 / alphas)` | `[T]` |
| $$\bar\alpha_{t-1}$$，约定 $$\bar\alpha_{-1} = 1$$ | `alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])` | `[T]` |
| $$\tilde\beta_t$$ | `posterior_variance = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)` | `[T]` |

代码里把采样公式直接写成了注释，方便对照（代码注释）：

```python
# p_sample:
# x_{t-1} = mu_theta(x_t, t) + sigma_t * noise
# mu_theta = sqrt_recip_alphas * (x_t - pred_noise * beta_t / sqrt_one_minus_alpha_bars)
# sigma_t = sqrt(beta_tilde_t),  beta_tilde_t = posterior_variance
```

每一项都是 shape `[T]` 的一维 tensor，下标就是 timestep。数值：

| 键 | t=0 | t=299 |
|---|---|---|
| `betas` | 0.0001 | 0.02 |
| `alpha_bars` | 0.9999 | 0.0481 |
| `sqrt_alpha_bars` | 0.99995 | 0.219 |
| `sqrt_one_minus_alpha_bars` | 0.010 | 0.976 |
| `sqrt_recip_alphas` | 1.00005 | 1.010 |
| `posterior_variance` | 0 | 0.020 |

> **易错点**
>
> 1. **下标从 0 开始。** 论文里 $$t = 1, \dots, T$$，代码里 $$t = 0, \dots, T-1$$。所以"最后一步"是 `T - 1`（本项目是 299），采样循环要写 `reversed(range(T))`，不是 `range(T, 0, -1)`。
> 2. **`cumprod` 包含当前这一项。** `alpha_bars[i]` $$= \alpha_0 \alpha_1 \cdots \alpha_i$$。代码顶部的注释写的是 `alpha_bar_i = alpha_0 * ... * alpha_{i-1}`，少乘了一项，和 `cumprod` 实际算的不一致，按注释推导会差一步。（注释后续可以顺手改掉。）
> 3. **`alpha_bars_prev` 要在最前面补 1**（代码注释）。$$t=0$$ 时"前一刻"就是干净图像，信号保留比例为 1。`torch.cat` 只接受 tensor，所以写 `torch.ones(1, device=device)` 而不是 Python 的 `1.0`，并且要和 `alpha_bars` 在同一个 device 上，否则 `cat` 会报错。
> 4. **$$\tilde\beta_0 = 0$$ 是故意的。** 因为 $$\bar\alpha_{-1} = 1$$，分子为 0：采样的最后一步只取均值，不再加噪声。副作用是，如果以后要用 $$\log \tilde\beta_t$$（比如学习方差的改进版 DDPM），`log(0) = -inf`，常见做法是用 $$\tilde\beta_1$$ 替换第 0 项再取 log。
> 5. **整张表只算一次。** schedule 不依赖数据，在训练开始前算好、作为常量查表；不要在每个 batch 里重新算。

### 3.3 前向加噪：`forward_process.py`

源码：[`src/forward_process.py`](https://github.com/chaowang15/deep-learning/blob/main/code/diffusion/src/forward_process.py)

**公式。** 闭式前向过程：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\qquad \epsilon \sim \mathcal{N}(0, I)
$$

| 公式 | 代码 | shape |
|---|---|---|
| $$x_0$$ | `x0` | `[B, C, H, W]` |
| $$t$$（每张图各自一个） | `t` | `[B]` |
| $$\epsilon \sim \mathcal{N}(0, I)$$ | `noise = torch.randn_like(x0)` | `[B, C, H, W]` |
| $$\sqrt{\bar\alpha_t}$$ | `sqrt_alpha_bars[t].view(-1, 1, 1, 1)` | `[T]` → `[B]` → `[B, 1, 1, 1]` |
| $$\sqrt{1-\bar\alpha_t}$$ | `sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)` | `[B, 1, 1, 1]` |
| $$x_t = \dots$$ | `xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bars_t * noise` | `[B, C, H, W]` |
| 训练目标 $$\epsilon$$ | 返回值 `noise` | `[B, C, H, W]` |

```python
def q_sample(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    t = t.to(device=x0.device, dtype=torch.long)
    sqrt_alpha_bars = sqrt_alpha_bars.to(device=x0.device, dtype=x0.dtype)
    sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.to(device=x0.device, dtype=x0.dtype)

    # [B] => [B, 1, 1, 1]
    sqrt_alpha_bar_t = sqrt_alpha_bars[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bars_t = sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

    xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bars_t * noise
    return xt, noise
```

> **易错点**
>
> 1. **`[B]` 必须变成 `[B, 1, 1, 1]` 再乘**（代码注释）。PyTorch 广播从**最后一维**开始对齐：`[B]` 直接乘 `[B, C, H, W]` 时，`B` 会去和 `W` 对齐。大多数时候会报错；但如果碰巧 `B == W`（比如 batch 是 28、图是 28×28），它**不会报错而是悄悄算错**，每一列像素乘上了不同图的系数。`.view(-1, 1, 1, 1)` 让第 i 张图的所有像素都乘第 i 个系数。
> 2. **`t` 是 `[B]`，不是标量。** 训练时 batch 里每张图独立随机抽 timestep，所以必须"按图查表"：`sqrt_alpha_bars[t]` 是用一个整数 tensor 做索引，一次取出 B 个系数。
> 3. **索引用的 `t` 必须是 `long`，并且和被索引的表在同一个 device 上。** schedule 通常在 CPU 上建好，图片在 GPU / MPS 上；用 GPU 上的 `t` 去索引 CPU 上的表会报 device 不一致。代码里统一把 `t` 和两张表搬到 `x0.device`，并把系数转成 `x0.dtype`，这样以后用半精度（AMP）也不会因为 float32 × float16 而类型提升。
> 4. **一定要把 `noise` 返回。** 训练的 loss 是 $$\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$$，目标必须是**生成这个 $$x_t$$ 时用的同一份噪声**。如果在外面再随机一份当目标，目标和输入就没有关系了，loss 永远降不下去。
> 5. **`noise` 参数可以从外面传入。** 测试和可视化时传固定的 noise，就能把结果和公式逐元素对比（见第 5 节的测试）。

### 3.4 可视化里的 batch 小技巧：`repeat_interleave` vs `repeat`

[`01_forward_process.ipynb`](https://github.com/chaowang15/deep-learning/blob/main/code/diffusion/notebooks/01_forward_process.ipynb) 想把 3 张图各自在 6 个 timestep 下的加噪结果排成一个网格，一次 `q_sample` 算完。需要构造两个对齐的 batch：

$$
\text{第 } k \text{ 个元素} = \big(\,x_0^{(\lfloor k/6 \rfloor)},\ t_{(k \bmod 6)}\,\big),\qquad k = 0, \dots, 17
$$

```python
selected_steps = torch.tensor([0, 25, 50, 100, 200, T - 1])     # [6]
x0_batch = x0.repeat_interleave(len(selected_steps), dim=0)      # [3,1,28,28] → [18,1,28,28]
t_batch = selected_steps.repeat(num_images)                      # [6] → [18]
xt, _ = q_sample(x0_batch, t_batch, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)
```

> **易错点**（代码注释）
>
> `repeat_interleave` 是**逐元素**复制，`repeat` 是**整体**复制。拿一张 RGB 图 `img` 的 shape `[3, 2, 2]` 举例：
>
> - `img.repeat_interleave(2, dim=0)` → `[R, R, G, G, B, B]`
> - `img.repeat(2, 1, 1)` → `[R, G, B, R, G, B]`
>
> 两者输出 shape 完全相同（都是 `[6, 2, 2]`），都不增加新维度，只是 dim 0 上的顺序不同，所以**写反了也不会报错，只是图和 timestep 错位**。这里图片要"每张连续出现 6 次"，用 `repeat_interleave`；timestep 要"整段重复 3 次"，用 `repeat`。最后 `make_grid(..., nrow=6)` 按行排开，正好一张图一行。

## 4. 运行与结果

在 Mac 上（Windows 见代码目录的 README）：

```bash
cd code/diffusion
python -m pip install -r requirements.txt
python scripts/check_env.py      # Apple Silicon 上应显示 device: mps
python -m pytest -q              # 8 passed
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

- transform 可以被 pickle（Mac 上 `num_workers > 0` 不会崩），`denormalize` 能还原；
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
