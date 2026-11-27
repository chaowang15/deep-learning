---
date: 2026-11-26 16:30:00 -0800
keywords: >-
  DDPM from scratch, PyTorch, noise schedule, forward process, sinusoidal timestep embedding, U-Net, training, sampling, DDIM, classifier-free guidance, v-prediction, SNR.
summary: >-
  用 PyTorch 从零实现完整的 DDPM：数据归一化、噪声调度、前向加噪、时间步编码与 Tiny U-Net、训练、反向采样、DDIM 少步采样、类别条件与 classifier-free guidance，以及 ε / x0 / v 三种预测目标。每一部分先讲公式，再给出逐行对应公式的代码和 tensor shape，最后总结实现上的易错点，附 FashionMNIST 上的真实训练和生成结果。
layout: note
title: "DDPM 从零实现：从公式到代码"
description: "A complete DDPM implementation from scratch in PyTorch, formula by formula: schedule, forward process, U-Net, training, sampling, DDIM, CFG, and prediction targets."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig svg .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;fill:#475467}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
</style>

# DDPM 从零实现：从公式到代码

目标是：**只看这一页，就能从公式出发，自己把一个能训练、能生成图像的 DDPM 一行行写出来**。每一部分都按同一个顺序讲：

1. 它是什么、为什么需要它，以及公式；
2. 代码实现，变量名和公式符号一一对应；
3. 易错点：公式写对了，代码也可能出错的地方。

完整的理论推导（ELBO 等）见之前的笔记 [Diffusion Models 扩散模型学习](/deep-learning/diffusion-study/)，这里只推导写代码需要的部分。

## 0. 全文路线

DDPM 分成训练和采样两条线，共用同一张噪声调度表和同一个网络：

| | 起点 | 方向 | 网络做什么 | 本文章节 |
|---|---|---|---|---|
| 训练 | 真实图像 $$x_0$$ | $$x_0 \to x_t$$（一步闭式加噪） | 预测加进去的噪声 | 2 数据 → 3 调度 → 4 前向 → 5 网络 → 6 训练 |
| 采样 | 纯噪声 $$x_T$$ | $$x_T \to x_{T-1} \to \cdots \to x_0$$ | 每一步预测并去掉一部分噪声 | 7 DDPM 采样 → 8 DDIM → 9 条件生成与 CFG |

第 10 节讨论网络除了噪声还能预测什么（$$x_0$$、$$v$$），第 11 节是一组不需要训练就能跑的自检，第 12 节是要点回顾。所有结果图都来自本文代码在 FashionMNIST（28×28 灰度服装图）上的真实运行。

## 1. 符号约定

先把公式符号和代码变量名对齐，后面所有公式和代码都按这张表来。

| 公式符号 | 含义 | 代码变量 | shape |
|---|---|---|---|
| $$T$$ | 总步数（本文取 300） | `T` | 标量 |
| $$t \in \{1, \dots, T\}$$ | 时间步 | `t`（**0-based 下标**，等于 $$t-1$$） | `[B]` |
| $$x_0$$ | 干净图像 | `x0` | `[B, C, H, W]` |
| $$x_t$$ | 第 $$t$$ 步的带噪图像 | `xt` | `[B, C, H, W]` |
| $$\epsilon$$ | 标准高斯噪声 | `noise` | `[B, C, H, W]` |
| $$\beta_t$$ | 第 $$t$$ 步加噪强度 | `betas[t-1]` | 整张表 `[T]` |
| $$\alpha_t = 1-\beta_t$$ | 第 $$t$$ 步信号保留比例 | `alphas[t-1]` | `[T]` |
| $$\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$$ | 累计信号保留比例 | `alpha_bars[t-1]` | `[T]` |
| $$\bar\alpha_{t-1}$$，约定 $$\bar\alpha_0 = 1$$ | 上一步的累计比例 | `alpha_bars_prev[t-1]` | `[T]` |
| $$\tilde\beta_t$$ | 反向过程的方差（第 7 节） | `posterior_variance[t-1]` | `[T]` |
| $$\epsilon_\theta(x_t, t)$$ | 网络预测的噪声（第 5 节） | `pred_noise` | `[B, C, H, W]` |

其中 $$B$$ 是 batch 大小，$$C, H, W$$ 是通道数和图像尺寸（灰度 28×28 时为 `1, 28, 28`，RGB 32×32 时为 `3, 32, 32`）。

**下标约定是第一个坑**：论文里 $$t$$ 从 1 数到 $$T$$，$$x_0$$ 是干净图像；代码里所有表都是长度 $$T$$ 的数组，下标从 0 开始。所以**代码里的 `t` 永远比论文的 $$t$$ 小 1**：`alpha_bars[0]` 是 $$\bar\alpha_1$$，`alpha_bars[T-1]` 是 $$\bar\alpha_T$$。干净图像 $$x_0$$ 在表里没有对应项，它的"$$\bar\alpha_0 = 1$$"只出现在 `alpha_bars_prev` 里。

<figure class="ddpm-fig">
<svg viewBox="0 0 760 330" role="img" aria-label="数据归一化、噪声调度和前向加噪之间的数据流">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker>
  </defs>
  <rect x="20" y="30" width="190" height="78" rx="10" fill="#eef4ff" stroke="#2a6fdb"/>
  <text x="115" y="56" text-anchor="middle">数据归一化</text>
  <text x="115" y="76" text-anchor="middle" class="sm">像素 [0,1] → [-1,1]</text>
  <text x="115" y="95" text-anchor="middle" class="sm">灰度图也保留通道维</text>
  <rect x="20" y="200" width="190" height="78" rx="10" fill="#eef4ff" stroke="#2a6fdb"/>
  <text x="115" y="226" text-anchor="middle">噪声调度</text>
  <text x="115" y="246" text-anchor="middle" class="mono">make_beta_schedule(T)</text>
  <text x="115" y="265" text-anchor="middle" class="sm">每个时间步的系数表</text>
  <rect x="250" y="125" width="130" height="50" rx="10" fill="#f8f9fa" stroke="#98a2b3"/>
  <text x="315" y="148" text-anchor="middle">随机时间步 t</text>
  <text x="315" y="166" text-anchor="middle" class="sm">每张图各抽一个</text>
  <rect x="420" y="105" width="170" height="90" rx="10" fill="#fff4ec" stroke="#e0662b"/>
  <text x="505" y="133" text-anchor="middle">前向加噪</text>
  <text x="505" y="153" text-anchor="middle" class="mono">q_sample(x0, t, …)</text>
  <text x="505" y="176" text-anchor="middle" class="sm">按图下公式混合 x0 和噪声</text>
  <rect x="630" y="70" width="110" height="46" rx="10" fill="#fff" stroke="#475467"/>
  <text x="685" y="98" text-anchor="middle">带噪图像 xt</text>
  <rect x="630" y="185" width="110" height="46" rx="10" fill="#fff" stroke="#475467"/>
  <text x="685" y="213" text-anchor="middle">噪声 noise</text>
  <path d="M210,69 C330,69 330,120 418,135" fill="none" stroke="#475467" marker-end="url(#ar)"/>
  <text x="300" y="62" class="mono">x0 [B,C,H,W]</text>
  <path d="M210,239 C330,239 340,190 418,170" fill="none" stroke="#475467" marker-end="url(#ar)"/>
  <text x="228" y="296" class="mono">系数表 [T]</text>
  <path d="M380,150 L418,150" fill="none" stroke="#475467" marker-end="url(#ar)"/>
  <text x="399" y="195" text-anchor="middle" class="mono">t [B]</text>
  <path d="M590,135 L628,98" fill="none" stroke="#475467" marker-end="url(#ar)"/>
  <path d="M590,170 L628,205" fill="none" stroke="#475467" marker-end="url(#ar)"/>
  <text x="685" y="134" text-anchor="middle" class="mono">[B,C,H,W]</text>
  <text x="685" y="250" text-anchor="middle" class="mono">[B,C,H,W]</text>
  <text x="685" y="300" text-anchor="middle" class="sm">noise 是训练目标</text>
</svg>
<figcaption>训练时前三部分之间的数据流和 tensor shape。框里用的是代码变量名，对应的公式见下。</figcaption>
</figure>

图中"前向加噪"一步做的就是（第 4 节详细推导）：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon
$$

其中两个系数 $$\sqrt{\bar\alpha_t}$$、$$\sqrt{1-\bar\alpha_t}$$ 来自"噪声调度"算好的系数表，按每张图的时间步 $$t$$ 查出来；输出 `xt` 就是 $$x_t$$，`noise` 就是 $$\epsilon$$，它是后面训练网络时要预测的目标。

## 2. 数据归一化

### 2.1 是什么、为什么

DDPM 会把图像和标准高斯噪声 $$\epsilon \sim \mathcal{N}(0, I)$$ 按比例混合。噪声均值为 0、方差为 1，所以图像也要先变成**以 0 为中心、尺度和噪声相近**的数，两者混合时才"势均力敌"。做法是把像素从 $$[0, 1]$$ 线性映射到 $$[-1, 1]$$：

$$
x_0 = 2u - 1,\quad u \in [0, 1] \qquad\Longleftrightarrow\qquad u = \frac{x_0 + 1}{2}
$$

左边是喂给模型前的归一化，右边是画图前的反归一化。

### 2.2 代码

```python
import torch
from torchvision import transforms


def scale_to_minus_one_one(x):
    # x0 = 2u - 1：[0, 1] → [-1, 1]
    return x * 2.0 - 1.0


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),          # uint8 像素 0..255 → float u ∈ [0, 1]，HWC → CHW
        scale_to_minus_one_one,         # u → x0 ∈ [-1, 1]
    ])


def denormalize(images):
    # u = (x0 + 1) / 2：画图前变回 [0, 1]
    return (images.clamp(-1.0, 1.0) + 1.0) / 2.0
```

用 torchvision 的 MNIST / FashionMNIST / CIFAR-10 加上这个 transform，再交给 `DataLoader`，每个 batch 就是 `images: [B, C, H, W]`，取值在 $$[-1, 1]$$。

> **易错点**
>
> 1. **灰度图也要保留通道维 `C=1`。** MNIST 一个 batch 是 `[B, 1, 28, 28]` 而不是 `[B, 28, 28]`。后面所有代码都按 `[B, C, H, W]` 写，灰度和 RGB 才能共用一套代码；网络第一层卷积也需要这个 `C`。`ToTensor` 已经会输出 `[1, H, W]`，不要手动 `squeeze` 掉。
> 2. **transform 里不要写 `lambda`。** 很自然的写法是 `transforms.Lambda(lambda x: x * 2 - 1)`。但在 macOS 和 Windows 上，`DataLoader(num_workers > 0)` 用 spawn 方式启动子进程，需要 pickle 整个 transform，而 lambda 无法 pickle，会直接报 `Can't get local object '...<lambda>'`。`num_workers=0` 时不会暴露，一开多进程加载就崩。写成普通的模块级函数即可。
> 3. **反归一化前先 `clamp`。** 现在输入都在 $$[-1, 1]$$ 内，看起来多余；但到了采样阶段，模型输出可能略微越界，不 clamp 的话画出来会报 warning 或颜色失真。

## 3. 噪声调度（noise schedule）

### 3.1 是什么、为什么

前向过程一共 $$T$$ 步，每步加多少噪声由 $$\beta_t$$ 决定。DDPM 用最简单的**线性 schedule**：$$\beta_t$$ 从很小的 $$\beta_1 = 10^{-4}$$ 线性增加到 $$\beta_T = 0.02$$，开始时加得少、后面加得多。

由 $$\beta_t$$ 可以算出后面所有公式要用的系数：

$$
\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1),\qquad
\alpha_t = 1 - \beta_t,\qquad
\bar\alpha_t = \prod_{s=1}^{t} \alpha_s
$$

- $$\alpha_t$$：单步之后信号保留的比例；
- $$\bar\alpha_t$$：从 $$x_0$$ 走到第 $$t$$ 步，信号累计保留的比例，随 $$t$$ 单调下降。

第 7 节反向采样要用的两个量也在这里一次算好。反向一步的均值和方差是：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\Big),\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t,\qquad \bar\alpha_0 = 1
$$

它们不依赖数据，所以整张表在训练前算一次，之后只查表。

### 3.2 代码

```python
def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    # 下标约定：代码下标 i = 0..T-1 对应论文 t = i + 1

    # β_t：线性从 β_1 到 β_T
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    # α_t = 1 - β_t
    alphas = 1.0 - betas
    # ᾱ_t = α_1 · α_2 · … · α_t（cumprod 包含当前项）
    alpha_bars = torch.cumprod(alphas, dim=0)

    # 前向加噪用的两个系数：x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    # 反向采样用：μ = 1/√α_t · (x_t - β_t / √(1-ᾱ_t) · ε_θ)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # ᾱ_{t-1}：在最前面补上 ᾱ_0 = 1
    alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
    # β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · β_t
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    # 每一项都是 shape [T] 的一维 tensor
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }
```

$$T = 300$$ 时几个时间步的数值（表头是论文的 $$t$$，代码下标是 $$t-1$$）：

| | $$t=1$$ | $$t=25$$ | $$t=50$$ | $$t=100$$ | $$t=200$$ | $$t=300$$ |
|---|---|---|---|---|---|---|
| $$\beta_t$$ | 0.0001 | 0.0017 | 0.0034 | 0.0067 | 0.0133 | 0.0200 |
| $$\bar\alpha_t$$ | 0.9999 | 0.978 | 0.917 | 0.712 | 0.259 | 0.048 |
| $$\sqrt{\bar\alpha_t}$$（$$x_0$$ 的系数） | 1.000 | 0.989 | 0.958 | 0.844 | 0.509 | 0.219 |
| $$\sqrt{1-\bar\alpha_t}$$（$$\epsilon$$ 的系数） | 0.010 | 0.149 | 0.288 | 0.537 | 0.861 | 0.976 |
| $$\tilde\beta_t$$ | 0 | 0.0016 | 0.0032 | 0.0066 | 0.0133 | 0.0200 |

![噪声调度曲线](/deep-learning/assets/ddpm-from-scratch/schedule.png)

右图是前向过程最重要的直觉：随着 $$t$$ 增大，图像的权重 $$\sqrt{\bar\alpha_t}$$ 下降，噪声的权重 $$\sqrt{1-\bar\alpha_t}$$ 上升，两条曲线在 $$t \approx 140$$ 附近交叉。

> **易错点**
>
> 1. **下标差一位。** 见第 1 节。写采样循环时，代码里要写 `for i in reversed(range(T))`，`i` 从 `T-1` 走到 `0`；写成 `range(T, 0, -1)` 会在 `i = T` 时越界。
> 2. **`cumprod` 包含当前这一项。** `alpha_bars[i]` $$= \alpha_1 \cdots \alpha_{i+1}$$（代码里就是 `alphas[0] * … * alphas[i]`），不是到 `alphas[i-1]` 为止。手推时很容易少乘一项，导致和代码差一步。
> 3. **`alpha_bars_prev` 要在最前面补 1。** $$t=1$$ 时"上一步"就是干净图像 $$x_0$$，信号保留比例为 1。注意 `torch.cat` 只接受 tensor，所以要写 `torch.ones(1, device=device)` 而不是 Python 的 `1.0`，并且要和 `alpha_bars` 在同一个 device 上，否则会报错。
> 4. **$$\tilde\beta_1 = 0$$ 是故意的。** 因为 $$\bar\alpha_0 = 1$$，分子为 0：采样的最后一步只取均值，不再加噪声。副作用是，以后如果要用 $$\log \tilde\beta_t$$（比如学习方差的改进版 DDPM），会出现 `log(0) = -inf`，常见做法是用 $$\tilde\beta_2$$ 替换第一项再取 log。
> 5. **整张表只算一次。** 不要在每个 batch 里重新调用 `make_beta_schedule`，在训练开始前算好，当常量查表。

## 4. 前向加噪（forward process）

### 4.1 是什么、为什么

前向过程是一条固定的马尔可夫链，每一步在上一步的基础上加一点高斯噪声：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\ \sqrt{\alpha_t}\,x_{t-1},\ \beta_t I\big)
\quad\Longleftrightarrow\quad
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}
$$

如果真的一步步加，训练时每取一个样本都要循环 $$t$$ 次，太慢。好在高斯噪声可以合并。展开两步：

$$
x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-2} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}}_{\text{两个独立高斯之和，方差 } \alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) \,=\, 1-\alpha_t\alpha_{t-1}}
$$

两个独立高斯之和仍是高斯，方差相加，所以可以写成一个 $$\sqrt{1-\alpha_t\alpha_{t-1}}\,\epsilon$$。一直展开到 $$x_0$$，就得到**闭式解**：

$$
\boxed{\,x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\qquad \epsilon \sim \mathcal{N}(0, I)\,}
$$

这一行公式是整个 DDPM 的核心，它带来两个直接后果：

- **训练时不用循环**：随机抽一个 $$t$$，一步算出 $$x_t$$。
- **训练目标就是 $$\epsilon$$**：网络看到 $$(x_t, t)$$，去预测当初加进去的那份噪声 $$\epsilon$$。所以实现时要把 $$\epsilon$$ 一起返回。

另外注意两个系数的平方和恒等于 1：$$(\sqrt{\bar\alpha_t})^2 + (\sqrt{1-\bar\alpha_t})^2 = 1$$。如果 $$x_0$$ 的方差接近 1，$$x_t$$ 的方差在每个时刻都保持在 1 附近，不会爆掉或塌缩。这也是第 2 节要把数据归一化的另一个原因。

### 4.2 代码

训练时一个 batch 里每张图独立抽一个时间步，所以 `t` 是长度为 $$B$$ 的向量，每张图用自己的系数：

```python
def q_sample(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, noise=None):
    # x0: [B, C, H, W]   干净图像
    # t:  [B]            每张图各自的时间步（0-based 下标）
    # sqrt_alpha_bars, sqrt_one_minus_alpha_bars: [T]  噪声调度表
    #
    # x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε

    # ε ~ N(0, I)，和 x0 同 shape、同 device、同 dtype
    if noise is None:
        noise = torch.randn_like(x0)

    # 下标和查表都放到 x0 所在的 device 上，系数转成 x0 的 dtype
    t = t.to(device=x0.device, dtype=torch.long)
    sqrt_alpha_bars = sqrt_alpha_bars.to(device=x0.device, dtype=x0.dtype)
    sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.to(device=x0.device, dtype=x0.dtype)

    # 按图查表：[T] --索引 t--> [B] --view--> [B, 1, 1, 1]
    sqrt_alpha_bar_t = sqrt_alpha_bars[t].view(-1, 1, 1, 1)                  # √ᾱ_t
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)  # √(1-ᾱ_t)

    # [B,1,1,1] 和 [B,C,H,W] 广播相乘：同一张图的所有像素用同一个系数
    xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    return xt, noise   # noise 就是训练目标 ε
```

训练时的用法（第 6 节会完整写出来）：

```python
t = torch.randint(0, T, (x0.shape[0],))          # [B]，每张图随机一个时间步
xt, noise = q_sample(x0, t, sched["sqrt_alpha_bars"], sched["sqrt_one_minus_alpha_bars"])
# loss = MSE(model(xt, t), noise)
```

> **易错点**
>
> 1. **`[B]` 必须先变成 `[B, 1, 1, 1]` 再乘。** PyTorch 广播从**最后一维**开始对齐：`[B]` 直接乘 `[B, C, H, W]` 时，`B` 会去和 `W` 对齐。大多数时候会报错；但如果碰巧 `B == W`（比如 batch 是 28、图是 28×28），它**不报错而是悄悄算错**：每一列像素乘上了不同图的系数。`.view(-1, 1, 1, 1)` 让第 $$i$$ 张图的所有像素都乘第 $$i$$ 个系数。
> 2. **`t` 是向量不是标量。** 每张图独立随机抽时间步，一个 batch 就能覆盖多个噪声水平，梯度方差更小。`sqrt_alpha_bars[t]` 是用整数 tensor 做索引，一次取出 $$B$$ 个系数。
> 3. **索引必须是 `long`，并且和表在同一个 device 上。** 调度表通常在 CPU 上建好，图像在 GPU / MPS 上；用 GPU 上的 `t` 去索引 CPU 上的表会报 device 不一致。上面的代码统一搬到 `x0.device`，并把系数转成 `x0.dtype`，以后用半精度训练时也不会因为 float32 × float16 发生类型提升。
> 4. **一定要返回同一份 `noise`。** 训练 loss 是 $$\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$$，目标必须是**生成这个 $$x_t$$ 时用的那份噪声**。如果在外面另外随机一份当目标，目标和输入就没有关系了，loss 永远降不下去。
> 5. **允许从外面传入 `noise`。** 检查代码时传一份固定的噪声，就能把结果和公式逐元素对比（见第 11 节）。

### 4.3 结果

FashionMNIST 上的前向加噪，每行一张图，第一列是 $$x_0$$，后面依次是 $$t = 25, 50, 100, 200, 300$$：

![FashionMNIST 前向加噪](/deep-learning/assets/ddpm-from-scratch/forward_fashion_mnist.png)

几个观察：

1. $$t \le 50$$ 时图像几乎不变，噪声主要在中后段才快速占上风，和第 3 节的曲线一致。
2. **$$T = 300$$ 时最后一步并不是纯噪声**：$$\sqrt{\bar\alpha_{300}} \approx 0.22$$，仔细看最后一列仍能隐约看到轮廓。DDPM 原论文用 $$T = 1000$$、同样的 $$\beta$$ 范围，$$\sqrt{\bar\alpha_{1000}} \approx 0.006$$，才接近纯噪声。这里选 300 是为了在小数据集上训练、采样更快；代价是采样起点 $$x_T \sim \mathcal{N}(0, I)$$ 和真实的 $$x_T$$ 分布有偏差。这一点在第 10 节讨论误差放大时还会用到。

### 4.4 可视化技巧：一次算出"每张图 × 每个时间步"

上面那张图要把 3 张图各自在 5 个时间步下的结果一次算出来。做法是构造两个对齐的长度为 15 的 batch，第 $$k$$ 个元素是"第 $$\lfloor k/5 \rfloor$$ 张图在第 $$k \bmod 5$$ 个时间步"：

```python
paper_t = [25, 50, 100, 200, 300]
steps = torch.tensor([t - 1 for t in paper_t])        # 转成 0-based 下标，[5]

x0_batch = x0.repeat_interleave(len(steps), dim=0)   # [3,1,28,28] → [15,1,28,28]
t_batch = steps.repeat(x0.shape[0])                   # [5] → [15]
xt, _ = q_sample(x0_batch, t_batch, sched["sqrt_alpha_bars"], sched["sqrt_one_minus_alpha_bars"])
```

> **易错点：`repeat_interleave` 和 `repeat` 的区别**
>
> `repeat_interleave` 是**逐元素**复制，`repeat` 是**整体**复制。拿一张 RGB 图 `img` 的 shape `[3, 2, 2]` 举例：
>
> - `img.repeat_interleave(2, dim=0)` → `[R, R, G, G, B, B]`
> - `img.repeat(2, 1, 1)` → `[R, G, B, R, G, B]`
>
> 两者输出 shape 完全相同（都是 `[6, 2, 2]`），都不增加新维度，只是第 0 维上的顺序不同。所以**写反了不会报错，只是图和时间步错位**。这里图片要"每张连续出现 5 次"，用 `repeat_interleave`；时间步要"整段重复 3 次"，用 `repeat`。再用 `torchvision.utils.make_grid(..., nrow=5)` 按行排开，正好一张图一行。

## 5. 噪声预测网络

### 5.1 是什么、为什么

反向生成时，我们需要一个网络 $$\epsilon_\theta(x_t, t)$$：输入带噪图像 $$x_t$$ 和时间步 $$t$$，输出它对"$$x_t$$ 里含有的那份噪声"的估计，shape 和 $$x_t$$ 完全一样：

$$
\epsilon_\theta:\ \big(x_t \in \mathbb{R}^{B\times C\times H\times W},\ t \in \{1,\dots,T\}^B\big) \ \longmapsto\ \hat\epsilon \in \mathbb{R}^{B\times C\times H\times W}
$$

这是一个"图像进、图像出"的任务，所以图像 diffusion 几乎都用 **U-Net**：编码器逐步下采样抓全局结构，解码器上采样恢复分辨率，skip connection 把高分辨率的细节直接送到解码器。和普通 U-Net 唯一的本质区别是：**网络必须知道当前是第几步**。同一张模糊的图，在 $$t=10$$ 时噪声很少、只需要轻微修补，在 $$t=290$$ 时几乎全是噪声。所以要把 $$t$$ 编码成向量，注入网络的每一层。

本节由三个部件组成：时间步编码、带时间条件的残差块、一个只下采样一次的 Tiny U-Net。

### 5.2 时间步编码（sinusoidal timestep embedding）

整数 $$t$$ 不适合直接喂给网络（数值范围大，相邻步之间也没有"相似"的结构）。常用做法和 Transformer 的位置编码一样：用一组不同频率的 sin / cos 把 $$t$$ 编码成长度为 $$d$$（代码里叫 `time_dim`）的向量。令 $$D = d/2$$：

$$
\omega_i = \exp\!\Big(-\frac{\ln 10000 \cdot i}{D-1}\Big),\quad i = 0,\dots,D-1
$$

$$
\operatorname{emb}(t) = \big[\sin(t\omega_0), \dots, \sin(t\omega_{D-1}),\ \cos(t\omega_0), \dots, \cos(t\omega_{D-1})\big] \in [-1, 1]^{d}
$$

$$\omega_0 = 1$$ 是最高频，$$\omega_{D-1} = 10^{-4}$$ 是最低频。低频维度区分"大概在前段还是后段"，高频维度区分相邻的步。

```python
import math

def sinusoidal_timestep_embedding(t, time_dim):
    # t: [B] long；time_dim = d，必须是 >= 4 的偶数
    if time_dim < 4 or time_dim % 2 != 0:
        raise ValueError("time_dim must be an even integer >= 4")

    half_dim = time_dim // 2                                   # D
    # ω_i = exp(-ln(10000) · i / (D - 1))，shape [D]
    omegas = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=t.device) / (half_dim - 1)
    )
    # angles[b, i] = t_b · ω_i：[B, 1] * [1, D] → [B, D]
    angles = t.float().unsqueeze(1) * omegas.unsqueeze(0)
    # [sin, cos] 拼接：[B, D] + [B, D] → [B, d]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
```

| 公式 | 代码 | shape |
|---|---|---|
| $$t$$ | `t` | `[B]` |
| $$D = d/2$$ | `half_dim` | 标量 |
| $$\omega_i$$ | `omegas` | `[D]` |
| $$t\,\omega_i$$ | `angles` | `[B, D]` |
| $$\operatorname{emb}(t)$$ | 返回值 | `[B, d]` |

> **易错点**
>
> 1. **`time_dim` 不是 `T`。** $$T$$ 是总步数（300），`time_dim` 是每个 $$t$$ 编码成的向量长度（128）。两者毫无关系，常被混为一谈。
> 2. **`time_dim` 必须是偶数且至少为 4。** 一半给 sin、一半给 cos，所以必须是偶数；分母是 $$D-1$$，`time_dim = 2` 时会除以 0。有的实现分母用 $$D$$ 而不是 $$D-1$$，两者都可以，只是最低频略有不同。
> 3. **训练和采样必须用同一种 $$t$$ 编号。** 这里直接把 0-based 下标 `t` 送进编码（论文的 $$t$$ 减 1）。只要训练和采样一致就没问题；如果训练用下标、采样时却传了论文的 $$t$$，整体就会错开一步。
> 4. **先 `t.float()` 再乘。** `t` 是 long，直接和 float 相乘虽然会自动提升类型，但显式转换更清楚，也避免某些后端对整数 tensor 的限制。

### 5.3 带时间条件的残差块（TimeResBlock）

残差块是 U-Net 的基本单元。加时间条件的做法很简单：把时间向量经过一个线性层投影到和特征图相同的通道数，当作**每个通道的偏置**加到特征上：

$$
h = \operatorname{Conv}_1(x) + W_t\,\operatorname{emb}(t),\qquad
\text{out} = \operatorname{SiLU}\big(\operatorname{Conv}_2(\operatorname{SiLU}(h)) + \operatorname{skip}(x)\big)
$$

其中 $$W_t\,\operatorname{emb}(t)$$ 的 shape 是 `[B, C_out]`，扩展成 `[B, C_out, 1, 1]` 后广播到所有像素：同一张图的同一个通道，所有位置加同一个数。

```python
import torch.nn as nn
import torch.nn.functional as F

class TimeResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)                     # W_t
        # 残差相加要求通道数一致：不一致时用 1x1 卷积对齐
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, time_emb):
        # x: [B, in_ch, H, W]；time_emb: [B, time_dim]
        h = self.conv1(x)                                        # [B, out, H, W]
        time_bias = self.time_proj(time_emb)[:, :, None, None]   # [B, out] → [B, out, 1, 1]
        h = h + time_bias                                        # 广播到 H, W
        h = F.silu(h)
        h = self.conv2(h)                                        # [B, out, H, W]
        return F.silu(h + self.skip(x))                          # [B, out, H, W]
```

> **易错点**
>
> 1. **时间偏置要扩成 `[B, C, 1, 1]`。** 和前向加噪里的 `[B] → [B, 1, 1, 1]` 是同一个道理：`[B, C]` 直接加 `[B, C, H, W]` 会按最后两维对齐而报错，或在 `C == W` 时悄悄算错。
> 2. **通道数变化时 skip 必须投影。** `in_ch != out_ch` 时 `x` 和 `h` 不能直接相加，用 1×1 卷积把 `x` 变到 `out_ch` 通道。
> 3. **最后要不要再接 SiLU？** 这里写的是 $$\operatorname{SiLU}(h + \operatorname{skip}(x))$$。很多 diffusion 实现的残差块最后不再加激活，直接返回 `h + skip(x)`，让残差通路保持线性。两种都能训练，这只是一个设计选择。
> 4. **教学版省略了归一化。** 实际的 diffusion U-Net 每个卷积前通常有 GroupNorm，并有 dropout。这里去掉是为了把注意力集中在"时间条件怎么注入"上，模型小、数据简单时影响不大。

### 5.4 Tiny U-Net

只下采样一次、上采样一次的最小 U-Net。时间编码在最前面算一次，经过一个小 MLP，然后**同一个** `time_emb` 送进所有残差块。

<figure class="ddpm-fig">
<svg viewBox="0 0 760 300" role="img" aria-label="Tiny U-Net 结构">
  <defs><marker id="ar3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker></defs>
  <rect x="10" y="40" width="90" height="44" rx="8" fill="#fff" stroke="#475467"/><text x="55" y="60" text-anchor="middle">输入 x_t</text><text x="55" y="76" text-anchor="middle" class="mono">[B,C,H,W]</text>
  <rect x="130" y="40" width="110" height="44" rx="8" fill="#eef4ff" stroke="#2a6fdb"/><text x="185" y="60" text-anchor="middle">encoder</text><text x="185" y="76" text-anchor="middle" class="mono">[B,c,H,W]</text>
  <rect x="200" y="150" width="110" height="44" rx="8" fill="#f8f9fa" stroke="#98a2b3"/><text x="255" y="170" text-anchor="middle">down 卷积 /2</text><text x="255" y="186" text-anchor="middle" class="mono">[B,2c,H/2,W/2]</text>
  <rect x="330" y="220" width="120" height="44" rx="8" fill="#eef4ff" stroke="#2a6fdb"/><text x="390" y="240" text-anchor="middle">bottleneck</text><text x="390" y="256" text-anchor="middle" class="sm">（可选：自注意力）</text>
  <rect x="470" y="150" width="110" height="44" rx="8" fill="#f8f9fa" stroke="#98a2b3"/><text x="525" y="170" text-anchor="middle">up 转置卷积 ×2</text><text x="525" y="186" text-anchor="middle" class="mono">[B,c,H,W]</text>
  <rect x="530" y="40" width="110" height="44" rx="8" fill="#eef4ff" stroke="#2a6fdb"/><text x="585" y="60" text-anchor="middle">decoder</text><text x="585" y="76" text-anchor="middle" class="mono">in 2c → out c</text>
  <rect x="660" y="40" width="90" height="44" rx="8" fill="#fff4ec" stroke="#e0662b"/><text x="705" y="60" text-anchor="middle">pred_noise</text><text x="705" y="76" text-anchor="middle" class="mono">[B,C,H,W]</text>
  <path d="M100,62 L128,62" stroke="#475467" fill="none" marker-end="url(#ar3)"/>
  <path d="M200,84 L240,148" stroke="#475467" fill="none" marker-end="url(#ar3)"/>
  <path d="M290,194 L345,218" stroke="#475467" fill="none" marker-end="url(#ar3)"/>
  <path d="M435,218 L490,194" stroke="#475467" fill="none" marker-end="url(#ar3)"/>
  <path d="M545,148 L575,86" stroke="#475467" fill="none" marker-end="url(#ar3)"/>
  <path d="M640,62 L658,62" stroke="#475467" fill="none" marker-end="url(#ar3)"/>
  <path d="M240,52 L528,52" stroke="#2a6fdb" stroke-dasharray="5 4" fill="none" marker-end="url(#ar3)"/>
  <text x="384" y="44" text-anchor="middle" class="sm">skip：按通道拼接（concat）</text>
  <rect x="300" y="100" width="170" height="34" rx="8" fill="#fff" stroke="#e0662b" stroke-dasharray="4 3"/>
  <text x="385" y="122" text-anchor="middle" class="sm">time_emb → 每个残差块</text>
</svg>
<figcaption>Tiny U-Net。c 为 base_channels。蓝色框是带时间条件的残差块。</figcaption>
</figure>

```python
class TinyUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_dim=128, use_attention=False):
        super().__init__()
        self.time_dim = time_dim
        c = base_channels
        # 时间编码之后再过一个小 MLP，让网络学习怎么使用它
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.encoder = TimeResBlock(in_channels, c, time_dim)
        self.down = nn.Conv2d(c, 2 * c, kernel_size=4, stride=2, padding=1)          # H → H/2
        self.bottleneck = TimeResBlock(2 * c, 2 * c, time_dim)
        self.attention = SpatialSelfAttention(2 * c) if use_attention else nn.Identity()  # 见 5.5
        self.up = nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1)   # H/2 → H
        self.decoder = TimeResBlock(2 * c, c, time_dim)       # 输入是 [up, skip] 拼接，2c 通道
        self.output = nn.Conv2d(c, in_channels, kernel_size=3, padding=1)

    def forward(self, x_t, t):
        time_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))  # [B, time_dim]
        skip = self.encoder(x_t, time_emb)            # [B, c, H, W]
        h = self.down(skip)                           # [B, 2c, H/2, W/2]
        h = self.bottleneck(h, time_emb)              # [B, 2c, H/2, W/2]
        h = self.attention(h)                         # [B, 2c, H/2, W/2]
        h = self.up(h)                                # [B, c, H, W]
        h = torch.cat([h, skip], dim=1)               # [B, 2c, H, W]
        h = self.decoder(h, time_emb)                 # [B, c, H, W]
        return self.output(h)                         # [B, C, H, W] = ε_θ(x_t, t)
```

> **易错点**
>
> 1. **H、W 必须能被 2 整除。** `kernel_size=4, stride=2, padding=1` 的卷积把 28 变成 14，转置卷积再把 14 变回 28。如果输入是奇数（比如 27），下采样后再上采样回不到原尺寸，`torch.cat` 时 skip 和 `h` 的尺寸对不上。多次下采样时要求更严：28 只能下采样两次（28→14→7），第三次 7 就不能整除了。
> 2. **拼接之后通道数翻倍。** `torch.cat([h, skip], dim=1)` 得到 `2c` 通道，所以 decoder 的输入通道必须写 `2c`，不是 `c`。skip connection 用拼接而不是相加，是 U-Net 的标准做法。
> 3. **输出层不要加激活。** 噪声 $$\epsilon$$ 是无界的高斯，用 `tanh` 或 `sigmoid` 会把输出压到有限区间，网络就无法表达较大的噪声值。最后一层就是一个普通卷积。
> 4. **输出通道等于输入通道。** 预测的是和 $$x_t$$ 同 shape 的噪声，所以 `self.output` 的输出通道是 `in_channels`（灰度 1、RGB 3）。

### 5.5 可选：在 bottleneck 加自注意力

卷积只能看到局部，自注意力让每个位置都能看到整张特征图。做法是把特征图的每个像素当成一个 token：

$$
[B, C, H, W] \xrightarrow{\text{flatten}} [B, HW, C] \xrightarrow{\ \operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(QK^\top/\sqrt{d}\right)V\ } [B, HW, C] \xrightarrow{\text{reshape}} [B, C, H, W]
$$

```python
class SpatialSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)           # [B, C, H*W] → [B, H*W, C]
        q = self.norm(tokens)
        attended, _ = self.attention(q, q, q, need_weights=False)   # Q = K = V：自注意力
        tokens = tokens + attended                      # 残差
        return tokens.transpose(1, 2).reshape(B, C, H, W)
```

> **易错点**
>
> 1. **放在最低分辨率。** 注意力的计算量和 token 数的平方成正比。28×28 有 784 个 token，14×14 只有 196 个，计算量差 16 倍。所以通常只在 bottleneck 或低分辨率层加注意力。
> 2. **先 `transpose` 再 `reshape`，顺序不能省。** `[B, H*W, C]` 不能直接 `reshape(B, C, H, W)`：shape 对，但内存里的数据顺序不对，结果是乱的。必须先 `transpose(1, 2)` 回 `[B, C, H*W]` 再 reshape。
> 3. **`batch_first=True`。** `nn.MultiheadAttention` 默认输入是 `[seq, batch, dim]`，不设这个参数就会把 batch 维和 token 维搞反，而且不会报错。
> 4. **`channels` 要能被 `num_heads` 整除。** 每个头分到 `C / num_heads` 维。

## 6. 训练

### 6.1 是什么、为什么

训练不需要走完整条链。每个 batch 只做四件事：随机抽时间步、随机抽噪声、用第 4 节的闭式公式一步造出 $$x_t$$、让网络预测那份噪声：

$$
t \sim \operatorname{Uniform}\{1,\dots,T\},\qquad
\epsilon \sim \mathcal{N}(0, I),\qquad
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon
$$

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\,t,\,\epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta(x_t, t) \big\rVert^2\Big]
$$

这就是 DDPM 的全部训练目标：一个普通的 MSE。它是从变分下界（ELBO）简化来的，推导见之前的笔记；写代码只需要这一行。

两个常见问题：

- **为什么每张图随机抽 $$t$$？** 网络要在所有噪声水平下都会去噪。同一个 batch 里的图各自抽一个 $$t$$，一次前向就覆盖了很多噪声水平。
- **为什么预测噪声，而不是直接预测 $$x_0$$？** 训练时真实噪声 $$\epsilon$$ 是已知的，目标始终是方差为 1 的标准高斯，尺度稳定；而且预测噪声和 score matching 等价，DDPM 论文实验里效果也最好。第 10 节会把 $$\epsilon$$、$$x_0$$、$$v$$ 三种目标放在一起比较。

### 6.2 代码

一个 batch 的训练步：

```python
def ddpm_training_step(model, optimizer, x0, sched):
    model.train()
    T = sched["betas"].shape[0]
    # t ~ Uniform：每张图一个，0-based 下标，[B]
    t = torch.randint(0, T, (x0.shape[0],), device=x0.device)
    # ε ~ N(0, I) 和 x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε（第 4 节的 q_sample）
    xt, noise = q_sample(x0, t, sched["sqrt_alpha_bars"], sched["sqrt_one_minus_alpha_bars"])
    # ε_θ(x_t, t)
    pred_noise = model(xt, t)                                # [B, C, H, W]
    # L = ||ε - ε_θ||²
    loss = F.mse_loss(pred_noise, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach()
```

外层循环只负责遍历 epoch 和 batch：

```python
def train_ddpm(model, optimizer, dataloader, sched, num_epochs, device):
    losses = []
    for epoch in range(num_epochs):
        for images, _ in dataloader:
            x0 = images.to(device)                           # [B, C, H, W]，已归一化到 [-1, 1]
            loss = ddpm_training_step(model, optimizer, x0, sched)
            losses.append(loss.item())
    return losses
```

| 公式 | 代码 | shape |
|---|---|---|
| $$t \sim \operatorname{Uniform}$$ | `torch.randint(0, T, (B,))` | `[B]` |
| $$\epsilon$$ | `noise`（`q_sample` 返回） | `[B, C, H, W]` |
| $$x_t$$ | `xt` | `[B, C, H, W]` |
| $$\epsilon_\theta(x_t, t)$$ | `model(xt, t)` | `[B, C, H, W]` |
| $$\mathcal{L}_{\text{simple}}$$ | `F.mse_loss(pred_noise, noise)` | 标量 |

> **易错点**
>
> 1. **loss 的目标必须是 `q_sample` 返回的那份 `noise`。** 见第 4 节。
> 2. **`torch.randint(0, T, ...)` 的上界不包含 `T`。** 得到的是 `0 … T-1`，正好是所有合法下标；写成 `randint(1, T+1)` 就会越界。
> 3. **记录 loss 用 `.item()` 或 `.detach()`。** 直接把带计算图的 `loss` 存进列表，计算图不会释放，显存（内存）会一直涨。
> 4. **loss 下降得快、之后很平是正常的。** 预测噪声的 MSE 在大 $$t$$ 时很容易（$$x_t$$ 几乎就是噪声），在小 $$t$$ 时很难（噪声很弱，难分辨），平均下来会很快降下来，然后几乎持平（本文的运行在 1000 步时约 0.09，4000 步时约 0.077）。loss 不能直接反映生成质量，要看采样结果。
> 5. **实际训练通常还会用 EMA。** 采样时用模型参数的指数滑动平均（EMA）而不是最新参数，生成质量更稳定。教学版为了简单没有加。

### 6.3 结果

在 FashionMNIST 上训练：Tiny U-Net（`base_channels=48`，约 46 万参数）、$$T=300$$、batch 128、AdamW 学习率 $$10^{-3}$$、4000 步，CPU 上约 25 分钟。这里训练的其实是第 9 节的带类别条件的版本（10% 的样本丢掉类别标签），所以同一个模型既能无条件生成，也能按类别生成。

![训练 loss](/deep-learning/assets/ddpm-from-scratch/loss.png)

灰线是每一步的 loss，蓝线是 50 步滑动平均：前 500 步降得很快，之后进入缓慢下降的平台，最后约 0.077。

## 7. 采样（反向过程）

### 7.1 是什么、为什么

生成时没有 $$x_0$$，只能从纯噪声 $$x_T \sim \mathcal{N}(0, I)$$ 出发，一步步往回走：$$x_T \to x_{T-1} \to \cdots \to x_0$$。每一步需要知道 $$p(x_{t-1} \mid x_t)$$。

关键事实：如果**同时知道** $$x_t$$ 和 $$x_0$$，前向过程的"反方向"是一个可以算出来的高斯（真实后验）：

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(x_{t-1};\ \tilde\mu_t(x_t, x_0),\ \tilde\beta_t I\big)
$$

$$
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,
\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
$$

采样时 $$x_0$$ 未知，但网络给出了噪声估计，由第 4 节的闭式公式反解就得到 $$x_0$$ 的估计：

$$
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}
$$

把 $$\hat x_0$$ 代进 $$\tilde\mu_t$$，化简后得到只含 $$x_t$$ 和 $$\epsilon_\theta$$ 的形式，也就是 DDPM 的反向一步：

$$
\boxed{\ \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\Big),\qquad
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z,\quad z \sim \mathcal{N}(0, I)\ }
$$

其中 $$\sigma_t^2 = \tilde\beta_t$$（第 3 节已经算好的 `posterior_variance`）。最后一步 $$t = 1$$ 时 $$\tilde\beta_1 = 0$$，不加噪声。$$z$$ 是每一步**新抽**的随机噪声，不是网络输出，它让生成有多样性。

### 7.2 代码

```python
@torch.no_grad()
def p_sample(model, x_t, t_index, sched):
    # t_index：Python int，0-based（论文 t = t_index + 1）
    # 网络需要 [B] 的时间步：整个 batch 用同一个 t
    t = torch.full((x_t.shape[0],), t_index, device=x_t.device, dtype=torch.long)
    pred_noise = model(x_t, t)                               # ε_θ(x_t, t)，[B, C, H, W]

    # 整个 batch 同一个 t，系数是标量，直接广播
    beta_t = sched["betas"][t_index]                                         # β_t
    sqrt_recip_alpha_t = sched["sqrt_recip_alphas"][t_index]                 # 1/√α_t
    sqrt_one_minus_alpha_bar_t = sched["sqrt_one_minus_alpha_bars"][t_index] # √(1-ᾱ_t)

    # μ_θ = 1/√α_t · (x_t − β_t / √(1-ᾱ_t) · ε_θ)
    mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise)
    if t_index == 0:                                         # 最后一步：不加噪声
        return mean
    z = torch.randn_like(x_t)                                # z ~ N(0, I)
    # x_{t-1} = μ_θ + √(β̃_t) · z
    return mean + torch.sqrt(sched["posterior_variance"][t_index]) * z


@torch.no_grad()
def sample_ddpm(model, shape, sched, device):
    model.eval()
    sched = {k: v.to(device) for k, v in sched.items()}
    x_t = torch.randn(shape, device=device)                  # x_T ~ N(0, I)
    for t_index in reversed(range(sched["betas"].shape[0])): # T-1, …, 0
        x_t = p_sample(model, x_t, t_index, sched)
    return x_t                                               # x_0，大致在 [-1, 1]
```

| 公式 | 代码 |
|---|---|
| $$\epsilon_\theta(x_t, t)$$ | `pred_noise` |
| $$1/\sqrt{\alpha_t}$$、$$\beta_t$$、$$\sqrt{1-\bar\alpha_t}$$ | `sqrt_recip_alpha_t`、`beta_t`、`sqrt_one_minus_alpha_bar_t` |
| $$\mu_\theta(x_t, t)$$ | `mean` |
| $$\sigma_t = \sqrt{\tilde\beta_t}$$ | `torch.sqrt(sched["posterior_variance"][t_index])` |
| $$z$$ | `z` |
| $$x_{t-1}$$ | 返回值 |

> **易错点**
>
> 1. **`posterior_variance` 是方差，乘 $$z$$ 之前要开根号。** 直接乘 $$\tilde\beta_t$$ 会让加的噪声小一个数量级，生成结果偏平滑、缺细节，而且不报错。
> 2. **循环变量是 Python int，网络要的是 `[B]` tensor。** `p_sample` 里用 `torch.full` 构造；查系数表时直接用 int 下标，得到的是标量，不需要 `view(-1, 1, 1, 1)`。
> 3. **最后一步不加噪声。** `t_index == 0` 时直接返回均值。即使忘了判断，因为 $$\tilde\beta_1 = 0$$ 也只是乘了 0；但如果改用 $$\sigma_t^2 = \beta_t$$（另一种常见选择，$$\beta_1 \ne 0$$），就会在最后一步多加一层噪声。
> 4. **`@torch.no_grad()` 和 `model.eval()`。** 采样要跑 $$T$$ 次前向，不关梯度会保存整条链的计算图，内存很快爆掉。
> 5. **采样慢是 DDPM 的固有问题。** 生成一批图要做 $$T$$ 次网络前向（这里 300 次，原论文 1000 次）。第 8 节的 DDIM 就是为了解决这个问题。
> 6. **显示前要 `clamp` 和反归一化。** 输出 $$x_0$$ 大致在 $$[-1, 1]$$，但可能略微越界，用第 2 节的 `denormalize` 处理。有的实现还会在每一步把 $$\hat x_0$$ 裁到 $$[-1, 1]$$ 再算均值，第 10 节的统一写法里会看到。

### 7.3 结果

用 6.3 训练好的模型，传入空类别 $$\varnothing$$ 做无条件生成，300 步，一次生成 32 张图在 CPU 上约 8 秒：

![DDPM 无条件生成](/deep-learning/assets/ddpm-from-scratch/samples_ddpm.png)

一个样本在反向过程中的样子，从左到右 $$t$$ 从 300 减到 0：

![反向过程](/deep-learning/assets/ddpm-from-scratch/trajectory.png)

和前向过程正好相反：前面的大部分步骤只是在噪声里确定大致轮廓，衣服的细节集中在最后几十步才出现。

这个小模型只训练了 4000 步，无条件样本能看出是衣服、鞋、包，但边缘和纹理还比较乱。加大网络、训练更久、加上 EMA 都会明显改善；下面第 9 节加上类别条件后效果也好很多。

## 8. DDIM：少步采样

### 8.1 是什么、为什么

DDIM 的出发点：训练目标只用到了边缘分布 $$q(x_t \mid x_0)$$（第 4 节的闭式公式），从来没用过"相邻两步"的转移。所以可以**不改训练、不重训模型**，换一种反向过程，从 $$t$$ 直接跳到更早的任意一步 $$s < t$$。

先用网络的噪声估计算出 $$\hat x_0$$（和 7.1 一样），再把它"重新加噪"到第 $$s$$ 步。加噪时用的噪声方向就是网络预测的 $$\epsilon_\theta$$：

$$
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}
$$

$$
x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\;\epsilon_\theta(x_t, t) + \sigma_{t\to s}\,z,
\qquad
\sigma_{t\to s} = \eta\sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}\Big(1-\frac{\bar\alpha_t}{\bar\alpha_s}\Big)}
$$

- $$\eta = 0$$：没有随机项，**确定性**采样，同一个初始噪声永远得到同一张图。
- $$\eta = 1$$ 且 $$s = t-1$$：退化回 DDPM。
- 最后一步跳到"$$s = 0$$"，约定 $$\bar\alpha_0 = 1$$，此时 $$x_s = \hat x_0$$。

### 8.2 代码

```python
@torch.no_grad()
def ddim_step(model, x_t, t_index, prev_index, sched, eta=0.0):
    # 从下标 t_index 跳到更早的 prev_index；prev_index = -1 表示跳到干净图像
    t = torch.full((x_t.shape[0],), t_index, device=x_t.device, dtype=torch.long)
    pred_noise = model(x_t, t)                                           # ε_θ
    abar_t = sched["alpha_bars"][t_index]                                # ᾱ_t
    abar_s = sched["alpha_bars"][prev_index] if prev_index >= 0 else abar_t.new_tensor(1.0)  # ᾱ_s

    # x̂0 = (x_t − √(1-ᾱ_t) · ε_θ) / √ᾱ_t
    x0_hat = (x_t - torch.sqrt(1 - abar_t) * pred_noise) / torch.sqrt(abar_t)
    # σ² = η² · (1-ᾱ_s)/(1-ᾱ_t) · (1 − ᾱ_t/ᾱ_s)
    sigma2 = eta**2 * (1 - abar_s) / (1 - abar_t) * (1 - abar_t / abar_s)
    # √(1 − ᾱ_s − σ²)，clamp 防止数值误差出现负数
    dir_coef = torch.sqrt(torch.clamp(1 - abar_s - sigma2, min=0.0))
    # x_s = √ᾱ_s · x̂0 + √(1-ᾱ_s-σ²) · ε_θ + σ · z
    return torch.sqrt(abar_s) * x0_hat + dir_coef * pred_noise + torch.sqrt(sigma2) * torch.randn_like(x_t)


@torch.no_grad()
def sample_ddim(model, shape, sched, device, num_steps=50, eta=0.0):
    model.eval()
    sched = {k: v.to(device) for k, v in sched.items()}
    T = sched["betas"].shape[0]
    # 只访问 num_steps 个时间步，例如 299, 293, …, 6, 0
    steps = torch.linspace(T - 1, 0, num_steps).round().long().tolist()
    x_t = torch.randn(shape, device=device)                  # x_T
    for i, t_index in enumerate(steps):
        prev_index = steps[i + 1] if i + 1 < len(steps) else -1
        x_t = ddim_step(model, x_t, t_index, prev_index, sched, eta)
    return x_t
```

> **易错点**
>
> 1. **不用重新训练。** DDIM 只换采样器，网络、schedule、训练目标全部复用 DDPM 的。
> 2. **`prev_index` 不是 `t_index - 1`。** 这正是能跳步的关键；最后一步用 `-1` 表示"跳到干净图像"，对应 $$\bar\alpha = 1$$。
> 3. **根号里要 `clamp`。** 理论上 $$1 - \bar\alpha_s - \sigma^2 \ge 0$$，但浮点误差可能给出极小的负数，`sqrt` 会得到 `nan`，并且一路传染到整张图。
> 4. **`num_steps` 不能超过 $$T$$。** 否则 `linspace` 取整后会出现重复的时间步，`prev_index == t_index` 时 $$\bar\alpha_t / \bar\alpha_s = 1$$，这一步什么也没做，浪费计算；步数太少（比如 5 步）质量会明显下降。

### 8.3 结果

同一个模型，DDIM 50 步（$$\eta = 0$$），32 张图约 1.2 秒，是 300 步 DDPM 的 6 倍快：

![DDIM 50 步](/deep-learning/assets/ddpm-from-scratch/samples_ddim.png)

质量和 300 步 DDPM 差不多，说明不重新训练、只换采样器，就能省下大部分时间。

## 9. 条件生成与 classifier-free guidance

### 9.1 类别条件

想指定生成哪一类（比如只生成"鞋"），最简单的做法是把类别标签也编码成向量，和时间编码**相加**，之后的网络完全不变：

$$
\operatorname{cond}(t, y) = \operatorname{MLP}\big(\operatorname{emb}(t)\big) + E[y],\qquad y \in \{0,\dots,K-1\}
$$

$$E$$ 是一个可学习的 embedding 表。为了下一节的 CFG，再额外留一个"空类别" $$y = \varnothing$$（下标 $$K$$），表示"不给条件"。

```python
class ClassConditionedTinyUNet(TinyUNet):
    def __init__(self, num_classes, in_channels=1, base_channels=32, time_dim=128):
        super().__init__(in_channels, base_channels, time_dim)
        self.null_class_id = num_classes                              # ∅ 的下标 = K
        self.class_embedding = nn.Embedding(num_classes + 1, time_dim)  # E，K+1 行

    def forward(self, x_t, t, labels=None):
        if labels is None:                                            # 不给条件 = 用 ∅
            labels = torch.full((x_t.shape[0],), self.null_class_id, device=x_t.device, dtype=torch.long)
        time_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))
        cond = time_emb + self.class_embedding(labels)                # [B, time_dim]
        # 下面和 TinyUNet 一样，只是把 time_emb 换成 cond
        skip = self.encoder(x_t, cond)
        h = self.attention(self.bottleneck(self.down(skip), cond))
        h = torch.cat([self.up(h), skip], dim=1)
        return self.output(self.decoder(h, cond))
```

### 9.2 Classifier-free guidance（CFG）

只加条件，模型生成的图往往"像但不够像"。CFG 在采样时把**有条件预测**和**无条件预测**的差放大：

$$
\hat\epsilon_{\text{cfg}} = \epsilon_\theta(x_t, t, \varnothing) + w\,\big(\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing)\big)
$$

- $$w = 0$$：无条件生成；$$w = 1$$：普通的条件生成；$$w > 1$$：更强调类别特征，但太大会降低多样性、颜色过饱和。

一个模型要同时会"有条件"和"无条件"两种预测，所以**训练时**以一定概率（常用 10%）把真实标签换成 $$\varnothing$$；**采样时**每一步算两次，再按上式组合，替换掉 `p_sample` / `ddim_step` 里的 `pred_noise`：

```python
def drop_labels_for_cfg(labels, null_class_id, drop_prob=0.1):
    # 训练用：每个样本以 drop_prob 的概率把标签换成 ∅
    drop = torch.rand(labels.shape, device=labels.device) < drop_prob
    return torch.where(drop, torch.full_like(labels, null_class_id), labels)

# 训练时：pred_noise = model(xt, t, drop_labels_for_cfg(labels, model.null_class_id))


def predict_noise_cfg(model, x_t, t, labels, w):
    # 采样用：ε_cfg = ε(∅) + w · (ε(y) − ε(∅))
    eps_uncond = model(x_t, t, torch.full_like(labels, model.null_class_id))
    eps_cond = model(x_t, t, labels)
    return eps_uncond + w * (eps_cond - eps_uncond)
```

> **易错点**
>
> 1. **embedding 表要有 $$K+1$$ 行。** 空类别占一个独立的下标 $$K$$，`nn.Embedding(num_classes, …)` 少一行时，传入 $$K$$ 会直接越界报错。
> 2. **训练时必须随机丢标签。** 否则模型从没见过 $$\varnothing$$，无条件预测是随机的，CFG 的差值也就没有意义。
> 3. **采样每步要两次前向。** 速度减半。实际实现常把有条件和无条件拼成一个 2B 的 batch 一次算完，结果相同。
> 4. **$$w$$ 不是越大越好。** 本例里 $$w = 3$$ 左右已经足够，太大时图像对比度过强、同一类的样本越来越像。

### 9.3 结果

每行一个类别（T 恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴），左边 4 列 $$w = 1$$，右边 4 列 $$w = 3$$，用 DDIM 50 步生成：

![CFG 按类别生成](/deep-learning/assets/ddpm-from-scratch/samples_cfg.png)

左右两边每个位置用的是同一个初始噪声，所以可以逐格对比。$$w = 1$$ 时每行已经大致是对应的类别，但轮廓松散；$$w = 3$$ 时类别特征清楚得多，最明显的是裤子（第 2 行）、凉鞋（第 6 行）、运动鞋（第 8 行）和短靴（第 10 行）。代价是同一行的样本更相似、多样性下降。

## 10. 网络到底该预测什么：ε、x₀ 还是 v

### 10.1 三种预测目标

前面一直让网络预测噪声 $$\epsilon$$。但 U-Net 只是一个"图像进、图像出"的结构，它也可以被训练成预测别的量。为了简洁，记

$$
a_t = \sqrt{\bar\alpha_t},\qquad s_t = \sqrt{1-\bar\alpha_t},\qquad x_t = a_t x_0 + s_t \epsilon,\qquad a_t^2 + s_t^2 = 1
$$

| 预测目标 | 训练时的 target | 从网络输出恢复 $$\hat x_0$$ |
|---|---|---|
| 噪声 $$\epsilon$$（DDPM） | $$\epsilon$$ | $$\hat x_0 = (x_t - s_t\hat\epsilon)/a_t$$ |
| 干净图像 $$x_0$$ | $$x_0$$ | 输出本身 |
| 速度 $$v$$ | $$v_t = a_t\epsilon - s_t x_0$$ | $$\hat x_0 = a_t x_t - s_t \hat v$$ |

$$v$$ 的来历：因为 $$a_t^2 + s_t^2 = 1$$，可以令 $$a_t = \cos\phi_t$$、$$s_t = \sin\phi_t$$，前向过程就是 $$x_0$$ 和 $$\epsilon$$ 之间的"旋转"：$$x_t = \cos\phi_t\,x_0 + \sin\phi_t\,\epsilon$$。对角度求导，得到旋转轨迹的切线方向：

$$
v_t = \frac{\partial x_t}{\partial \phi_t} = -\sin\phi_t\,x_0 + \cos\phi_t\,\epsilon = a_t\epsilon - s_t x_0
$$

不管预测哪一种，只要先换算成 $$\hat x_0$$，就可以用 7.1 的真实后验均值 $$\tilde\mu_t(x_t, \hat x_0)$$ 走反向一步，采样器的后半段完全共用。

```python
def training_target(x0, noise, t, sched, prediction_type):
    a_t = sched["sqrt_alpha_bars"][t].view(-1, 1, 1, 1)               # a_t
    s_t = sched["sqrt_one_minus_alpha_bars"][t].view(-1, 1, 1, 1)     # s_t
    if prediction_type == "epsilon":
        return noise                                                  # ε
    if prediction_type == "x0":
        return x0                                                     # x0
    if prediction_type == "v":
        return a_t * noise - s_t * x0                                 # v = a_t ε − s_t x0
    raise ValueError(prediction_type)


def output_to_x0(x_t, out, t, sched, prediction_type):
    a_t = sched["sqrt_alpha_bars"][t].view(-1, 1, 1, 1)
    s_t = sched["sqrt_one_minus_alpha_bars"][t].view(-1, 1, 1, 1)
    if prediction_type == "epsilon":
        return (x_t - s_t * out) / a_t                                # x̂0 = (x_t − s_t ε̂) / a_t
    if prediction_type == "x0":
        return out
    if prediction_type == "v":
        return a_t * x_t - s_t * out                                  # x̂0 = a_t x_t − s_t v̂
    raise ValueError(prediction_type)

# 训练：loss = F.mse_loss(model(xt, t), training_target(x0, noise, t, sched, prediction_type))
# 采样：x0_hat = output_to_x0(...).clamp(-1, 1)，再代入 7.1 的 μ̃_t(x_t, x0_hat)
```

### 10.2 信噪比（SNR）和误差放大

衡量一个时间步"有多吵"的量是信噪比：

$$
\operatorname{SNR}(t) = \frac{a_t^2}{s_t^2} = \frac{\bar\alpha_t}{1-\bar\alpha_t}
$$

$$t$$ 小时 SNR 高（$$x_t \approx x_0$$），$$t$$ 大时 SNR 趋于 0（$$x_t \approx \epsilon$$）。不同的预测目标，网络输出的一点误差传到 $$\hat x_0$$ 上会被放大不同的倍数：

$$
\hat\epsilon = \epsilon + \delta \ \Rightarrow\ \hat x_0 - x_0 = -\frac{s_t}{a_t}\,\delta = -\frac{\delta}{\sqrt{\operatorname{SNR}(t)}},
\qquad
\hat v = v + \delta \ \Rightarrow\ \hat x_0 - x_0 = -s_t\,\delta
$$

- **预测 $$\epsilon$$**：高噪声端 $$a_t \to 0$$，$$\hat x_0$$ 的误差被放大 $$1/a_t$$ 倍。本例 $$T=300$$ 时 $$a_T \approx 0.22$$，放大约 4.5 倍；$$T=1000$$ 时 $$a_T \approx 0.006$$，放大约 160 倍。
- **预测 $$x_0$$**：没有放大，但高噪声端输入里几乎没有 $$x_0$$ 的信息，这个任务本身很难。
- **预测 $$v$$**：系数 $$s_t \le 1$$，误差永远不会被放大。低噪声端 $$v \approx \epsilon$$，高噪声端 $$v \approx -x_0$$，在两者之间平滑过渡。

> **易错点**
>
> 1. **训练目标和采样器的解释必须一致。** 用 $$v$$ 训练的模型，采样时如果按 $$\epsilon$$ 来解释输出，结果完全错误，但不会报错。加载别人的 checkpoint 时要确认它的预测类型。
> 2. **预测 $$\epsilon$$ 时换算 $$\hat x_0$$ 要除以 $$a_t$$。** 高噪声端 $$a_t$$ 很小，$$\hat x_0$$ 可能远远超出 $$[-1, 1]$$，所以很多实现会 `clamp(-1, 1)` 后再算后验均值。
> 3. **怎么选？** 从零学习、小数据集、复现经典论文，用 $$\epsilon$$ 就够了，资料最多也最稳定；少步采样、蒸馏、高分辨率或视频生成里 $$v$$ 更常见（例如 progressive distillation、Imagen Video）；$$x_0$$ 一般配合专门的 loss 加权使用。

## 11. 自检

写完上面的代码，用几行断言检查形状和公式。这些检查不需要下载数据，一秒内跑完，以后改代码后都可以再跑一遍：

```python
T = 300
sched = make_beta_schedule(T)

# 1. 每一项都是 [T]，ᾱ_t 单调下降
assert all(v.shape == (T,) for v in sched.values())
assert torch.all(sched["alpha_bars"][1:] < sched["alpha_bars"][:-1])

# 2. (√ᾱ_t)² + (√(1-ᾱ_t))² = 1
total = sched["sqrt_alpha_bars"] ** 2 + sched["sqrt_one_minus_alpha_bars"] ** 2
assert torch.allclose(total, torch.ones(T), atol=1e-6)

# 3. β̃_1 = 0，且 β̃_t ≤ β_t
assert sched["posterior_variance"][0] == 0
assert torch.all(sched["posterior_variance"] <= sched["betas"])

# 4. 灰度和 RGB 的 shape 都对，固定噪声时和公式逐元素一致
for shape in [(4, 1, 28, 28), (4, 3, 32, 32)]:
    x0 = torch.randn(shape)
    t = torch.randint(0, T, (shape[0],))
    noise = torch.randn_like(x0)
    xt, eps = q_sample(x0, t, sched["sqrt_alpha_bars"], sched["sqrt_one_minus_alpha_bars"], noise)
    assert xt.shape == x0.shape and eps is noise
    for i in range(shape[0]):
        expected = sched["sqrt_alpha_bars"][t[i]] * x0[i] + sched["sqrt_one_minus_alpha_bars"][t[i]] * noise[i]
        assert torch.allclose(xt[i], expected)

# 5. 归一化的 transform 可以被 pickle（多进程 DataLoader 需要）
import pickle
pickle.dumps(build_transform(28))

# 6. 时间步编码：shape 是 [B, d]，sin/cos 的值都在 [-1, 1]
emb = sinusoidal_timestep_embedding(torch.tensor([0, 10, 299]), 128)
assert emb.shape == (3, 128) and emb.abs().max() <= 1

# 7. 网络输出和输入同形状（带不带 attention 都一样）
for use_attention in [False, True]:
    net = TinyUNet(in_channels=1, base_channels=32, use_attention=use_attention)
    x = torch.randn(4, 1, 28, 28)
    assert net(x, torch.randint(0, T, (4,))).shape == x.shape

# 8. 三种预测目标都能精确还原 x0
x0, noise = torch.randn(4, 1, 28, 28), torch.randn(4, 1, 28, 28)
t = torch.randint(0, T, (4,))
xt, _ = q_sample(x0, t, sched["sqrt_alpha_bars"], sched["sqrt_one_minus_alpha_bars"], noise)
for kind in ["epsilon", "x0", "v"]:
    target = training_target(x0, noise, t, sched, kind)
    assert torch.allclose(output_to_x0(xt, target, t, sched, kind), x0, atol=1e-4)

# 9. 未训练的网络也能跑通采样，输出 shape 正确（只检查流程，不看图）
with torch.no_grad():
    assert sample_ddim(net, (2, 1, 28, 28), sched, "cpu", num_steps=10).shape == (2, 1, 28, 28)
```

## 12. 要点回顾

- **为什么可以一步从 $$x_0$$ 得到 $$x_t$$？** 每步都是线性高斯，高斯的线性组合仍是高斯，均值系数累乘成 $$\sqrt{\bar\alpha_t}$$，方差合并成 $$1-\bar\alpha_t$$（第 4.1 节的两步展开）。
- **前向过程有可学习参数吗？** 没有，$$\beta_t$$ 是预先定好的。学习只发生在反向过程，也就是预测噪声的网络。
- **为什么数据要归一化到 $$[-1, 1]$$？** 和标准高斯噪声的尺度匹配，保证 $$x_t$$ 方差稳定在 1 左右；也让 $$x_T$$ 接近 $$\mathcal{N}(0, I)$$，采样才有合理的起点。
- **训练时 $$t$$ 为什么每张图各抽一个？** 一次前向就覆盖多个噪声水平，梯度方差更小。实现上要注意 `[B] → [B, 1, 1, 1]` 的广播。
- **$$T$$ 取多大有什么影响？** $$T$$ 太小，最后一步残留信号多，采样起点和训练分布不一致；$$T$$ 大则采样步数多、慢（DDIM 等方法就是为了解决这个问题）。
- **时间步怎么告诉网络？** 先用正弦编码把整数 $$t$$ 变成 $$d$$ 维向量，再经过 MLP，在每个残差块里投影成 `[B, C, 1, 1]` 加到特征图上。$$t$$ 不能直接当一个标量通道拼进去，否则网络很难区分相邻的时间步。
- **训练目标为什么这么简单？** ELBO 化简后，每一步都变成预测噪声的 MSE。训练时不用走马尔可夫链，任意 $$t$$ 都能一步造出 $$x_t$$。
- **采样为什么要一步一步走？** 网络只学会了"在噪声水平 $$t$$ 去掉一点噪声"。DDPM 每步还要加回一点随机噪声（最后一步除外），所以必须走完 $$T$$ 步。
- **DDIM 为什么能跳步？** 它和 DDPM 共用同一个训练好的网络，只是换了一个非马尔可夫的反向过程：先估计 $$\hat x_0$$，再直接跳到任意更早的时间步。$$\eta = 0$$ 时完全确定，同一个初始噪声总是得到同一张图。
- **CFG 的代价是什么？** 训练时只需以 10% 的概率把标签换成空类别；采样时每步要算两次网络（有条件、无条件），速度减半，换来更强的类别一致性。
- **$$\epsilon$$、$$x_0$$、$$v$$ 选哪个？** 三者可以互相换算，采样器的后半段共用。差别在于网络误差传到 $$\hat x_0$$ 时放大多少：$$\epsilon$$ 在高噪声端放大 $$1/\sqrt{\bar\alpha_t}$$ 倍，$$v$$ 永远不放大。
