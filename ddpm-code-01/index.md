---
date: 2026-11-26 16:30:00 -0800
keywords: >-
  DDPM from scratch, PyTorch, data normalization, linear beta schedule, alpha_bar, forward process q_sample, tensor shapes.
summary: >-
  DDPM 从零实现系列第一篇：数据归一化、线性 beta schedule、前向加噪。每一部分先讲清楚它是什么和公式，再给出逐行对应公式的 PyTorch 实现和 tensor shape，最后总结广播、下标、device、pickle 等实现上的易错点。不看源码，只看这一篇也能自己写出来。
layout: note
title: "DDPM 代码解读（一）：数据、噪声调度与前向加噪"
description: "DDPM from scratch, part 1: data normalization, the linear beta schedule, and the forward noising process, formula by formula with PyTorch code and tensor shapes."
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

这是"从零实现 DDPM"系列的第一篇。目标是：**只看这个系列，就能从公式出发，自己把 DDPM 一行行写出来**。每一部分都按同一个顺序讲：

1. 它是什么、为什么需要它，以及公式；
2. 代码实现，变量名和公式符号一一对应；
3. 易错点：公式写对了，代码也可能出错的地方。

完整的公式推导（ELBO、为什么预测噪声等）见之前的笔记 [Diffusion Models 扩散模型学习](/deep-learning/diffusion-study/)，这里只推导写代码需要的部分。

## 0. 系列路线和本篇范围

DDPM 的完整流程可以拆成五步，每一步都能单独理解、单独检查：

1. **数据与前向过程**（本篇）：图像怎么归一化，噪声怎么一步步加上去。
2. **噪声预测网络**：时间步编码 + 一个小 U-Net，输入带噪图像和时间步，输出预测的噪声。
3. **训练**：把前向过程和网络连起来，用 MSE loss 训练。
4. **采样**：从纯噪声开始，一步步反向去噪生成图像。
5. **扩展**：DDIM 加速采样、classifier-free guidance。

本篇**还不涉及任何神经网络**。前向加噪是一个固定的、没有参数的过程，但它决定了后面训练时"模型的输入是什么、要预测的目标是什么"。

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
| $$\tilde\beta_t$$ | 反向过程的方差（第 4 篇用） | `posterior_variance[t-1]` | `[T]` |
| $$\epsilon_\theta(x_t, t)$$ | 网络预测的噪声（第 2 篇） | `pred_noise` | `[B, C, H, W]` |

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
<figcaption>本篇三个部分之间的数据流和 tensor shape。框里用的是代码变量名，对应的公式见下。</figcaption>
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

第 4 篇反向采样要用的两个量也在这里一次算好。反向一步的均值和方差是：

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

![噪声调度曲线](/deep-learning/assets/ddpm-code-01/schedule.png)

右图是本篇最重要的直觉：随着 $$t$$ 增大，图像的权重 $$\sqrt{\bar\alpha_t}$$ 下降，噪声的权重 $$\sqrt{1-\bar\alpha_t}$$ 上升，两条曲线在 $$t \approx 140$$ 附近交叉。

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

这一行公式就是本篇代码的核心，它带来两个直接后果：

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

训练时的用法（第 3 篇会完整写出来）：

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
> 5. **允许从外面传入 `noise`。** 检查代码时传一份固定的噪声，就能把结果和公式逐元素对比（见第 6 节）。

### 4.3 结果

FashionMNIST 上的前向加噪，每行一张图，第一列是 $$x_0$$，后面依次是 $$t = 25, 50, 100, 200, 300$$：

![FashionMNIST 前向加噪](/deep-learning/assets/ddpm-code-01/forward_fashion_mnist.png)

几个观察：

1. $$t \le 50$$ 时图像几乎不变，噪声主要在中后段才快速占上风，和第 3 节的曲线一致。
2. **$$T = 300$$ 时最后一步并不是纯噪声**：$$\sqrt{\bar\alpha_{300}} \approx 0.22$$，仔细看最后一列仍能隐约看到轮廓。DDPM 原论文用 $$T = 1000$$、同样的 $$\beta$$ 范围，$$\sqrt{\bar\alpha_{1000}} \approx 0.006$$，才接近纯噪声。这里选 300 是为了在小数据集上训练、采样更快；代价是采样起点 $$x_T \sim \mathcal{N}(0, I)$$ 和真实的 $$x_T$$ 分布有偏差。第 4 篇可以对比 $$T = 300$$ 和 $$T = 1000$$ 的生成效果，或者换成 cosine schedule。

## 5. 可视化技巧：一次算出"每张图 × 每个时间步"

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

## 6. 自检

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
```

## 7. 要点回顾

- **为什么可以一步从 $$x_0$$ 得到 $$x_t$$？** 每步都是线性高斯，高斯的线性组合仍是高斯，均值系数累乘成 $$\sqrt{\bar\alpha_t}$$，方差合并成 $$1-\bar\alpha_t$$（第 4.1 节的两步展开）。
- **前向过程有可学习参数吗？** 没有，$$\beta_t$$ 是预先定好的。学习只发生在反向过程，也就是预测噪声的网络。
- **为什么数据要归一化到 $$[-1, 1]$$？** 和标准高斯噪声的尺度匹配，保证 $$x_t$$ 方差稳定在 1 左右；也让 $$x_T$$ 接近 $$\mathcal{N}(0, I)$$，采样才有合理的起点。
- **训练时 $$t$$ 为什么每张图各抽一个？** 一次前向就覆盖多个噪声水平，梯度方差更小。实现上要注意 `[B] → [B, 1, 1, 1]` 的广播。
- **$$T$$ 取多大有什么影响？** $$T$$ 太小，最后一步残留信号多，采样起点和训练分布不一致；$$T$$ 大则采样步数多、慢（DDIM 等方法就是为了解决这个问题）。

## 下一步

第 2 篇：噪声预测网络。先写正弦时间步编码（让网络知道当前是第几步），再写一个小 U-Net，输入 $$(x_t, t)$$，输出和 $$x_t$$ 同 shape 的噪声预测，并在一个小 batch 上确认 MSE loss 能下降。
