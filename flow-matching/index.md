---
date: 2026-11-28 01:00:00 -0800
tags: [Flow Matching, Sampling]
keywords: >-
  Flow Matching, Rectified Flow, Reflow, conditional flow matching, velocity field, probability path, continuity equation, ODE, Euler, Heun, DDIM, v-prediction, logit-normal, timestep shift, Stable Diffusion 3, FLUX, PyTorch.
summary: >-
  Flow Matching 与 Rectified Flow：为什么只要回归一个"速度" ε − x₀ 就能学会把噪声搬成数据。从速度场、概率路径和连续性方程讲起，推导条件 Flow Matching 损失为什么和真正的目标等价，Rectified Flow 的直线路径与 Reflow 拉直，和 DDPM / DDIM / v-prediction 的精确对应（线性路径上 Euler 一步就是 DDIM 一步），再到 SD3、FLUX 实际用的 logit-normal 时间采样和分辨率相关的时间偏移。附二维 Reflow 实验和 FashionMNIST 上的完整实现。
layout: note
title: "Flow Matching 与 Rectified Flow：从速度场到 SD3"
description: "Flow matching and rectified flow from scratch: velocity fields, the conditional flow matching loss, straight paths and reflow, the exact link to DDPM/DDIM, SD3/FLUX training details, and a PyTorch implementation with experiments."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
</style>

# Flow Matching 与 Rectified Flow：从速度场到 SD3

[DDPM](/deep-learning/ddpm-from-scratch/) 把生成拆成几百个"去掉一点噪声"的小步；[DDIM](/deep-learning/ddim/) 发现确定性的采样其实是在解一个 ODE，于是可以用几十步走完。**Flow Matching** 把这个观察推到底：既然最终用的是 ODE，那就**直接学这个 ODE 的速度场**，不再经过"加噪的马尔可夫链"和"预测噪声"这层包装。

它最常用的形式（Rectified Flow）只有两行：

$$
x_t = (1-t)\,x_0 + t\,\epsilon,\qquad \mathcal{L} = \mathbb{E}\,\big\lVert v_\theta(x_t, t) - (\epsilon - x_0)\big\rVert^2
$$

即在数据和噪声之间连一条直线，让网络预测沿直线运动的速度。Stable Diffusion 3、FLUX，以及 Movie Gen、Wan、HunyuanVideo 等视频模型都用这个目标训练。

本文的记号和前面的扩散系列一致：**$$t = 0$$ 是数据，$$t = 1$$ 是噪声**，生成时从 $$t = 1$$ 走到 $$t = 0$$。这和 SD3 论文相同，但和 Flow Matching、Rectified Flow 两篇原论文的方向相反，第 1 节给出对照表。

## 参考资料

本文主要参考 Flow Matching 原论文 [1] 和 Rectified Flow 原论文 [2]，SD3 的训练设定来自 [3]；想系统学习可以看 Meta 的教程和代码库 [4]，和扩散模型的对应关系推荐看博客 [5]。正文中用 [n] 标注具体出处。

- <a id="ref-1"></a>**[1] Flow Matching 主论文**：Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling*. ICLR 2023. [arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)。条件 Flow Matching 损失和它与边缘目标等价的证明（第 3 节）出自这篇。
- <a id="ref-2"></a>**[2] Rectified Flow 主论文**：Liu, Gong, Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR 2023. [arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)。直线插值、Reflow 及其性质（第 4 节）出自这篇。
- <a id="ref-3"></a>**[3] SD3**：Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. ICML 2024. [arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)。logit-normal 时间采样、分辨率相关的时间偏移（第 8 节）出自这篇。
- <a id="ref-4"></a>**[4] 最系统的教程和代码**：Lipman et al. *Flow Matching Guide and Code*. 2024. [arxiv.org/abs/2412.06264](https://arxiv.org/abs/2412.06264)，代码库 [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)。定义、证明和各种变体都有，推导细节不清楚时查这篇。
- <a id="ref-5"></a>**[5] 讲清和扩散关系的博客**：Gao, Hoogeboom, Heek, Bortoli, Murphy, Salimans. *Diffusion Meets Flow Matching: Two Sides of the Same Coin*. 2024. [diffusionflow.github.io](https://diffusionflow.github.io/)。第 5 节"Euler 一步等于 DDIM 一步"、损失加权的换算都可以在这里找到更多讨论。
- <a id="ref-6"></a>**[6] Reflow 用在大模型上**：Liu et al. *InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation*. ICLR 2024. [arxiv.org/abs/2309.06380](https://arxiv.org/abs/2309.06380)。对 Stable Diffusion 做 Reflow 再蒸馏，得到一步生成的模型（第 4.4 节）。

## 背景

本节介绍后文用到的基础知识和基本概念。熟悉 DDIM 的 ODE 解释和速度场概念的读者可以直接阅读第 0 节。

### 前置知识

- **DDPM 的前向加噪** $$x_t = a_t x_0 + s_t \epsilon$$（$$a_t = \sqrt{\bar\alpha_t}$$，$$s_t = \sqrt{1-\bar\alpha_t}$$），以及 ε / x₀ / v 三种预测目标：见 [DDPM 从零实现](/deep-learning/ddpm-from-scratch/) 第 4、10 节。
- **ODE 和 Euler 法**，以及"确定性 DDIM 是一个 ODE 的 Euler 法"：见 [DDIM](/deep-learning/ddim/) 背景和第 5 节。
- **生成模型要解决的问题**（从一个简单分布采样，变成从数据分布采样）：见 [VAE](/deep-learning/generative-models-1-vae/)、[GAN](/deep-learning/generative-models-2-gan/)。本文的二维实验和 GAN 篇用的是同一个"8 个高斯"的数据，可以对照着看。

### 速度场和流

想象平面上每个位置都画着一个小箭头，表示"一个粒子到了这里，下一瞬间往哪个方向、以多快的速度走"。这些箭头合起来叫**速度场**（vector field）$$u_t(x)$$，它可以随时间 $$t$$ 变化。把一个粒子放在某处，让它一直顺着箭头走，走出的轨迹满足 ODE

$$
\frac{dx_t}{dt} = u_t(x_t)
$$

把**一大群**粒子同时放进去，它们整体的分布也会随之变化。Flow Matching 要做的，就是找一个速度场，使得：把一群服从标准高斯的粒子放在 $$t = 1$$，顺着箭头反向走到 $$t = 0$$，它们恰好变成服从数据分布的一群点。

<figure class="ddpm-fig">
<svg viewBox="0 0 760 280" role="img" aria-label="速度场把一团高斯噪声搬运成两团数据">
  <defs><marker id="fa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#98a2b3"/></marker>
  <marker id="fb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#e0662b"/></marker></defs>
  <circle cx="620" cy="140" r="70" fill="#eef4ff" stroke="#2a6fdb"/>
  <text x="620" y="236" text-anchor="middle">t = 1：标准高斯噪声 ε</text>
  <ellipse cx="130" cy="78" rx="42" ry="26" fill="#fff4ec" stroke="#e0662b"/>
  <ellipse cx="130" cy="202" rx="42" ry="26" fill="#fff4ec" stroke="#e0662b"/>
  <text x="130" y="258" text-anchor="middle">t = 0：数据 x₀（两团）</text>
  <g stroke="#98a2b3" stroke-width="1.2" marker-end="url(#fa)">
    <line x1="520" y1="95" x2="485" y2="88"/><line x1="520" y1="140" x2="485" y2="140"/><line x1="520" y1="185" x2="485" y2="192"/>
    <line x1="420" y1="80" x2="385" y2="72"/><line x1="420" y1="125" x2="385" y2="112"/><line x1="420" y1="155" x2="385" y2="168"/><line x1="420" y1="200" x2="385" y2="208"/>
    <line x1="320" y1="70" x2="285" y2="70"/><line x1="320" y1="110" x2="285" y2="96"/><line x1="320" y1="170" x2="285" y2="184"/><line x1="320" y1="210" x2="285" y2="210"/>
    <line x1="230" y1="75" x2="195" y2="76"/><line x1="230" y1="205" x2="195" y2="203"/>
  </g>
  <path d="M600,120 C500,110 420,100 300,80 C240,72 200,76 160,78" fill="none" stroke="#e0662b" stroke-width="2.2" marker-end="url(#fb)"/>
  <path d="M640,165 C540,180 430,195 300,203 C240,206 200,204 160,202" fill="none" stroke="#e0662b" stroke-width="2.2" marker-end="url(#fb)"/>
  <circle cx="600" cy="120" r="4" fill="#2a6fdb"/><circle cx="640" cy="165" r="4" fill="#2a6fdb"/>
  <text x="380" y="30" text-anchor="middle" class="sm">灰色箭头：速度场 u_t(x)，每个位置、每个时刻一个方向</text>
  <text x="380" y="50" text-anchor="middle" class="sm">橙色曲线：单个粒子顺着速度场走出的轨迹（从 t = 1 走到 t = 0）</text>
</svg>
<figcaption>Flow Matching 学的是灰色箭头。生成时从右边的高斯里抽一个点，顺着箭头走到左边，就得到一个数据样本。</figcaption>
</figure>

和 DDPM 对比：DDPM 学的是"每一步该减掉多少噪声"，采样器再把它换算成移动方向；Flow Matching 直接学移动方向本身，采样器就是普通的 ODE 解法。

### 条件期望是 MSE 的最优解

本文反复用到一个事实：用 MSE 训练一个网络 $$f(x)$$ 去拟合随机目标 $$y$$，最优解是**条件期望**：

$$
\arg\min_f\ \mathbb{E}\,\lVert f(x) - y \rVert^2 = \mathbb{E}[\,y \mid x\,]
$$

同一个输入 $$x$$ 对应多个可能的目标时，网络学到的是它们的平均。DDIM 篇 5.1 节"一步得到的是一张模糊的平均图"就是这个原因，第 3、4 节会再用到它。

### 数值示例：一个像素

取 SD3 的直线路径 $$x_t = (1-t)x_0 + t\epsilon$$，干净像素 $$x_0 = 0.6$$，噪声 $$\epsilon = -1.2$$（和 DDPM 篇背景里的数值示例相同）：

1. **构造训练样本。** 随机抽到 $$t = 0.5$$，带噪像素 $$x_{0.5} = 0.5 \times 0.6 + 0.5 \times (-1.2) = -0.3$$。
2. **训练目标。** 这条直线的速度是 $$\epsilon - x_0 = -1.2 - 0.6 = -1.8$$，和 $$t$$ 无关。网络只看到 $$x_{0.5} = -0.3$$ 和 $$t = 0.5$$，假设它预测 $$\hat v = -1.7$$，损失就是 $$(-1.7 + 1.8)^2 = 0.01$$。
3. **往回走一步（采样）。** 从 $$t = 0.5$$ 走到 $$t = 0.4$$，Euler 法：$$x_{0.4} = x_{0.5} + (0.4 - 0.5)\times\hat v = -0.3 + 0.17 = -0.13$$。
4. **一步走到底。** 如果直接从 $$t = 0.5$$ 走到 $$t = 0$$：$$\hat x_0 = -0.3 + 0.5 \times 1.7 = 0.55$$，离真实的 0.6 不远。路径是直线时，速度估得准，一步就能走到终点。

### 一个可以运行的最小例子

数据是一维高斯 $$\mathcal{N}(2, 0.5^2)$$ 时，边缘速度场（第 3.2 节）可以精确写出来。下面直接用它把标准高斯搬运成数据分布，不训练任何网络：

```python
import torch

mu, sigma = 2.0, 0.5                     # 数据分布 p_data = N(2, 0.5²)

def u(x, t):
    """边缘速度场 u_t(x) = E[ε − x0 | x_t = x]，x_t = (1−t) x0 + t ε。一维高斯可以精确算出。"""
    var = (1 - t) ** 2 * sigma ** 2 + t ** 2          # Var(x_t)
    cov = t - (1 - t) * sigma ** 2                    # Cov(ε − x0, x_t)
    return -mu + cov / var * (x - (1 - t) * mu)

x = torch.randn(100_000)                 # t = 1：标准高斯噪声
N = 50
for i in range(N):                       # Euler 法，从 t = 1 走到 t = 0
    t = 1 - i / N
    x = x + (-1 / N) * u(x, t)           # x_{t−Δt} = x_t − Δt · u_t(x_t)
print(x.mean(), x.std())                 # 约 2.00、0.48：噪声被搬成了数据分布
```

标准差是 0.48 而不是 0.5，差的部分是 Euler 法的离散误差：$$N = 200$$ 时变成 0.496。真实问题里 $$u_t$$ 写不出来，要用神经网络 $$v_\theta$$ 去拟合，这就是 Flow Matching。

## 0. 全文路线

| 节 | 内容 | 关键结论 |
|---|---|---|
| 1 | 符号约定 | $$t=0$$ 数据、$$t=1$$ 噪声；和原论文方向相反 |
| 2 | 用 ODE 生成数据 | 速度场、概率路径、连续性方程 |
| 3 | Flow Matching | 边缘速度场写不出来，但可以回归条件速度场，梯度相同 |
| 4 | Rectified Flow | 直线路径；边缘轨迹仍是弯的；Reflow 把它拉直 |
| 5 | 和 DDPM / DDIM 的关系 | 同属高斯路径；预测目标、损失加权可以互换；Euler 一步就是 DDIM 一步 |
| 6 | 实现 | 训练损失、时间采样、Euler / Heun 采样、CFG |
| 7 | FashionMNIST 实验 | 步数、采样器阶数 |
| 8 | SD3 / FLUX 的做法 | logit-normal 时间采样、分辨率相关的时间偏移、引导蒸馏 |
| 9–10 | 自检、要点回顾 | |

## 1. 符号约定

| 公式符号 | 含义 | 代码变量 | shape |
|---|---|---|---|
| $$t \in [0, 1]$$ | 连续时间，$$t=0$$ 数据，$$t=1$$ 噪声 | `t` | `[B]` |
| $$x_0 \sim p_{\text{data}}$$ | 干净数据 | `x0` | `[B, C, H, W]` |
| $$\epsilon \sim \mathcal{N}(0, I)$$ | 标准高斯噪声，也就是 $$t=1$$ 的样本 | `eps` | `[B, C, H, W]` |
| $$x_t$$ | 时刻 $$t$$ 的中间状态 | `x_t` | `[B, C, H, W]` |
| $$a_t,\ s_t$$ | 一般高斯路径 $$x_t = a_t x_0 + s_t \epsilon$$ 的两个系数 | `a_t`, `s_t` | `[B, 1, 1, 1]` |
| $$p_t$$ | $$x_t$$ 的边缘分布（概率路径），$$p_0 = p_{\text{data}}$$，$$p_1 = \mathcal{N}(0, I)$$ | 无 | |
| $$u_t(x \mid x_0, \epsilon)$$ | 条件速度：单条路径 $$x_t$$ 对 $$t$$ 的导数 | `u` | `[B, C, H, W]` |
| $$u_t(x)$$ | 边缘速度场：所有经过 $$x$$ 的路径速度的平均 | 无（写不出来） | |
| $$v_\theta(x_t, t)$$ | 网络预测的速度 | `model(x_t, t)`、`v` | `[B, C, H, W]` |
| $$N$$ | 采样步数 | `num_steps` | int |

**两个"速度"不要混淆。** DDPM 篇第 10 节的 v-prediction 目标是 $$a_t\epsilon - s_t x_0$$，它是对"角度" $$\phi$$ 求导得到的；本文的 $$u_t$$ 是对时间 $$t$$ 求导得到的。在直线路径上 $$u = \epsilon - x_0$$，两者不是同一个量，第 5.2 节给出换算。

**和原论文的记号对照。** 不同论文对时间方向和变量名的约定不同，本文统一成和扩散系列、SD3 一致的写法：

| | 数据在 | 噪声在 | 直线路径 | 速度目标 |
|---|---|---|---|---|
| 本文、SD3 [[3]](#ref-3) | $$t = 0$$，$$x_0$$ | $$t = 1$$，$$\epsilon$$ | $$(1-t)x_0 + t\epsilon$$ | $$\epsilon - x_0$$ |
| Flow Matching [[1]](#ref-1) | $$t = 1$$，$$x_1$$ | $$t = 0$$，$$x_0$$ | $$t x_1 + (1-t)x_0$$（取 $$\sigma_{\min}=0$$） | $$x_1 - x_0$$ |
| Rectified Flow [[2]](#ref-2) | $$t = 1$$，$$X_1$$ | $$t = 0$$，$$X_0$$ | $$tX_1 + (1-t)X_0$$ | $$X_1 - X_0$$ |

原论文里 $$x_0$$ 是**噪声**、$$x_1$$ 是数据，正好和 DDPM 相反；对照原文时，把 $$t$$ 换成 $$1-t$$、速度取负号即可。

## 2. 用 ODE 生成数据

### 2.1 流和概率路径

给定速度场 $$u_t(x)$$，从任意起点 $$x_1$$ 出发解 ODE $$dx_t/dt = u_t(x_t)$$，得到一条轨迹。把"起点 $$\mapsto$$ 时刻 $$t$$ 的位置"这个映射记作 $$\psi_t$$，叫做**流**（flow）。如果起点服从 $$p_1 = \mathcal{N}(0, I)$$，那么 $$\psi_t(x_1)$$ 服从某个分布 $$p_t$$，随 $$t$$ 连续变化，这一族分布叫做**概率路径**（probability path）。

生成模型的目标可以写成：**找一个速度场，使它产生的概率路径在 $$t = 0$$ 时等于 $$p_{\text{data}}$$。** 采样时从 $$\mathcal{N}(0, I)$$ 抽一个点，用 ODE 解法从 $$t = 1$$ 积分到 $$t = 0$$ 即可。

### 2.2 连续性方程

速度场和它产生的概率路径之间的关系由**连续性方程**给出：

$$
\frac{\partial p_t(x)}{\partial t} + \nabla \cdot \big(p_t(x)\, u_t(x)\big) = 0
\tag{1}
$$

它表达的是"概率质量守恒"：某处的密度变化，等于流入和流出这一点的概率之差。$$p_t u_t$$ 是概率的流量，$$\nabla\cdot$$（散度）衡量净流出。只要 $$u_t$$ 和 $$p_t$$ 满足式 (1)，就说 **$$u_t$$ 产生 $$p_t$$**。第 3.2 节靠这个方程证明边缘速度场是正确的。

### 2.3 难点在哪里

用 ODE 做生成模型的想法比 Flow Matching 更早，叫 Continuous Normalizing Flow（CNF）。CNF 用最大似然训练：要算 $$\log p_0(x)$$，必须把 ODE 从头到尾解一遍，还要沿途积分速度场的散度。每个训练样本都要跑一次 ODE 求解器，训练极慢，很难扩展到图像。

Flow Matching 换了一个思路：**先人为指定概率路径 $$p_t$$，再直接回归产生它的速度场**，训练时完全不需要解 ODE。困难在于：路径上的边缘速度场 $$u_t(x)$$ 写不出来。第 3 节解决这个问题。

## 3. Flow Matching

### 3.1 先给每对样本设计一条路径

直接设计边缘的 $$p_t$$ 很难，但给**一对具体的样本** $$(x_0, \epsilon)$$ 设计一条路径很容易，比如 [[1]](#ref-1)

$$
x_t = a_t x_0 + s_t \epsilon,\qquad a_0 = 1,\ s_0 = 0,\qquad a_1 = 0,\ s_1 = 1
\tag{2}
$$

$$t = 0$$ 时是数据，$$t = 1$$ 时是噪声。这条路径（**条件路径**）的速度可以直接求导得到：

$$
u_t(x_t \mid x_0, \epsilon) = \frac{dx_t}{dt} = \dot a_t\, x_0 + \dot s_t\, \epsilon
\tag{3}
$$

$$\dot a_t$$、$$\dot s_t$$ 表示对 $$t$$ 的导数。对所有 $$(x_0, \epsilon)$$ 对取平均，就得到边缘概率路径：

$$
p_t(x) = \mathbb{E}_{x_0, \epsilon}\big[\, p_t(x \mid x_0, \epsilon)\,\big]
$$

这和 DDPM 的前向过程一样：给定 $$x_0$$，$$x_t$$ 是一个高斯；对 $$x_0$$ 取平均，就是 $$x_t$$ 的边缘分布。$$p_0 = p_{\text{data}}$$，$$p_1 = \mathcal{N}(0, I)$$，满足第 2 节的要求。

### 3.2 边缘速度场：条件速度的条件期望

产生边缘路径 $$p_t$$ 的速度场是

$$
u_t(x) = \mathbb{E}\big[\, u_t(x_t \mid x_0, \epsilon) \;\big|\; x_t = x \,\big]
= \mathbb{E}\big[\, \dot a_t x_0 + \dot s_t \epsilon \;\big|\; x_t = x \,\big]
\tag{4}
$$

即：所有恰好在时刻 $$t$$ 经过位置 $$x$$ 的条件路径，把它们的速度取平均。

**证明。** 每条条件路径的速度产生它自己的条件分布，满足连续性方程：

$$
\frac{\partial p_t(x \mid z)}{\partial t} + \nabla\cdot\big(p_t(x \mid z)\, u_t(x \mid z)\big) = 0,\qquad z = (x_0, \epsilon)
$$

对 $$z$$ 取期望，左边第一项变成 $$\partial p_t(x)/\partial t$$；第二项里，由贝叶斯公式 $$p_t(x \mid z)\,q(z) = p_t(x)\,q(z \mid x_t = x)$$：

$$
\mathbb{E}_z\big[p_t(x \mid z)\, u_t(x \mid z)\big] = p_t(x)\,\mathbb{E}\big[u_t(x \mid z) \mid x_t = x\big] = p_t(x)\, u_t(x)
$$

所以 $$\partial_t p_t + \nabla\cdot(p_t u_t) = 0$$，$$u_t$$ 产生 $$p_t$$。$$\square$$

背景里一维高斯的例子用的就是式 (4)：$$x_t$$ 和 $$\epsilon - x_0$$ 是联合高斯，条件期望有闭式解。一般的数据分布没有闭式解，因为要对所有可能经过 $$x$$ 的 $$x_0$$ 求平均，而 $$x_0$$ 的分布正是我们不知道的东西。

### 3.3 条件 Flow Matching 损失

想要的训练目标是让网络逼近边缘速度场（Flow Matching 损失）：

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t,\, x_t \sim p_t}\big\lVert v_\theta(x_t, t) - u_t(x_t)\big\rVert^2
$$

$$u_t(x_t)$$ 算不出来，这个损失没法直接用。Flow Matching 的关键结论是：把目标换成**条件速度**，得到的条件 Flow Matching（CFM）损失

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, x_0,\, \epsilon}\big\lVert v_\theta(x_t, t) - u_t(x_t \mid x_0, \epsilon)\big\rVert^2
\tag{5}
$$

和 $$\mathcal{L}_{\text{FM}}$$ **只差一个与 $$\theta$$ 无关的常数**，因此梯度相同、最优解相同 [[1]](#ref-1)。

**证明。** 记 $$v = v_\theta(x_t, t)$$，$$u_c = u_t(x_t \mid x_0, \epsilon)$$，$$u = u_t(x_t)$$。展开平方：

$$
\lVert v - u_c\rVert^2 = \lVert v\rVert^2 - 2\langle v, u_c\rangle + \lVert u_c\rVert^2
$$

先固定 $$x_t$$、对 $$(x_0, \epsilon)$$ 取条件期望。$$v$$ 只依赖 $$x_t$$，所以 $$\mathbb{E}[\langle v, u_c\rangle \mid x_t] = \langle v, \mathbb{E}[u_c \mid x_t]\rangle = \langle v, u\rangle$$（式 4）。于是

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}\big[\lVert v\rVert^2 - 2\langle v, u\rangle\big] + \mathbb{E}\lVert u_c\rVert^2
= \mathcal{L}_{\text{FM}} + \underbrace{\mathbb{E}\lVert u_c\rVert^2 - \mathbb{E}\lVert u\rVert^2}_{\text{与 }\theta\text{ 无关}}
\qquad\square
$$

这和背景里"MSE 的最优解是条件期望"是同一件事：用同一个输入 $$x_t$$ 去回归许多个不同的条件速度，网络学到的就是它们的平均，也就是式 (4) 的边缘速度场。

**这个常数不是 0。** 同一个 $$x_t$$ 可以由许多不同的 $$(x_0, \epsilon)$$ 产生，它们的条件速度各不相同，网络不可能同时拟合所有这些目标。所以 CFM 损失**不会降到 0**，训练曲线停在一个较高的值附近是正常的（第 7 节的损失曲线最后约 0.25），不能据此判断模型没训好。

### 3.4 常见的路径选择

式 (2) 里 $$a_t, s_t$$ 怎么选，决定了是哪一种方法。它们都属于**高斯路径**：给定 $$x_0$$，$$x_t \sim \mathcal{N}(a_t x_0, s_t^2 I)$$。

| 路径 | $$a_t$$ | $$s_t$$ | 条件速度 $$\dot a_t x_0 + \dot s_t \epsilon$$ | 出处 |
|---|---|---|---|---|
| 直线（Rectified Flow） | $$1 - t$$ | $$t$$ | $$\epsilon - x_0$$ | [[2]](#ref-2)、SD3 [[3]](#ref-3) |
| Flow Matching 的"最优传输"路径 | $$1 - t$$ | $$\sigma_{\min} + (1-\sigma_{\min})\,t$$ | $$(1-\sigma_{\min})\,\epsilon - x_0$$ | [[1]](#ref-1)，$$\sigma_{\min}$$ 如 $$10^{-4}$$ |
| 方差保持（DDPM 的连续版本） | $$\sqrt{\bar\alpha(t)}$$ | $$\sqrt{1-\bar\alpha(t)}$$ | 随调度而定，轨迹是圆弧 | DDPM、DDIM |
| 余弦 | $$\cos\frac{\pi t}{2}$$ | $$\sin\frac{\pi t}{2}$$ | $$\frac{\pi}{2}\big(\cos\tfrac{\pi t}{2}\,\epsilon - \sin\tfrac{\pi t}{2}\,x_0\big)$$ | 改进版 DDPM 的余弦调度 |

Flow Matching 论文 [[1]](#ref-1) 里的 $$\sigma_{\min}$$ 让 $$t = 0$$ 时保留一点点噪声，实践中几乎都直接取 0，就是第一行的直线。这几种路径的区别只是"在什么时刻加多少噪声"，第 5 节会看到它们可以互相换算。

> **易错点**
>
> 1. **网络学到的是边缘速度，不是条件速度。** 训练目标是 $$\epsilon - x_0$$，但网络在 $$x_t$$ 处看不到 $$x_0$$ 和 $$\epsilon$$，只能输出所有可能目标的平均。生成时顺着这个平均速度走出来的轨迹，和训练用的直线不是一回事（第 4.2 节）。
> 2. **CFM 损失有一个降不下去的下限。** 它等于条件速度在给定 $$x_t$$ 下的方差。比较不同路径、不同数据集的损失绝对值没有意义。
> 3. **时间是连续的。** 训练时 $$t \sim U(0, 1)$$（或第 8 节的 logit-normal），不是从 $$\{1, \dots, T\}$$ 里抽整数。网络的时间编码要能接收小数；本文沿用 DDPM 的正弦编码，输入前把 $$t$$ 乘以 1000，让它的数值范围和整数时间步相同。

## 4. Rectified Flow：直线路径与 Reflow

### 4.1 直线路径

Rectified Flow [[2]](#ref-2) 取最简单的路径：在 $$x_0$$ 和 $$\epsilon$$ 之间连直线，匀速走过去。

$$
x_t = (1-t)\,x_0 + t\,\epsilon,\qquad u_t(x_t \mid x_0, \epsilon) = \epsilon - x_0
\tag{6}
$$

损失就是

$$
\mathcal{L}_{\text{RF}}(\theta) = \mathbb{E}_{t \sim U(0,1),\, x_0,\, \epsilon}\big\lVert v_\theta\big((1-t)x_0 + t\epsilon,\ t\big) - (\epsilon - x_0)\big\rVert^2
\tag{7}
$$

选直线的理由是：**直线是用 Euler 法最容易积分准确的轨迹**。如果 ODE 的轨迹全是直线，速度在轨迹上不变，Euler 法一步就能精确走到终点。

### 4.2 条件路径是直线，生成轨迹却是弯的

训练时 $$x_0$$ 和 $$\epsilon$$ 是**独立随机配对**的，不同的直线会大量交叉（下图 a）。在交叉点 $$x_t$$，网络看到的是"往好几个方向走"的训练目标，只能输出它们的平均（式 4）。顺着平均速度走出来的轨迹不会交叉，但会**弯曲**（下图 b）：从高斯中心出发的点，先朝所有数据的平均方向走，越接近终点才越确定自己要去哪个模式。

![二维 Rectified Flow 与 Reflow 的轨迹](/deep-learning/assets/flow-matching/toy_paths.png)

图中数据是半径 3 的圆上 8 个高斯（和 GAN 篇的二维实验相同），网络是一个 4 层 MLP。(a) 是训练时随机配对的直线；(b) 是训练好的模型用 100 步 Euler 解 ODE 得到的轨迹，靠近中心的点轨迹明显弯曲；(c) 是第 4.3 节 Reflow 之后的轨迹。

轨迹弯曲的直接后果是**少步采样不准**。下图上面一行是同一个模型用不同步数采样：

![二维实验：不同步数的采样结果](/deep-learning/assets/flow-matching/toy_steps.png)

- **1 步**：所有点都落在原点附近。$$t = 1$$ 时 $$x_1 = \epsilon$$ 和 $$x_0$$ 完全无关，网络学到的速度是 $$\mathbb{E}[\epsilon - x_0 \mid \epsilon] = \epsilon - \mathbb{E}[x_0]$$，走一整步就落在数据的平均值 $$\mathbb{E}[x_0] = 0$$ 上。这和 DDIM 篇"一步得到平均图"是同一个现象。
- **2 步、4 步**：点停在 8 个模式之间的连线上，还没分开。
- **100 步**：96.6% 的点落在某个模式 0.5 以内，8 个模式都覆盖到了。

### 4.3 Reflow：用模型自己的配对重新训练

既然弯曲来自"随机配对导致的交叉"，那就换一种不交叉的配对。Rectified Flow 的做法是 [[2]](#ref-2)：

1. 训练好第一个模型（称为 1-Rectified Flow）；
2. 抽大量噪声 $$\epsilon$$，用它解 ODE 得到对应的生成结果 $$\hat x_0 = \mathrm{ODE}(\epsilon)$$，得到配对 $$(\hat x_0, \epsilon)$$；
3. 在**这些固定配对**上用同样的损失 (7) 重新训练一个模型（2-Rectified Flow）。

这个操作叫 **Reflow**，可以重复多次（k-Rectified Flow）。它有三个性质 [[2]](#ref-2)：

- **边缘分布不变。** 第 1 步模型的 ODE 把 $$\mathcal{N}(0, I)$$ 搬到它学到的数据分布，所以新配对里 $$\epsilon$$ 仍然服从标准高斯、$$\hat x_0$$ 仍然服从（模型学到的）数据分布，只是配对方式变了。
- **新的直线几乎不交叉。** ODE 的轨迹不会交叉，用它产生的配对连成的直线也很少交叉，于是在每个 $$x_t$$ 上，训练目标几乎是唯一的，网络学到的平均速度就是那条直线自己的速度，轨迹变直。
- **传输代价不增加。** 对任意凸代价函数 $$c$$，$$\mathbb{E}\,c(\hat x_0 - \epsilon)$$ 在 Reflow 后不会变大。直观地说，配对变得更"就近"了。

二维实验里，用 1-RF 的 100 步 ODE 生成 5 万对 $$(\hat x_0, \epsilon)$$，再训练一个同样结构的 2-RF。上面的图 (c) 中轨迹几乎都是直线，第二张图的下面一行在 **1 步**时已经覆盖了全部 8 个模式。用"直度"$$S = \mathbb{E}_t\lVert v_\theta(x_t, t) - (\epsilon - \hat x_0)\rVert^2$$ 来量化：沿模型自己从 $$\epsilon$$ 到 $$\hat x_0$$ 的轨迹，测量每一处的速度和首尾连线方向的偏差，0 表示完全是直线。

| | 直度 $$S$$ | 1 步 | 2 步 | 4 步 | 8 步 | 100 步 |
|---|---|---|---|---|---|---|
| 1-RF：落在模式附近的比例 | 2.23 | 0% | 17.8% | 78.5% | 90.6% | 96.6% |
| 1-RF：和自己 100 步结果的平方误差 | | 8.85 | 0.88 | 0.089 | 0.022 | 0 |
| 2-RF：落在模式附近的比例 | 0.0027 | 78.9% | 81.4% | 84.9% | 88.0% | 91.4% |
| 2-RF：和自己 100 步结果的平方误差 | | 0.050 | 0.025 | 0.011 | 0.0033 | 0 |

Reflow 让直度下降了约 800 倍，1 步的误差从 8.85 降到 0.05。代价是 2-RF 在 100 步时的质量（91.4%）**低于** 1-RF（96.6%）：第二个模型学的是第一个模型的输出，第一个模型的误差被继承下来，第二次训练又引入新的误差。这正是论文和后续工作通常只做一到两次 Reflow、然后接蒸馏的原因。

### 4.4 Reflow 之后：一步生成

2-RF 的轨迹已经很直，再把"多步 ODE 的结果"蒸馏进一个一步模型就容易得多。InstaFlow [[6]](#ref-6) 对 Stable Diffusion 1.5 做了一次 Reflow，再做蒸馏，得到一步出图的文生图模型。直接对原始 SD 做一步蒸馏效果很差，因为原始模型的轨迹弯曲，"噪声 → 图像"的映射太复杂，学生网络学不会。

除了 Reflow，还有别的办法让配对更"就近"，例如在每个 batch 内部用最优传输重新配对噪声和数据（minibatch OT），思路相同，这里不展开。

> **易错点**
>
> 1. **Reflow 的配对必须固定。** 第 3 步训练时 $$(\hat x_0, \epsilon)$$ 是一起抽出来的，不能再把 $$\hat x_0$$ 和新抽的随机噪声配对，否则又退回了独立配对，什么也没改变。
> 2. **生成配对要用确定性的 ODE。** 用随机采样器（比如每步加噪的 SDE）得到的 $$\hat x_0$$ 和 $$\epsilon$$ 之间没有确定的对应关系，配对没有意义。
> 3. **Reflow 不是免费的。** 每做一次，都要先用 ODE 生成大量配对（二维实验里是 5 万对 × 100 步），再从头训练；多步采样的质量还可能下降。它的价值在于少步和一步生成。

## 5. 和 DDPM、DDIM 的关系

### 5.1 同一个家族

第 3.4 节的表格说明：DDPM 的前向过程 $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$ 也是一条高斯路径，只是 $$a_t, s_t$$ 不同。**扩散模型和 Flow Matching 用的是同一类条件路径，区别只在三处**：

1. 路径系数 $$a_t, s_t$$（噪声调度）；
2. 网络预测什么（$$\epsilon$$、$$x_0$$ 还是速度）；
3. 损失在不同 $$t$$ 上的权重。

下面逐一说明这三者都可以互相换算 [[5]](#ref-5)。

### 5.2 预测目标可以互相换算

在直线路径上，$$x_t = (1-t)x_0 + t\epsilon$$、$$u = \epsilon - x_0$$，这是关于 $$(x_0, \epsilon)$$ 的两个线性方程，解出来：

$$
\hat x_0 = x_t - t\,\hat v,\qquad \hat\epsilon = x_t + (1-t)\,\hat v
\tag{8}
$$

验证：$$x_t - t(\epsilon - x_0) = (1-t)x_0 + t\epsilon - t\epsilon + tx_0 = x_0$$。所以一个预测速度的网络，同时也就给出了 $$\hat x_0$$ 和 $$\hat\epsilon$$，DDPM 篇第 10 节的所有换算在这里同样成立。

和 DDPM 篇 v-prediction 的关系：方差保持路径可以写成 $$a_t = \cos\phi_t$$、$$s_t = \sin\phi_t$$，那里的 $$v = a_t\epsilon - s_t x_0 = dx_t/d\phi$$ 是对角度求导；本文的速度是对时间求导，$$u = dx_t/dt = (d\phi/dt)\,v$$。两者方向相同，只差一个随时间变化的倍数。

### 5.3 损失权重可以互相换算

由式 (8)，预测误差之间的关系是

$$
\hat\epsilon - \epsilon = (1-t)(\hat v - u),\qquad \hat x_0 - x_0 = -t\,(\hat v - u)
$$

所以

$$
\lVert \hat v - u\rVert^2 = \frac{1}{(1-t)^2}\,\lVert\hat\epsilon - \epsilon\rVert^2 = \frac{1}{t^2}\,\lVert\hat x_0 - x_0\rVert^2
\tag{9}
$$

也就是说，**Flow Matching 的速度损失，等价于"预测噪声的 DDPM 损失乘以权重 $$1/(1-t)^2$$"**。这个权重在 $$t \to 1$$（高噪声端）时变大，相当于让模型更重视高噪声端，而原始 DDPM 的 $$\epsilon$$ 损失在高噪声端的权重偏小。SD3 论文 [[3]](#ref-3) 就是在这个统一视角下比较了十几种"路径 × 预测目标 × 权重"的组合（第 8 节）。

### 5.4 线性路径上，Euler 一步就是 DDIM 一步

DDIM（$$\eta = 0$$）的一步是：用网络估计 $$\hat x_0$$、$$\hat\epsilon$$，保持它们不变，把时间从 $$t$$ 换到更早的 $$t' < t$$（DDIM 篇记作 $$s$$，这里为了不和噪声系数 $$s_t$$ 混淆改写成 $$t'$$），重新组合：

$$
x_{t'}^{\text{DDIM}} = a_{t'}\,\hat x_0 + s_{t'}\,\hat\epsilon
$$

在直线路径上 $$a_{t'} = 1 - t'$$、$$s_{t'} = t'$$；Euler 法的一步是 $$x_{t'} = x_t + (t' - t)\,\hat v$$。代入式 (8)：

$$
(1-t')\,\hat x_0 + t'\,\hat\epsilon
= (1-t')(x_t - t\hat v) + t'(x_t + (1-t)\hat v)
= x_t + (t' - t)\,\hat v
$$

**两者完全相同。** 原因很直接：DDIM 的一步是"沿着由当前 $$\hat x_0$$、$$\hat\epsilon$$ 确定的那条条件路径走到 $$t'$$"；直线路径上的条件路径就是一条直线，沿直线走等于用当前速度乘以步长。第 7 节的代码在训练好的网络上数值验证了这一点（最大误差 $$10^{-7}$$ 量级，即浮点误差）。

因此，在 SD3 这样的模型上，"Euler 采样器"和"DDIM 采样器"是同一个东西；DDIM 篇第 5.2 节"DDIM 是在 $$(\bar x, \lambda)$$ 坐标下做 Euler"的结论，到了直线路径上就变成了"直接在 $$(x, t)$$ 坐标下做 Euler"。

### 5.5 对照表

| | DDPM / DDIM（本系列前两篇） | Flow Matching / Rectified Flow |
|---|---|---|
| 时间 | 离散 $$t \in \{1, \dots, T\}$$ | 连续 $$t \in [0, 1]$$ |
| 路径 | $$\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$（圆弧） | $$(1-t)x_0 + t\epsilon$$（直线） |
| 网络预测 | 噪声 $$\epsilon$$ | 速度 $$\epsilon - x_0$$ |
| 损失 | $$\lVert\hat\epsilon - \epsilon\rVert^2$$ | $$\lVert\hat v - u\rVert^2 = \lVert\hat\epsilon-\epsilon\rVert^2/(1-t)^2$$ |
| 起点（$$t = T$$ 或 $$t = 1$$）是否纯噪声 | 不是（$$\bar\alpha_T > 0$$，有残留信号） | 是（$$a_1 = 0$$） |
| 采样 | DDPM（SDE）或 DDIM（ODE） | ODE：Euler、Heun 等 |
| 少步采样 | DDIM 约 20–50 步 | Euler 约 20–50 步；Reflow 或蒸馏后 1–4 步 |

"起点是否纯噪声"这一行值得注意：DDPM 篇里 $$T = 300$$ 时 $$\sqrt{\bar\alpha_T} \approx 0.22$$，最后一步仍有残留信号，训练和采样的起点不一致；直线路径在 $$t = 1$$ 时 $$a_1 = 0$$，严格是纯噪声，没有这个问题。

## 6. 实现

### 6.1 训练

把式 (7) 直接写成代码。网络沿用 DDPM 篇的类别条件 Tiny U-Net，只把时间输入从整数步换成 $$[0, 1]$$ 里的小数：

```python
import torch
import torch.nn.functional as F

TIME_SCALE = 1000.0      # 正弦时间编码是为整数步 0..T-1 设计的，t ∈ [0,1] 先放大到同一量级


class FlowUNet(ClassConditionedTinyUNet):              # DDPM 篇第 9 节的类别条件 U-Net
    """v_θ(x_t, t, y)：t 是 [0,1] 里的连续时间，输出和 x_t 同形状的速度。"""
    def forward(self, x_t, t, labels=None):
        return super().forward(x_t, t * TIME_SCALE, labels)


def fm_loss(model, x0, labels, p_drop=0.1):
    """条件 Flow Matching 损失（直线路径），式 (7)。x0: [B,C,H,W]，已归一化到 [-1,1]。"""
    B = x0.shape[0]
    eps = torch.randn_like(x0)                         # ε ~ N(0, I)
    t = torch.rand(B, device=x0.device)                # t ~ U(0, 1)，每张图一个
    tt = t.view(B, 1, 1, 1)                            # [B] -> [B,1,1,1]，便于广播
    x_t = (1 - tt) * x0 + tt * eps                     # x_t = (1−t) x0 + t ε，式 (6)
    u = eps - x0                                       # 目标速度 u = ε − x0
    drop = torch.rand(B, device=x0.device) < p_drop    # CFG：10% 的样本丢掉类别
    labels = torch.where(drop, torch.full_like(labels, model.null_class_id), labels)
    return F.mse_loss(model(x_t, t, labels), u)        # ‖v_θ(x_t, t, y) − u‖²
```

和 DDPM 的训练循环相比，变化只有三处：时间是连续的；$$x_t$$ 用直线插值，没有噪声调度表；目标从 `noise` 换成 `eps - x0`。

时间 $$t$$ 也可以不均匀采样。SD3 用 logit-normal 分布（第 8.1 节），代码只差一行：

```python
def sample_t(B, mode="uniform", m=0.0, s=1.0):
    if mode == "uniform":
        return torch.rand(B)                           # t ~ U(0, 1)
    if mode == "logit_normal":
        return torch.sigmoid(m + s * torch.randn(B))   # t = sigmoid(n)，n ~ N(m, s²)
    raise ValueError(mode)
```

### 6.2 采样：Euler、Heun 和 CFG

生成就是从 $$t = 1$$ 到 $$t = 0$$ 解 ODE $$dx/dt = v_\theta(x, t)$$。把 $$[0, 1]$$ 分成 $$N$$ 段，时间点 $$1 = t_0 > t_1 > \dots > t_N = 0$$，步长 $$\Delta = t_{i+1} - t_i < 0$$：

- **Euler（一阶）**：$$x_{t_{i+1}} = x_{t_i} + \Delta\, v_\theta(x_{t_i}, t_i)$$。每步调用一次网络。
- **Heun（二阶）**：先用 Euler 预测一个点 $$\tilde x = x_{t_i} + \Delta\, v_\theta(x_{t_i}, t_i)$$，再用起点和预测点两处速度的平均走一步：$$x_{t_{i+1}} = x_{t_i} + \tfrac{\Delta}{2}\big(v_\theta(x_{t_i}, t_i) + v_\theta(\tilde x, t_{i+1})\big)$$。每步调用两次网络，误差随步长平方下降。最后一步走到 $$t = 0$$ 时通常退回 Euler，因为 $$t = 0$$ 附近网络不太可靠。

Classifier-free guidance 直接作用在速度上：

$$
\tilde v = v_\theta(x, t, \varnothing) + w\,\big(v_\theta(x, t, y) - v_\theta(x, t, \varnothing)\big)
$$

由式 (8)，$$\hat\epsilon$$ 是 $$\hat v$$ 的仿射函数，两路共享同一个 $$x_t$$，所以对速度做 CFG 和对噪声做 CFG 是等价的，$$w$$ 的含义和 DDPM 篇第 9.2 节相同。

```python
@torch.no_grad()
def velocity(model, x, t, labels, w):
    """CFG 后的速度：v = v_uncond + w (v_cond − v_uncond)。两路拼成一个 2B 的 batch 一次算完。"""
    B = x.shape[0]
    tb = torch.full((B,), float(t), device=x.device)
    null = torch.full_like(labels, model.null_class_id)
    v = model(torch.cat([x, x]), torch.cat([tb, tb]), torch.cat([labels, null]))
    v_c, v_u = v.chunk(2)
    return v_u + w * (v_c - v_u)


@torch.no_grad()
def sample(model, eps, labels, num_steps, w=3.0, solver="euler"):
    """从 t=1（x = ε）积分到 t=0。eps: [B,C,H,W]，labels: [B]。"""
    x = eps.clone()
    ts = torch.linspace(1.0, 0.0, num_steps + 1)       # t_0 = 1 > t_1 > ... > t_N = 0
    for i in range(num_steps):
        t, t_next = ts[i], ts[i + 1]
        dt = t_next - t                                # Δ < 0
        v = velocity(model, x, t, labels, w)
        if solver == "euler" or t_next == 0:           # Heun 的最后一步退回 Euler
            x = x + dt * v                             # Euler：x ← x + Δ·v
        else:
            x_pred = x + dt * v                        # Heun：先预测
            v_next = velocity(model, x_pred, t_next, labels, w)
            x = x + dt * 0.5 * (v + v_next)            # 再用两端速度的平均
    return x                                           # 约在 [-1, 1]，显示前 (x+1)/2
```

### 6.3 总结：训练和采样的完整流程

| 步骤 | 训练（每个 batch） | 采样 |
|---|---|---|
| 1 | 取数据 $$x_0$$，抽 $$\epsilon \sim \mathcal{N}(0, I)$$、$$t \sim U(0,1)$$ | 抽 $$x \sim \mathcal{N}(0, I)$$，$$t = 1$$ |
| 2 | $$x_t = (1-t)x_0 + t\epsilon$$ | 选时间点 $$1 = t_0 > \dots > t_N = 0$$ |
| 3 | 以概率 0.1 把类别换成空类别 | 每步算有条件、无条件两路速度，做 CFG |
| 4 | 损失 $$\lVert v_\theta(x_t, t, y) - (\epsilon - x_0)\rVert^2$$ | Euler：$$x \leftarrow x + (t_{i+1} - t_i)\,\tilde v$$ |
| 5 | 反向传播，更新 $$\theta$$，更新 EMA | 走到 $$t = 0$$，输出 $$x$$ |

> **易错点**
>
> 1. **时间方向和符号。** 本文 $$t = 1$$ 是噪声，采样时 $$t$$ 递减，步长 $$\Delta$$ 是负数，更新是 $$x + \Delta\cdot v$$。照着 Flow Matching 原论文（$$t = 1$$ 是数据）写代码时，速度目标是 $$x_{\text{data}} - x_{\text{noise}}$$，采样时 $$t$$ 递增。两种约定混用，会得到方向完全相反的速度，模型能训练，但采样结果是噪声。加载别人的权重前要先确认约定。
> 2. **目标是 $$\epsilon - x_0$$，不是 $$\epsilon$$。** 从 DDPM 改代码时最容易漏改这一处。
> 3. **$$t$$ 要广播到图像的形状。** `t` 是 `[B]`，插值前要 `view(B, 1, 1, 1)`；直接写 `(1 - t) * x0` 会报错，或者在 B 恰好等于某个维度时静默地广播错。
> 4. **时间编码的尺度。** 正弦编码的频率是为 0 到几百的整数设计的；直接输入 $$[0, 1]$$ 的 $$t$$，大部分频率几乎不变，网络很难分辨不同时刻。乘以 1000（或者改编码的频率范围）后再输入。
> 5. **端点处的数值问题。** 用式 (8) 由 $$\hat v$$ 反推 $$\hat x_0$$ 时没有除法，不会像 DDPM 那样在高噪声端放大误差；但如果自己写的换算里出现 $$1/t$$ 或 $$1/(1-t)$$（例如由 $$\hat\epsilon$$ 反推速度），在端点附近要截断 $$t$$，如限制在 $$[10^{-5}, 1 - 10^{-5}]$$。
> 6. **CFG 时两路共用同一个 $$t$$ 和 $$x$$。** 把有条件和无条件拼成一个 2B 的 batch 时，时间和输入都要复制两份，标签一半是真实类别、一半是空类别。

## 7. 实验：FashionMNIST

网络与 DDPM 篇相同（类别条件 Tiny U-Net，`base_channels = 48`，46 万参数），用第 6.1 节的损失训练 2500 步（batch 128，Adam，峰值学习率 $$10^{-3}$$，EMA 0.999），在 4 核 CPU 上约 19 分钟。下面所有采样都用 EMA 权重、CFG $$w = 3$$。

![Flow Matching 训练损失](/deep-learning/assets/flow-matching/fmnist_curve.png)

损失在前 500 步降到 0.3 以下，之后缓慢降到约 0.25 就几乎不动。如第 3.3 节所说，这是条件速度的方差决定的下限，不代表模型停止进步。

### 7.1 步数

每行一个类别（裤子、运动鞋、包、短靴、连衣裙、套头衫），同一行从同一个 $$\epsilon$$ 出发，只改 Euler 的步数：

![FashionMNIST：不同 Euler 步数](/deep-learning/assets/flow-matching/fmnist_steps.png)

- **1 步**：一步从 $$t = 1$$ 走到 $$t = 0$$，得到的是 $$\hat x_0 = \epsilon - \hat v$$，也就是网络在纯噪声上对 $$x_0$$ 的平均猜测：轮廓和类别对，但模糊，和二维实验的 1 步、DDIM 篇的 1 步是同一个现象。
- **2 步**已经能看出具体形状，**5 步**基本成形，**10 步以后**和 50 步几乎看不出区别。同一行的图形状一致，说明它们是同一条 ODE 轨迹在不同精度下的近似。
- 这个模型没有做 Reflow，少步的表现和 DDIM 篇里 DDPM 模型用 DDIM 采样时相近，符合第 5.4 节的结论：在线性路径上 Euler 就是 DDIM。

在这个训练好的网络上，取 $$t = 0.7$$、$$t' = 0.4$$，分别用 Euler 一步和"先由式 (8) 换算 $$\hat x_0, \hat\epsilon$$ 再做 DDIM 一步"，两者最大差值为 $$3.6 \times 10^{-7}$$，即 float32 的舍入误差。

### 7.2 采样器的阶数

用 64 个不同的 $$\epsilon$$，以 200 步 Heun 的结果为参照，比较 Euler 和 Heun 在相同网络调用次数下的误差：

![Euler 与 Heun 的收敛](/deep-learning/assets/flow-matching/fmnist_solvers.png)

- 网络调用次数很少（3、5 次）时，Heun 只能走 2、3 步，和 Euler 差不多，没有优势。
- 调用次数到 10 次以上，Heun 明显更准：约 20 次调用时，Euler 的误差是 $$1.7 \times 10^{-3}$$，Heun 是 $$5.0 \times 10^{-4}$$，约小 3.4 倍。
- Euler 在双对数坐标下的斜率约为 $$-2$$（10 次到 50 次调用，平方误差从 $$5.3 \times 10^{-3}$$ 降到 $$3.0 \times 10^{-4}$$），和 DDIM 篇第 8.2 节一样，是一阶方法的特征。Heun 的斜率更陡，但小于理论上的 $$-4$$，因为参照本身也只是 200 步的近似解，而且 CFG 让速度场不那么平滑。

实际的大模型里两种都常用：SD3 默认 Euler，EDM 风格的采样器用 Heun。轨迹越弯，高阶方法的优势越大；轨迹越直（例如 Reflow 之后），Euler 就越够用。

## 8. SD3 和 FLUX 里的 Flow Matching

SD3 [[3]](#ref-3) 在同一个网络结构上系统比较了多种"路径 × 预测目标 × 时间采样"的组合（第 5 节的统一视角），最后选择 Rectified Flow 加 logit-normal 时间采样。下面是它在第 6 节基础上的主要改动。

### 8.1 logit-normal 时间采样

均匀采样 $$t$$ 时，每个时刻得到的训练量相同。但 $$t$$ 接近 0 或 1 时任务相对简单：接近 0 时 $$x_t \approx x_0$$，图像已经确定，只剩下去掉一点点噪声；接近 1 时 $$x_t \approx \epsilon$$，输入里几乎没有图像信息，最优速度就是 $$\epsilon - \mathbb{E}[x_0]$$，网络能学到的东西很少。决定图像结构的是中间段。SD3 让 $$t$$ 服从 logit-normal 分布：

$$
t = \operatorname{sigmoid}(n),\quad n \sim \mathcal{N}(m, s^2),\qquad
\pi(t) = \frac{1}{s\sqrt{2\pi}}\,\frac{1}{t(1-t)}\exp\!\Big(-\frac{(\operatorname{logit}(t) - m)^2}{2s^2}\Big)
$$

$$m = 0, s = 1$$ 时密度在 $$t = 0.5$$ 附近最高，两端趋于 0。论文的消融实验中，这个设置（记作 rf/lognorm(0.00, 1.00)）在不同采样步数下综合表现最好。代码就是第 6.1 节 `sample_t` 的 `logit_normal` 分支。

由式 (9)，对 $$t$$ 做非均匀采样等价于给损失乘一个随 $$t$$ 变化的权重，所以它和第 5.3 节"不同预测目标对应不同权重"是同一类调整。

### 8.2 分辨率相关的时间偏移

同样的 $$t$$，对高分辨率图像来说"噪声不够多"：图像越大，相邻像素越相关，同样强度的独立噪声更容易被平均掉，粗略的结构在较大的 $$t$$ 时仍然可见。SD3 的做法是把时间往高噪声端推 [[3]](#ref-3)：

$$
t' = \frac{\alpha\, t}{1 + (\alpha - 1)\, t},\qquad \alpha = \sqrt{\frac{m}{n}}
$$

其中 $$n$$ 是基准分辨率的像素数，$$m$$ 是目标分辨率的像素数。$$\alpha = 1$$ 时不变；$$\alpha > 1$$ 时，$$t' \ge t$$，每个时刻的噪声更多，并且 $$t' = 0, 1$$ 两个端点不变。SD3 在 1024×1024 上用 $$\alpha = 3$$，训练和采样都做这个偏移。

```python
def shift_t(t, alpha=3.0):
    """t' = α t / (1 + (α−1) t)。α > 1 把时间往高噪声端推。"""
    return alpha * t / (1 + (alpha - 1) * t)

ts = shift_t(torch.linspace(1.0, 0.0, num_steps + 1))  # 采样用的时间点也要偏移
```

FLUX 用的是同一个函数，写法是 $$t' = e^{\mu}/(e^{\mu} + (1/t - 1))$$，即 $$\alpha = e^{\mu}$$，并让 $$\mu$$ 随图像 token 数线性增大：分辨率越高，偏移越多。

### 8.3 其他

- **主干网络。** SD3 和 FLUX 都把 U-Net 换成了 Transformer（MM-DiT），文本 token 和图像 token 在同一个注意力里交互。Flow Matching 只决定训练目标和采样方式，和网络结构无关；本文的 Tiny U-Net 换成任意"输入 $$x_t, t$$、输出同形状速度"的网络都可以。
- **latent 空间。** 和 [LDM](/deep-learning/latent-diffusion/) 一样，$$x_0$$ 是自编码器的 latent，不是像素。SD3 的 latent 有 16 个通道（SD 1.5 是 4 个）。
- **采样步数。** SD3 默认用 28 步 Euler；在线性路径上这就是 28 步 DDIM（第 5.4 节）。
- **引导蒸馏。** FLUX.1-dev 把 CFG 的 $$w$$ 当成一个输入（类似时间 $$t$$）蒸馏进网络，每步只需调用一次网络，不再需要两路；FLUX.1-schnell 进一步蒸馏到 1–4 步。

## 9. 自检

下面的检查不需要训练好的模型，可以在改动代码后快速确认公式没有写错：

```python
import torch

x0, eps = torch.randn(4, 1, 28, 28), torch.randn(4, 1, 28, 28)
t = torch.rand(4).view(4, 1, 1, 1)
x_t = (1 - t) * x0 + t * eps
u = eps - x0

# 1. 端点：t=0 是数据，t=1 是噪声
assert torch.allclose((1 - 0) * x0 + 0 * eps, x0) and torch.allclose(0 * x0 + 1 * eps, eps)

# 2. 速度和 x_t 关于 t 的数值导数一致
h = 1e-3
x_th = (1 - (t + h)) * x0 + (t + h) * eps
assert torch.allclose((x_th - x_t) / h, u, atol=1e-3)

# 3. 式 (8)：由真实速度精确还原 x0 和 ε
assert torch.allclose(x_t - t * u, x0, atol=1e-5)
assert torch.allclose(x_t + (1 - t) * u, eps, atol=1e-5)

# 4. 速度完美时，Euler 一步从 t=1 走到 t=0 就精确还原 x0（路径是直线）
assert torch.allclose(eps + (0 - 1) * u, x0, atol=1e-5)

# 5. 式 (9)：速度误差和噪声误差的换算
dv = torch.randn_like(u)
eps_hat = x_t + (1 - t) * (u + dv)
assert torch.allclose(eps_hat - eps, (1 - t) * dv, atol=1e-5)

# 6. 第 5.4 节：任意速度下，Euler 一步等于 DDIM 一步
v_hat, t_prev = torch.randn_like(u), 0.3
t1 = torch.full_like(t, 0.8); x1 = (1 - t1) * x0 + t1 * eps
euler = x1 + (t_prev - t1) * v_hat
ddim = (1 - t_prev) * (x1 - t1 * v_hat) + t_prev * (x1 + (1 - t1) * v_hat)
assert torch.allclose(euler, ddim, atol=1e-5)
print("all checks passed")
```

## 10. 要点回顾

- **Flow Matching 学的是速度场。** 生成就是从标准高斯出发解 ODE $$dx/dt = v_\theta(x, t)$$，从 $$t = 1$$ 积分到 $$t = 0$$。
- **为什么能训练。** 边缘速度场写不出来，但它等于条件速度的条件期望（式 4）；用条件速度做 MSE 回归（式 5），梯度和直接回归边缘速度场完全相同。
- **Rectified Flow 用直线路径。** $$x_t = (1-t)x_0 + t\epsilon$$，目标 $$\epsilon - x_0$$，代码比 DDPM 还短；$$t = 1$$ 严格是纯噪声。
- **生成轨迹仍然是弯的。** 随机配对的直线会交叉，网络在交叉处学到的是平均速度；1 步采样落在数据均值上。Reflow 用模型自己的配对重新训练，把轨迹拉直，换来少步生成，代价是多步质量可能略降。
- **和扩散模型是一回事的不同写法。** 都是高斯路径；预测 $$\epsilon$$、$$x_0$$、速度可以互相换算（式 8）；速度损失等于加权的噪声损失（式 9）；在直线路径上 Euler 一步就是 DDIM 一步。
- **SD3 / FLUX 的实际配方。** Rectified Flow + logit-normal 时间采样 + 随分辨率偏移的时间 + Transformer 主干 + Euler 采样。
