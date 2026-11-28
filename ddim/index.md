---
date: 2026-11-26 19:30:00 -0800
tags: [Diffusion, Sampling]
keywords: >-
  DDIM, denoising diffusion implicit models, fast sampling, non-Markovian, deterministic sampling, eta, probability flow ODE, Euler, DDIM inversion, timestep spacing, PyTorch.
summary: >-
  DDIM 为什么不用重新训练就能把 DDPM 的采样从几百步降到几十步：训练只依赖边缘分布，于是可以换一个能跳步的反向过程；确定性 DDIM 等价于一个 ODE 的 Euler 解法。从推导到逐行对应公式的 PyTorch 实现，再到 DDIM 反演，附 FashionMNIST 上的步数、收敛、η、插值和反演实验。
layout: note
title: "DDIM：为什么能跳步采样，以及怎么实现"
description: "Why DDIM can sample in a few dozen steps with an unchanged DDPM model: derivation, ODE view, a formula-by-formula PyTorch implementation, inversion, and experiments."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
</style>

# DDIM：为什么能跳步采样，以及怎么实现

上一篇 [DDPM 从零实现](/deep-learning/ddpm-from-scratch/) 训练好的模型，用 DDPM 采样一张图要走 $$T = 300$$ 步，也就是调用 300 次网络。DDIM（Denoising Diffusion Implicit Models）**不改模型、不重新训练**，只换一个采样器，就能用 20～50 步得到几乎一样的结果。

本文围绕三个问题展开：

1. **为什么可以跳步？** 关键在于：DDPM 的训练只用到了边缘分布 $$q(x_t \mid x_0)$$。
2. **为什么跳了步结果还准？** 确定性的 DDIM 等价于用 Euler 法求解一个平滑的 ODE，步长可以很大。
3. **具体怎么实现？** 给出时间步选择、单步更新、完整采样循环和 DDIM 反演的代码，变量名和公式一一对应，并列出易错点。

最后用上一篇在 FashionMNIST 上训练的同一个模型做实验，所有图都是真实运行结果。

## 参考资料

本文主要参考 DDIM 原论文 [1]；它建立在 DDPM [3] 之上。正文中用 [n] 标注具体出处。

- <a id="ref-1"></a>**[1] 主论文**：Song, Meng, Ermon. *Denoising Diffusion Implicit Models*. ICLR 2021. [arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502)。非马尔可夫的反向过程、$$\eta$$、跳步采样和反演都来自这篇（第 4 节和附录）。
- <a id="ref-2"></a>**[2] 官方代码**：ermongroup/ddim [github.com/ermongroup/ddim](https://github.com/ermongroup/ddim)。PyTorch 实现，采样部分可以和第 6 节对照。
- <a id="ref-3"></a>**[3] DDPM**：Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)。训练目标和记号，见 [DDPM 篇](/deep-learning/ddpm-from-scratch/)。

## 背景与预备知识

本节介绍后文用到的基础知识：从上一篇带过来的三个公式，以及两个概念（马尔可夫链与边缘分布、ODE 与 Euler 法）。熟悉这些内容的读者可以直接阅读第 1 节。

### 从 DDPM 带过来的三个公式

DDIM 完全复用 DDPM 训练好的网络 $$\epsilon_\theta(x_t, t)$$ 和同一张调度表 $$\bar\alpha_t$$，只用到下面三个公式（都在 [DDPM 篇](/deep-learning/ddpm-from-scratch/)推导过）：

| 公式 | 含义 | DDPM 篇 |
|---|---|---|
| $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$ | 从干净图像一步加噪到第 $$t$$ 步 | 第 4 节 |
| $$\hat x_0 = \big(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)\big) / \sqrt{\bar\alpha_t}$$ | 用网络预测的噪声反推干净图像 | 第 7 节 |
| $$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$$ | DDPM 的反向一步，每次只能退一步 | 第 7 节 |

第一行是理解 DDIM 的关键：$$x_t$$ 只由"一张干净图"和"一份噪声"按比例混合而成，比例只取决于 $$t$$。

### 马尔可夫链和边缘分布

**马尔可夫链**：一串随机变量 $$x_0, x_1, \dots, x_T$$，每一个只依赖于前一个，即 $$x_t$$ 只由 $$x_{t-1}$$ 决定。DDPM 的前向加噪就是这样一条链：每步在上一步的结果上加一点噪声。

**边缘分布**：不管中间怎么走，只看"第 $$t$$ 步的 $$x_t$$ 长什么样"。DDPM 里它就是第一行公式 $$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t) I)$$。

类比来说：从一楼走到十楼，"每次只能上一层"是一种走法（马尔可夫链），"坐电梯直达"是另一种走法；但只要每一层楼的位置一样（边缘分布相同），站在第 7 层看到的风景就是一样的。第 3 节会看到，DDPM 训练时只看过"每一层的风景"，从没要求"必须一层一层走"，DDIM 就利用这一点换成了能直达的走法。

### ODE 和 Euler 法

**常微分方程（ODE）** $$\frac{dy}{dx} = f(y, x)$$ 描述的是"在每个位置，$$y$$ 该朝哪个方向、以多快的速度变化"。知道起点和每处的斜率 $$f$$，就能沿着斜率一路走出整条轨迹。

**Euler 法**是最简单的数值解法：在当前点算一次斜率，假设这一小段里斜率不变，直接走一步：

$$
y_{\text{next}} = y + (x_{\text{next}} - x)\cdot f(y, x)
$$

例如 $$\frac{dy}{dx} = -y$$、$$y(0) = 1$$，精确解是 $$y(1) = e^{-1} \approx 0.368$$：

```python
def euler(num_steps):
    y, h = 1.0, 1.0 / num_steps      # 起点 y(0) = 1，步长 h
    for _ in range(num_steps):
        y = y + h * (-y)              # 斜率 f(y) = -y
    return y

print(euler(1), euler(10), euler(100))   # 0.0  0.3487  0.3660，步数越多越接近 0.368
```

两个要点后面会用到：**步长越大误差越大**（误差大致和步长成正比）；**轨迹越平滑，大步长也越准**。第 5 节会说明确定性的 DDIM 正是在用 Euler 法解一个关于去噪的 ODE，网络预测的噪声就是斜率，"跳步"就是"大步长"。

### 数值示例：一个像素的跳步

与 DDPM 篇相同，用一个像素的具体数值演示 DDIM 的一步（$$T = 300$$ 的调度表；第 4 节会推导公式）：

1. 干净像素 $$x_0 = 0.6$$、噪声 $$\epsilon = -1.2$$，在 $$t = 300$$ 时 $$\sqrt{\bar\alpha_{300}} = 0.219$$、$$\sqrt{1-\bar\alpha_{300}} = 0.976$$，所以 $$x_{300} = 0.219 \times 0.6 + 0.976 \times (-1.2) = -1.039$$。
2. 网络预测噪声 $$\hat\epsilon = -1.15$$（只差 0.05），反推 $$\hat x_0 = (-1.039 - 0.976 \times (-1.15)) / 0.219 = 0.378$$。噪声只猜错 0.05，干净像素却差了 0.22：高噪声时误差被放大了 $$0.976 / 0.219 \approx 4.5$$ 倍。
3. DDIM 直接跳到 $$t = 200$$（$$\sqrt{\bar\alpha_{200}} = 0.509$$、$$\sqrt{1-\bar\alpha_{200}} = 0.861$$）：用**同一个** $$\hat x_0$$ 和**同一个** $$\hat\epsilon$$ 按新比例重新混合，$$x_{200} = 0.509 \times 0.378 + 0.861 \times (-1.15) = -0.798$$。一次网络调用就跨过了 100 步，DDPM 要调用 100 次。
4. 到了 $$t = 200$$ 噪声变小，网络会给出更准的 $$\hat\epsilon$$，$$\hat x_0$$ 的误差也被放大得更少，于是后面的步骤会逐步修正第 2 步的偏差。

## 1. 符号约定

沿用上一篇的记号，新增的只有"跳到哪一步"的 $$s$$ 和控制随机性的 $$\eta$$：

| 公式 | 含义 | 代码 | shape |
|---|---|---|---|
| $$t$$ | 当前时间步（论文记号，$$1 \dots T$$） | `t_index = t - 1` | Python int |
| $$s$$ | 要跳到的更早时间步，$$s < t$$；$$s = 0$$ 表示干净图像 | `prev_index = s - 1`（`-1` 表示干净图像） | Python int |
| $$\bar\alpha_t$$ | 累乘系数，$$x_t$$ 里信号的"保留比例" | `sched["alpha_bars"][t_index]` | 标量 |
| $$\bar\alpha_0 = 1$$ | 干净图像没有噪声 | `prev_index == -1` 时取 `1.0` | 标量 |
| $$\epsilon_\theta(x_t, t)$$ | 网络预测的噪声 | `pred_noise` | `[B, C, H, W]` |
| $$\hat x_0$$ | 由预测噪声反推的干净图像 | `x0_hat` | `[B, C, H, W]` |
| $$\sigma_{t \to s}$$ | 这一步注入的随机噪声的标准差 | `torch.sqrt(sigma2)` | 标量 |
| $$\eta$$ | 随机性开关：0 为确定性，1 为 DDPM 式 | `eta` | float |
| $$N$$ | 采样步数（网络调用次数） | `num_steps` | int |

## 2. DDPM 为什么慢

DDPM 的反向过程是一条马尔可夫链，每一步从 $$x_t$$ 走到 $$x_{t-1}$$：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\Big) + \sigma_t z,\qquad z \sim \mathcal{N}(0, I)
$$

它不能跳步，原因有两个：

- **高斯近似只对小步成立。** 网络学的是 $$p_\theta(x_{t-1} \mid x_t)$$，它被假设成高斯分布。$$\beta_t$$ 很小时这个假设近似成立；如果一步从 $$t$$ 跳到 $$t-50$$，真实的 $$q(x_{t-50} \mid x_t)$$ 会变成多峰分布（同一张噪声图可能对应很多不同的干净图），一个高斯描述不了。
- **每一步都注入新噪声。** DDPM 每步都加 $$\sigma_t z$$，后面的步骤还要负责把这些新噪声去掉。轨迹是随机的、锯齿状的，只能小步慢走。

所以 DDPM 采样的代价就是 $$T$$ 次网络前向。

## 3. 关键观察：训练只用到了边缘分布

回头看 DDPM 的训练目标 [[1]](#ref-1)：

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\,t,\,\epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big) \big\rVert^2\Big]
$$

这里只出现了**单个时间步**的分布 $$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t) I)$$，完全没有用到"$$x_{t-1}$$ 和 $$x_t$$ 之间是什么关系"。也就是说：

> 任何一个前向过程，只要每个 $$t$$ 的边缘分布 $$q(x_t \mid x_0)$$ 和 DDPM 相同，训练出来的就是**同一个** $$\epsilon_\theta$$。

DDPM 的马尔可夫链只是满足这个条件的其中一种。DDIM 的做法是换一种同样满足条件、但**允许在任意两个时间步之间直接跳**的过程，然后用它的反向过程采样。网络不用动。

## 4. 构造能跳步的反向过程

### 4.1 公式

对任意 $$s < t$$，DDIM 定义"已知 $$x_t$$ 和 $$x_0$$ 时 $$x_s$$ 的分布"为 [[1]](#ref-1)：

$$
q_\sigma(x_s \mid x_t, x_0) = \mathcal{N}\Big(
\sqrt{\bar\alpha_s}\,x_0 + \sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\cdot\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}},\ \ 
\sigma_{t\to s}^2 I\Big)
$$

这个式子可以这样理解：$$\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}$$ 正好就是 $$x_t$$ 里的那份噪声 $$\epsilon$$。所以这个分布的意思是：

$$
x_s = \sqrt{\bar\alpha_s}\,x_0 \;+\; \sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\;\epsilon \;+\; \sigma_{t\to s}\,z
$$

**用同一张干净图 $$x_0$$、（大部分）同一份噪声 $$\epsilon$$，按时间步 $$s$$ 的比例重新组合一遍。**

### 4.2 验证：边缘分布没变

给定 $$x_0$$ 时，$$\epsilon$$ 和 $$z$$ 是独立的标准高斯，所以 $$x_s$$ 是高斯分布，均值是 $$\sqrt{\bar\alpha_s}\,x_0$$，方差是

$$
\big(1-\bar\alpha_s-\sigma_{t\to s}^2\big) + \sigma_{t\to s}^2 = 1-\bar\alpha_s
$$

也就是 $$q(x_s \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_s}\,x_0, (1-\bar\alpha_s) I)$$，和 DDPM 完全一样。这对**任意** $$s < t$$ 都成立，不要求 $$s = t-1$$，所以第 3 节的条件满足，而且天然支持跳步。$$\sigma_{t\to s}$$ 可以在 $$[0, \sqrt{1-\bar\alpha_s}]$$ 之间随便选，每种选法都是一个合法的过程。

### 4.3 采样时用 $$\hat x_0$$ 代替 $$x_0$$

采样时不知道 $$x_0$$，就用网络的估计。网络预测噪声 $$\epsilon_\theta(x_t, t)$$，由 $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$ 反解：

$$
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}
$$

代入 4.1，就得到 DDIM 的更新公式：

$$
\boxed{\;x_s = \underbrace{\sqrt{\bar\alpha_s}\,\hat x_0}_{\text{干净图像部分}} \;+\; \underbrace{\sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\;\epsilon_\theta(x_t, t)}_{\text{指向 } x_t \text{ 的噪声方向}} \;+\; \underbrace{\sigma_{t\to s}\,z}_{\text{随机部分}}\;}
$$

每一步都在做同一件事：**先猜出最终的干净图像 $$\hat x_0$$，再把它重新加噪到更早的时间步 $$s$$，而且加回去的主要是刚才预测出来的那份噪声。**

### 4.4 $$\eta$$：在 DDIM 和 DDPM 之间切换

$$\sigma_{t\to s}$$ 的常用选法是：

$$
\sigma_{t\to s} = \eta\,\sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}}\;\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_s}}
$$

- **$$\eta = 0$$**：$$\sigma = 0$$，没有随机项，整个采样是确定性的：同一个初始噪声 $$x_T$$ 永远得到同一张图。这就是通常说的 DDIM。
- **$$\eta = 1$$ 且 $$s = t-1$$**：因为 $$\bar\alpha_t / \bar\alpha_{t-1} = \alpha_t = 1 - \beta_t$$，

$$
\sigma_{t\to t-1}^2 = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t = \tilde\beta_t
$$

  正好是 DDPM 的后验方差，DDIM 退化回 DDPM。所以 **DDPM 是 DDIM 的一个特例**，$$\eta$$ 连续地控制随机性。
- **$$0 < \eta < 1$$**：介于两者之间。

> **易错点**
>
> 1. **$$\sigma$$ 公式里是 $$\bar\alpha_t / \bar\alpha_s$$，不是 $$\beta_t$$。** 跳步时两步之间隔了很多个 $$\beta$$，这一跳的"等效 $$\beta$$"是 $$1 - \bar\alpha_t/\bar\alpha_s$$。直接套 DDPM 的 $$\beta_t$$ 会让随机项小一个数量级。
> 2. **噪声方向用的是预测噪声 $$\epsilon_\theta$$，不是新采样的噪声。** 新噪声只出现在 $$\sigma z$$ 这一项。如果把方向项也换成 `randn`，$$\eta = 0$$ 时也不再是确定性的，结果会变成一团噪声。
> 3. **$$\eta = 0$$ 的确定性只来自采样器本身。** 如果网络里有 dropout 却忘了 `model.eval()`，或者训练时随机丢标签的逻辑在采样时也被调用了，同一个 $$x_T$$ 的结果仍然会变。

## 5. 为什么跳了步结果还准

第 4 节说明了 DDIM **允许**跳步。但跳一大步之后，结果为什么还和走完 300 步差不多？ [[1]](#ref-1)

### 5.1 如果噪声预测是完美的，一步就够

假设网络对某张图 $$x_0$$ 的噪声预测完全准确，即 $$\epsilon_\theta(x_t, t) = \epsilon$$。那么 $$\hat x_0 = x_0$$ 恰好成立，$$\eta = 0$$ 时一步从 $$t = T$$ 跳到 $$s = 0$$（$$\bar\alpha_0 = 1$$）：

$$
x_0^{\text{DDIM}} = \sqrt{1}\cdot\hat x_0 + \sqrt{1 - 1}\cdot\epsilon = x_0
$$

一步就精确还原。第 10 节的自检会用一个"知道答案"的假网络验证：步数取 1、5 还是 50，结果都和 $$x_0$$ 一致。

真实网络做不到这一点：在噪声很大的 $$x_T$$ 上，它最好也只能预测**所有可能的干净图的平均**（MSE 训练出来的就是条件期望），所以一步得到的 $$\hat x_0$$ 是一张模糊的平均图（第 8.1 节实验的第一列）。走到噪声小一些的时间步后，网络看到了更多信息，$$\hat x_0$$ 会越来越清楚。**多走几步，就是为了让网络在越来越清楚的输入上反复修正 $$\hat x_0$$。** 只要 $$\epsilon_\theta$$ 沿着轨迹变化得平滑，就不需要每个时间步都修正一次。

### 5.2 确定性 DDIM 是一个 ODE 的 Euler 法

把 $$\eta = 0$$ 的更新公式两边都除以 $$\sqrt{\bar\alpha_s}$$，并代入 $$\hat x_0$$：

$$
\frac{x_s}{\sqrt{\bar\alpha_s}}
= \frac{x_t}{\sqrt{\bar\alpha_t}} - \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\,\epsilon_\theta + \sqrt{\frac{1-\bar\alpha_s}{\bar\alpha_s}}\,\epsilon_\theta
$$

记 $$\bar x = x / \sqrt{\bar\alpha}$$（把信号缩放回原尺度），$$\lambda = \sqrt{(1-\bar\alpha)/\bar\alpha}$$（噪声和信号的比例，从 $$t=0$$ 的 0 增大到 $$t=T$$ 的最大值），就变成

$$
\bar x_s = \bar x_t + (\lambda_s - \lambda_t)\,\epsilon_\theta(x_t, t)
$$

这正是常微分方程

$$
\frac{d\bar x}{d\lambda} = \epsilon_\theta(x, t)
$$

的 **Euler 法**：当前点的斜率 $$\times$$ 步长 $$(\lambda_s - \lambda_t)$$。这样就能解释为什么能加速：

- **ODE 的轨迹是确定、平滑的。** 没有每步新注入的噪声，斜率 $$\epsilon_\theta$$ 沿轨迹缓慢变化，大步长也不会偏得太远。
- **误差随步长线性下降。** Euler 法是一阶方法，步长缩小 10 倍，误差缩小约 10 倍，平方误差缩小约 100 倍。第 8.2 节的收敛实验正好是这个斜率。
- **更快的采样器就是更好的 ODE 解法。** DPM-Solver、Heun、UniPC 等用高阶方法解同一个 ODE，用更少的步数达到同样的误差（第 9 节）。

DDPM（$$\eta = 1$$）对应的是带随机项的 SDE，随机项要靠小步长才能被准确模拟，所以它少步时退化得比 DDIM 快。

### 5.3 几何直觉：在一段圆弧上转动

因为 $$(\sqrt{\bar\alpha_t})^2 + (\sqrt{1-\bar\alpha_t})^2 = 1$$，可以把 $$x_t$$ 想成干净图像和噪声这两个方向构成的平面上、单位圆上的一个点，角度越大噪声越多。$$\eta = 0$$ 的一步，就是**固定当前对"干净图像"和"噪声"这两个方向的估计，把角度从 $$t$$ 转到 $$s$$**：

<figure class="ddpm-fig">
<svg viewBox="0 0 760 330" role="img" aria-label="DDIM 一步在圆弧上转动的示意图">
  <defs><marker id="arA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker>
  <marker id="arB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#2a6fdb"/></marker></defs>
  <line x1="80" y1="290" x2="320" y2="290" stroke="#475467" marker-end="url(#arA)"/>
  <line x1="80" y1="290" x2="80" y2="50" stroke="#475467" marker-end="url(#arA)"/>
  <text x="300" y="312" text-anchor="middle">干净图像的估计</text>
  <text x="90" y="44">噪声的估计</text>
  <path d="M80,90 A200,200 0 0,1 280,290" fill="none" stroke="#d0d5dd" stroke-width="2"/>
  <line x1="80" y1="290" x2="148.4" y2="102.1" stroke="#98a2b3" stroke-dasharray="4 3"/>
  <line x1="80" y1="290" x2="243.8" y2="175.3" stroke="#98a2b3" stroke-dasharray="4 3"/>
  <path d="M152,103.5 A200,200 0 0,1 239.5,171.5" fill="none" stroke="#2a6fdb" stroke-width="2.5" marker-end="url(#arB)"/>
  <circle cx="148.4" cy="102.1" r="6" fill="#e5484d"/>
  <circle cx="243.8" cy="175.3" r="6" fill="#2a6fdb"/>
  <circle cx="280" cy="290" r="6" fill="#12b76a"/>
  <circle cx="80" cy="90" r="5" fill="#98a2b3"/>
  <text x="158" y="92">当前时刻 t</text>
  <text x="258" y="168">跳到时刻 s</text>
  <text x="290" y="282">干净图像（角度为 0）</text>
  <text x="72" y="94" class="sm" text-anchor="end">纯噪声</text>
  <text x="430" y="95">一步 DDIM（η = 0）：</text>
  <text x="430" y="125" class="sm">1. 用网络在当前点估计噪声，并反推干净图像</text>
  <text x="430" y="150" class="sm">2. 两个估计都不变，只把角度从 t 转到 s</text>
  <text x="430" y="175" class="sm">3. 角度转到 0 就是干净图像</text>
  <text x="430" y="215" class="sm">网络估计得越准，这两个方向越稳定，</text>
  <text x="430" y="238" class="sm">一次可以转过的角度就越大。</text>
</svg>
<figcaption>示意图：横轴、纵轴分别是干净图像和噪声的方向，点到原点的距离始终为 1（信号系数和噪声系数的平方和为 1）。</figcaption>
</figure>

这和上一篇第 10 节 $$v$$-prediction 的"旋转"是同一个图：$$v$$ 就是这段圆弧的切线方向。

## 6. 实现

### 6.1 选哪些时间步

训练用了 $$T$$ 个时间步，采样时只挑其中 $$N$$ 个，从大到小排列。最简单的选法是在 $$[T-1, 0]$$（0-based 下标）上等间隔取 $$N$$ 个点：

```python
def ddim_timesteps(T, num_steps):
    # 从大到小的 0-based 下标，第一个一定是 T-1（对应纯噪声 x_T），最后一个是 0
    if not 1 <= num_steps <= T:
        raise ValueError("num_steps must be in [1, T]")
    return torch.linspace(T - 1, 0, num_steps).round().long().tolist()

ddim_timesteps(300, 5)    # [299, 224, 150, 75, 0]
ddim_timesteps(300, 50)   # [299, 293, 287, ..., 6, 0]
```

最后一个选中的时间步之后，再跳一次到"干净图像"（$$\bar\alpha = 1$$），所以总共调用 $$N$$ 次网络。

### 6.2 单步更新

```python
@torch.no_grad()
def ddim_step(model, x_t, t_index, prev_index, sched, eta=0.0, clip_x0=False):
    # t_index = t - 1；prev_index = s - 1，-1 表示跳到干净图像（ᾱ_0 = 1）
    t = torch.full((x_t.shape[0],), t_index, device=x_t.device, dtype=torch.long)   # [B]
    pred_noise = model(x_t, t)                                                       # ε_θ(x_t, t)
    abar_t = sched["alpha_bars"][t_index]                                            # ᾱ_t
    abar_s = sched["alpha_bars"][prev_index] if prev_index >= 0 else abar_t.new_tensor(1.0)  # ᾱ_s

    # x̂0 = (x_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t
    x0_hat = (x_t - torch.sqrt(1 - abar_t) * pred_noise) / torch.sqrt(abar_t)
    if clip_x0:
        # 可选：把 x̂0 裁到 [-1, 1]，并用裁剪后的 x̂0 反推 ε，保证两者仍然满足 x_t 的关系
        x0_hat = x0_hat.clamp(-1, 1)
        pred_noise = (x_t - torch.sqrt(abar_t) * x0_hat) / torch.sqrt(1 - abar_t)

    # σ² = η² · (1−ᾱ_s)/(1−ᾱ_t) · (1 − ᾱ_t/ᾱ_s)
    sigma2 = eta**2 * (1 - abar_s) / (1 - abar_t) * (1 - abar_t / abar_s)
    # √(1 − ᾱ_s − σ²)，clamp 防止浮点误差让根号里出现负数
    dir_coef = torch.sqrt(torch.clamp(1 - abar_s - sigma2, min=0.0))
    # x_s = √ᾱ_s · x̂0 + √(1−ᾱ_s−σ²) · ε_θ + σ · z
    return torch.sqrt(abar_s) * x0_hat + dir_coef * pred_noise + torch.sqrt(sigma2) * torch.randn_like(x_t)
```

| 公式 | 代码 | shape |
|---|---|---|
| $$\epsilon_\theta(x_t, t)$$ | `pred_noise` | `[B, C, H, W]` |
| $$\bar\alpha_t,\ \bar\alpha_s$$ | `abar_t`, `abar_s` | 标量 |
| $$\hat x_0$$ | `x0_hat` | `[B, C, H, W]` |
| $$\sigma_{t\to s}^2$$ | `sigma2` | 标量 |
| $$\sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}$$ | `dir_coef` | 标量 |
| $$z$$ | `torch.randn_like(x_t)` | `[B, C, H, W]` |
| $$x_s$$ | 返回值 | `[B, C, H, W]` |

### 6.3 完整采样循环

```python
@torch.no_grad()
def sample_ddim(model, shape, sched, device, num_steps=50, eta=0.0, x_T=None, clip_x0=False):
    model.eval()
    sched = {k: v.to(device) for k, v in sched.items()}
    steps = ddim_timesteps(sched["betas"].shape[0], num_steps)        # 例如 [299, 293, ..., 0]
    # x_T ~ N(0, I)；也可以传入固定的 x_T，η = 0 时同一个 x_T 一定得到同一张图
    x_t = torch.randn(shape, device=device) if x_T is None else x_T.to(device)
    for i, t_index in enumerate(steps):
        prev_index = steps[i + 1] if i + 1 < len(steps) else -1       # 最后一步跳到干净图像
        x_t = ddim_step(model, x_t, t_index, prev_index, sched, eta, clip_x0)
    return x_t                                                        # x_0
```

用 classifier-free guidance 时，只需把 `ddim_step` 里的 `model(x_t, t)` 换成上一篇第 9 节的 $$\epsilon_{\text{cfg}} = \epsilon_\theta(x_t, t, \varnothing) + w\,(\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing))$$，其余不变。

> **易错点**
>
> 1. **`linspace(...).long()` 会截断，不是四舍五入。** 例如 `linspace(299, 0, 4)` 是 `[299, 199.33, 99.67, 0]`，`.long()` 得到 `[299, 199, 99, 0]`，而四舍五入应该是 `[299, 199, 100, 0]`。截断会让所有中间时间步系统性地偏小，间隔也不均匀。先 `.round()` 再 `.long()`。
> 2. **最后要跳到 $$\bar\alpha = 1$$。** 常见的写法只在选中的时间步之间跳（`for i in range(len(steps) - 1)`），最后停在下标 0，也就是论文的 $$t = 1$$，返回的还是带一点噪声的 $$x_1$$。本例里 $$\bar\alpha_1 = 0.9999$$，看不出差别，但 $$\beta_1$$ 更大或步数更少时就会留下残余噪声。
> 3. **第一步必须从 $$T-1$$ 开始。** 采样的起点 $$x_T \sim \mathcal{N}(0, I)$$ 只在 $$t = T$$ 时成立。有的实现（例如 `arange(0, T, T // N)` 再反转）第一步是 $$T - T/N$$，等于把纯噪声当成噪声较少的 $$x_t$$，图像会偏灰、对比度不足。
> 4. **根号里的 `1 - abar_s - sigma2` 要先 `clamp`。** $$\eta \le 1$$ 时它理论上非负，但跳到 $$\bar\alpha_s = 1$$ 时恰好是 0，浮点误差可能让它变成极小的负数；有的实现还允许 $$\eta > 1$$，那时它真的会变负。`sqrt` 一旦得到 `nan`，会一路传到最后。`clamp(min=0)` 再开根号。
> 5. **裁剪 $$\hat x_0$$ 后要重新计算 $$\epsilon$$。** 只裁 $$\hat x_0$$、仍然用原来的 $$\epsilon_\theta$$ 算方向项，两项就对不上同一个 $$x_t$$ 了。`clip_x0` 默认关闭；像素空间、步数很少时打开会更稳，latent diffusion 一般不裁。
> 6. **步数太少时，最后一个选中的时间步是浪费的。** 等间隔选法总会选中下标 0，而从 0 跳到干净图像几乎什么都不做。例如 $$N = 2$$ 时选中的是 `[299, 0]`，实际只有第一步有用，效果和 $$N = 1$$ 一样（见 8.2）。改用"末端对齐"的选法（例如 $$N=2$$ 取 `[299, 149]`）可以缓解，$$N \ge 20$$ 时两种选法几乎没有区别。

### 6.4 总结：DDIM 采样的完整流程

| 步骤 | 公式 | 代码 |
|---|---|---|
| 1 | 选 $$N$$ 个从大到小的时间步 $$\tau_1 > \tau_2 > \dots > \tau_N$$ | `ddim_timesteps(T, N)` |
| 2 | 抽 $$x_{\tau_1} \sim \mathcal{N}(0, I)$$ | `torch.randn(shape)` |
| 3 | 对每一对相邻时间步 $$(t, s)$$：网络预测 $$\hat\epsilon = \epsilon_\theta(x_t, t)$$ | `model(x_t, t)` |
| 4 | 反推 $$\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon) / \sqrt{\bar\alpha_t}$$ | `ddim_step` 前半 |
| 5 | 重新混合 $$x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1-\bar\alpha_s-\sigma^2}\,\hat\epsilon + \sigma z$$（$$\eta = 0$$ 时 $$\sigma = 0$$） | `ddim_step` 后半 |
| 6 | 最后一步跳到 $$\bar\alpha = 1$$，输出 $$\hat x_0$$ | `sample_ddim` 末尾 |

和 DDPM 采样相比，只有两处不同：时间步是挑出来的一部分；每一步先估计 $$\hat x_0$$ 再按目标时间步的比例重新组合，而不是"退一步"。网络、调度表、训练过程完全不变。

## 7. DDIM 反演：从图像找回噪声

### 7.1 为什么可以反演

$$\eta = 0$$ 时，DDIM 是一个确定的映射 $$x_T \mapsto x_0$$，而且它是在解一个 ODE（第 5.2 节）。ODE 可以反着积分，所以给定一张**真实图像** $$x_0$$，也能找到一个 $$x_T$$，让 DDIM 从这个 $$x_T$$ 出发恰好生成 $$x_0$$。这叫 **DDIM inversion**，是很多图像编辑方法的基础：先把真实图反演成噪声，改一改条件（例如换文字提示），再用 DDIM 生成回来。 [[1]](#ref-1)

更新公式和采样完全一样，只是方向反过来，从小的时间步 $$s$$ 走到大的时间步 $$t$$：

$$
x_t = \sqrt{\bar\alpha_t}\,\hat x_0 + \sqrt{1-\bar\alpha_t}\;\epsilon_\theta(x_s, s),\qquad
\hat x_0 = \frac{x_s - \sqrt{1-\bar\alpha_s}\,\epsilon_\theta(x_s, s)}{\sqrt{\bar\alpha_s}}
$$

注意这里有一个近似：精确的反演需要终点处的 $$\epsilon_\theta(x_t, t)$$，但 $$x_t$$ 正是要求的东西，所以用起点的 $$\epsilon_\theta(x_s, s)$$ 代替。步长越小，两者越接近。

### 7.2 代码

```python
@torch.no_grad()
def ddim_invert(model, x0, sched, num_steps=50):
    model.eval()
    steps = ddim_timesteps(sched["betas"].shape[0], num_steps)[::-1]   # 0, ..., T-1（从小到大）
    seq = [-1] + steps                                                  # -1 表示干净图像（ᾱ = 1）
    x = x0
    for s_index, t_index in zip(seq[:-1], seq[1:]):
        # 用起点的 ε_θ(x_s, s) 近似；干净图像处没有对应的时间步，用下标 0
        s = torch.full((x.shape[0],), max(s_index, 0), device=x.device, dtype=torch.long)
        eps = model(x, s)                                               # ε_θ(x_s, s)
        abar_s = sched["alpha_bars"][s_index] if s_index >= 0 else x.new_tensor(1.0)
        abar_t = sched["alpha_bars"][t_index]
        x0_hat = (x - torch.sqrt(1 - abar_s) * eps) / torch.sqrt(abar_s)   # x̂0
        x = torch.sqrt(abar_t) * x0_hat + torch.sqrt(1 - abar_t) * eps     # x_t = √ᾱ_t x̂0 + √(1−ᾱ_t) ε
    return x                                                            # x_T

# 重建：x_T = ddim_invert(model, x0, sched, 50)
#       x0_rec = sample_ddim(model, x0.shape, sched, device, num_steps=50, x_T=x_T)
```

> **易错点**
>
> 1. **只有 $$\eta = 0$$ 才能反演。** $$\eta > 0$$ 时每步都有新的随机噪声，同一个 $$x_T$$ 每次生成的图都不同，反演就没有意义了。
> 2. **反演和生成要用同一组时间步、同一个条件。** 反演时用 50 步、生成时用 20 步，或者反演用类别 A、生成用类别 B，都不会还原原图（后者正是"编辑"想要的效果）。
> 3. **CFG 会破坏反演。** $$w > 1$$ 时 $$\epsilon_{\text{cfg}}$$ 不再是一个"一致"的噪声估计，误差在每一步被放大，重建会走样（8.5 的最后一行）。反演时用 $$w = 1$$（或者不加条件）更准；实际的编辑方法（如 null-text inversion）会专门处理这个问题。
> 4. **反演得到的 $$x_T$$ 不像标准高斯噪声。** 它常常还看得出原图的轮廓，统计量也不完全是 $$\mathcal{N}(0, I)$$。在它上面做插值、编辑时要记住它是"非典型"的噪声。

## 8. 实验

下面所有结果都用上一篇训练好的同一个模型（FashionMNIST，$$T = 300$$，带类别条件的 Tiny U-Net），没有任何重新训练。除特别说明外，都用 CFG $$w = 3$$、$$\eta = 0$$，并且同一行用同一个初始噪声 $$x_T$$。

### 8.1 步数：从 1 步到 300 步

每行一个类别（裤子、运动鞋、包、短靴、连衣裙、套头衫），同一行的所有 DDIM 列都从同一个 $$x_T$$ 出发：

![DDIM 不同步数](/deep-learning/assets/ddim/ddim_steps.png)

- **1 步**（以及 2 步，原因见易错点 6）：一步跳到底，得到的是网络对 $$x_0$$ 的"平均猜测"，轮廓对、细节糊，和 5.1 节的分析一致。
- **5 步**就基本成形，**10 步以后**和 300 步几乎看不出区别。同一行的图形状一致，说明它们收敛到的是同一条 ODE 轨迹的终点。
- **最后一列 DDPM** 用同一个 $$x_T$$ 却得到了不同的图：DDPM 每一步都加新的随机噪声，终点不由 $$x_T$$ 决定。

在这台机器的 CPU 上生成 6 张图：DDIM 10 步约 0.12 秒，50 步约 0.67 秒，DDPM 300 步约 4.0 秒。耗时和网络调用次数成正比（用了 CFG，每步调用两次网络）。

### 8.2 收敛速度

用 64 个不同的 $$x_T$$，把 300 步 DDIM 的结果当作"准确解"，计算 $$N$$ 步结果和它的均方误差：

![DDIM 收敛](/deep-learning/assets/ddim/ddim_convergence.png)

从 5 步开始，双对数坐标下大致是一条斜率约 $$-2$$ 的直线：步数增加 10 倍，平方误差下降约 100 倍（10 步 $$6.5 \times 10^{-3}$$，100 步 $$4.3 \times 10^{-5}$$）。这正是一阶 Euler 法的特征（误差和步长成正比，平方误差和步长的平方成正比），印证了 5.2 节的 ODE 解释。

1 步和 2 步误差相同，就是易错点 6 说的浪费。把时间步换成末端对齐的选法，少步时明显更好，步数多了以后两者一样：

| 步数 $$N$$ | 等间隔（含 0） | 平方误差 | 末端对齐 | 平方误差 |
|---|---|---|---|---|
| 2 | `[299, 0]` | 0.183 | `[299, 149]` | 0.062 |
| 3 | `[299, 150, 0]` | 0.060 | `[299, 199, 99]` | 0.035 |
| 5 | `[299, 224, 150, 75, 0]` | 0.023 | `[299, 239, ..., 59]` | 0.017 |
| 10 | `[299, 266, ..., 0]` | 0.0065 | `[299, 269, ..., 29]` | 0.0057 |
| 20 | `[299, 283, ..., 0]` | 0.0018 | `[299, 284, ..., 14]` | 0.0018 |

### 8.3 $$\eta$$：确定性与多样性

同一个 $$x_T$$、50 步，每个 $$\eta$$ 跑 5 次（每次随机项 $$z$$ 的种子不同）。上面三行是裤子，下面三行是运动鞋：

![DDIM eta](/deep-learning/assets/ddim/ddim_eta.png)

- $$\eta = 0$$：5 次结果完全相同。图像只由 $$x_T$$ 决定，多样性全部来自初始噪声。
- $$\eta = 0.5$$、$$1$$：每次都不同，$$x_T$$ 只决定大致的姿态，细节由每步的随机项决定。$$\eta$$ 越大，差异越大。

需要可复现、可反演、可插值时用 $$\eta = 0$$；想在同一个 $$x_T$$ 附近多要几种变化时调大 $$\eta$$。

### 8.4 噪声空间插值

$$\eta = 0$$ 时 $$x_T \mapsto x_0$$ 是一个连续的映射，所以在两个初始噪声之间插值，生成的图也会平滑过渡。每行固定一个类别（短靴、运动鞋、T 恤），列是插值系数 $$u$$：

![DDIM 插值](/deep-learning/assets/ddim/ddim_interp.png)

插值要用**球面插值（slerp）**，不能用线性插值：

$$
\operatorname{slerp}(a, b; u) = \frac{\sin((1-u)\theta)}{\sin\theta}\,a + \frac{\sin(u\theta)}{\sin\theta}\,b,\qquad \cos\theta = \frac{\langle a, b\rangle}{\lVert a\rVert\,\lVert b\rVert}
$$

高维标准高斯向量的长度几乎都在 $$\sqrt{d}$$ 附近（这里 $$d = 784$$，约 28）。两个独立噪声几乎正交，线性插值的中点 $$\tfrac12(a+b)$$ 长度只有约 $$\sqrt{d/2}$$，已经不像 $$\mathcal{N}(0, I)$$ 的样本了，生成的图会变灰、变糊。slerp 沿球面走，长度保持不变。

### 8.5 反演与重建

用测试集里的 8 张真实图像（模型训练时没见过），按各自的真实类别、$$w = 1$$ 做 DDIM 反演，再从得到的 $$x_T$$ 生成回来：

![DDIM 反演](/deep-learning/assets/ddim/ddim_inversion.png)

- 第 2 行是反演得到的 $$x_T$$（为了显示做了缩放）。它明显带着原图的轮廓，标准差约 0.6，而不是标准高斯的 1，正是易错点 4 说的"非典型噪声"。
- 50 步重建的平方误差约 $$3.6 \times 10^{-4}$$，300 步约 $$1.1 \times 10^{-5}$$，肉眼看不出差别。步数越多，"用起点的 $$\epsilon$$ 近似终点"的误差越小。
- 最后一行改用 $$w = 3$$ 反演和重建，平方误差升到 $$1.2 \times 10^{-2}$$，连衣裙（第 4 列）的下摆明显变形。

## 9. 和其他采样器的关系

| | DDPM | DDIM（$$\eta=0$$） | 高阶 ODE 采样器 |
|---|---|---|---|
| 训练 | 噪声预测 MSE | 同一个模型 | 同一个模型 |
| 反向过程 | 随机，马尔可夫 | 确定，非马尔可夫 | 确定 |
| 数学上相当于 | SDE，小步模拟 | ODE，一阶 Euler | ODE，二阶及以上（Heun、DPM-Solver、UniPC 等） |
| 常用步数 | $$T$$（几百到 1000） | 20～100 | 10～25 |
| 同一个 $$x_T$$ | 每次结果不同 | 结果固定，可反演 | 结果固定 |

Stable Diffusion 等库里的 `DDIMScheduler`、`EulerDiscreteScheduler`、`DPMSolverMultistepScheduler` 就是这张表里的不同列：U-Net 是同一个，换的只是怎么从 $$\epsilon_\theta$$ 一步步走到 $$x_0$$。理解了 DDIM 的"预测 $$\hat x_0$$，再重新加噪"和它的 ODE 解释，其余采样器都是在此基础上提高精度。

## 10. 自检

用一个"知道答案"的假网络（它对固定的 $$x_0$$ 返回精确的噪声）检查上面的推导和代码，几秒内跑完：

```python
T = 300
sched = make_beta_schedule(T)

class OracleNoise(torch.nn.Module):
    # 已知 x0，返回精确噪声 ε = (x_t − √ᾱ_t x0) / √(1−ᾱ_t)
    def __init__(self, x0, alpha_bars):
        super().__init__()
        self.x0, self.alpha_bars = x0, alpha_bars
    def forward(self, x_t, t):
        abar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return (x_t - torch.sqrt(abar) * self.x0) / torch.sqrt(1 - abar)

x0 = torch.rand(2, 1, 8, 8) * 2 - 1
oracle = OracleNoise(x0, sched["alpha_bars"])

# 1. 时间步：从 T-1 开始，没有重复
for n in [1, 2, 10, 50, 300]:
    steps = ddim_timesteps(T, n)
    assert steps[0] == T - 1 and len(set(steps)) == n

# 2. η = 1 且 s = t−1 时，σ² 就是 DDPM 的后验方差 β̃_t（4.4 节）
ab = sched["alpha_bars"]
for t in [1, 50, 299]:
    sigma2 = (1 - ab[t - 1]) / (1 - ab[t]) * (1 - ab[t] / ab[t - 1])
    assert torch.allclose(sigma2, sched["posterior_variance"][t], rtol=1e-4)

# 3. 噪声预测完美时，1 步、5 步、50 步都精确还原 x0（5.1 节）
for n in [1, 5, 50]:
    assert torch.allclose(sample_ddim(oracle, x0.shape, sched, "cpu", num_steps=n), x0, atol=1e-4)

# 4. η = 0 是确定性的，反演后再生成能还原 x0（第 7 节）
x_T = ddim_invert(oracle, x0, sched, num_steps=20)
a = sample_ddim(oracle, x0.shape, sched, "cpu", num_steps=20, x_T=x_T)
b = sample_ddim(oracle, x0.shape, sched, "cpu", num_steps=20, x_T=x_T)
assert torch.equal(a, b) and torch.allclose(a, x0, atol=1e-4)
```

## 11. 要点回顾

- **DDIM 为什么不用重新训练？** DDPM 的训练目标只依赖每个 $$t$$ 的边缘分布 $$q(x_t \mid x_0)$$。DDIM 构造了一个边缘分布相同、但能在任意两个时间步之间跳的过程，所以训练目标、网络都不变，只换采样器。
- **DDIM 的一步在做什么？** 先由 $$\epsilon_\theta$$ 反推 $$\hat x_0$$，再按时间步 $$s$$ 的比例把 $$\hat x_0$$ 和同一份预测噪声重新组合：$$x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1-\bar\alpha_s-\sigma^2}\,\epsilon_\theta + \sigma z$$。
- **为什么跳步还准？** $$\eta = 0$$ 时 DDIM 是 ODE $$d\bar x / d\lambda = \epsilon_\theta$$ 的 Euler 法，轨迹确定而平滑，误差随步长线性下降。实验中 10 步和 300 步已经几乎一样。
- **$$\eta$$ 控制什么？** 每步注入多少新噪声。$$\eta = 0$$ 确定性，同一个 $$x_T$$ 得到同一张图；$$\eta = 1$$、不跳步时就是 DDPM。
- **DDPM 和 DDIM 是什么关系？** DDPM 是 DDIM 在 $$\eta = 1$$、$$s = t-1$$ 时的特例。
- **DDIM 反演靠什么？** 确定性 ODE 可以反着积分。反演和生成要用同一组时间步和条件，$$w > 1$$ 的 CFG 会让反演变差。
- **实现时最容易错的地方：** 时间步要 `round` 而且从 $$T-1$$ 开始；最后要跳到 $$\bar\alpha = 1$$；$$\sigma$$ 用 $$\bar\alpha_t/\bar\alpha_s$$；方向项用预测噪声而不是新噪声；根号里先 `clamp`；裁剪 $$\hat x_0$$ 后重新计算 $$\epsilon$$。
