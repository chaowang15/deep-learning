---
date: 2026-06-01 00:00:00 -0800
tags: [Diffusion, Math]
keywords: >-
  Diffusion models, DDPM, forward process, reverse process, Markov chain, ELBO, KL divergence, posterior q(x_{t-1}|x_t,x_0), noise prediction, beta schedule, training loss, sampling.
summary: >-
  扩散模型（DDPM）的原理推导：前向过程如何逐步加噪并得到任意时刻的闭式解，反向过程为什么可以用高斯近似，ELBO 的两种分解，后验 q(x_{t-1}|x_t,x_0) 的均值和方差怎么推出来，以及训练目标为什么最后简化成预测噪声的 MSE。附 beta schedule 的选择、训练与采样算法、一个可运行的二维小例子和总结表。
layout: note
title: "Diffusion Models 扩散模型学习"
description: "Diffusion models and DDPM from first principles: forward process, closed form q(x_t|x_0), reverse process, two ELBO decompositions, posterior mean and variance, the simplified noise-prediction loss, beta schedules, training and sampling."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
</style>

# Diffusion Models 扩散模型学习

扩散模型（Diffusion Model）是目前图像、视频生成的主流方法，Stable Diffusion、DALL·E 2/3、Sora 等都建立在它之上。它的思路是：先定义一个**固定的**加噪过程，把真实图像一步步变成纯高斯噪声；再训练一个神经网络学习**反过来**的去噪过程。生成时从纯噪声出发，反复调用这个网络去噪，就得到一张新图像。

本文是扩散模型的原理篇，围绕四个问题展开：

1. **前向加噪怎么定义？** 为什么每一步是 $$\sqrt{\alpha_t}\,x_{t-1}$$ 加噪声，以及为什么可以一步从 $$x_0$$ 算出任意 $$x_t$$。
2. **反向去噪为什么可以学？** 为什么反向的每一步可以近似成高斯分布，以及在已知 $$x_0$$ 时它的均值和方差的闭式解。
3. **训练目标从哪来？** 从最大似然出发推导 ELBO，再一步步化简成"预测噪声的 MSE"。
4. **怎么训练和采样？** 两个算法各自只有几行，最后给出一个可以直接运行的二维小例子。

完整的 PyTorch 实现（U-Net、训练、采样、在 FashionMNIST 上的实验）见 [DDPM 从零实现](/deep-learning/ddpm-from-scratch/)；少步采样见 [DDIM](/deep-learning/ddim/)。

## 参考资料

本文主要参考 DDPM 论文 [1] 和 Lilian Weng 的综述博客 [3]，推导的细节参考 [4][5]；正文中用 [n] 标注具体出处。

- <a id="ref-1"></a>**[1] DDPM 主论文**：Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)。前向过程的参数化、简化损失 $$L_\text{simple}$$、训练和采样算法都来自这篇。
- <a id="ref-2"></a>**[2] 扩散模型的起点**：Sohl-Dickstein et al. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML 2015. [arxiv.org/abs/1503.03585](https://arxiv.org/abs/1503.03585)。最早提出"逐步加噪、学习逆过程"的扩散概率模型（DPM）。
- <a id="ref-3"></a>**[3] 写得最清楚的综述博客**：Lilian Weng. [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) 2021。推导完整，第 2、3 节的大部分公式可以和它对照。
- <a id="ref-4"></a>**[4] 推导最细的长文**：Calvin Luo. *Understanding Diffusion Models: A Unified Perspective*. 2022. [arxiv.org/abs/2208.11970](https://arxiv.org/abs/2208.11970)。ELBO 的两种分解（3.2 节）按这篇的思路展开。
- <a id="ref-5"></a>**[5] 入门教程**：LearnOpenCV. [An In-Depth Guide to Denoising Diffusion Probabilistic Models](https://learnopencv.com/denoising-diffusion-probabilistic-models/)；以及 Kemal Erdem. [Step by Step Visual Introduction to Diffusion Models](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models)。前者讲动机和整体流程，后者每一步都配了图和最终公式，适合先看。
- <a id="ref-6"></a>**[6] 中文推导**：张振虎. [扩散概率模型](https://www.zhangzhenhu.com/aigc/%E6%89%A9%E6%95%A3%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B.html)。中文里公式推导最详细的一篇，第 1 节的部分表述参考了它。

建议的阅读顺序：先看 [5] 建立直观，再读本文或 [3] 跟一遍推导，[4] 用来查推导细节，最后读 [1] 原文和 [DDPM 从零实现](/deep-learning/ddpm-from-scratch/) 的代码。

## 背景：生成模型与需要的数学工具

本节介绍后文用到的基础知识和基本概念。

### 生成模型要解决什么问题

一张 $$256 \times 256$$ 的彩色图像可以看成 $$256 \times 256 \times 3 \approx 20$$ 万维空间中的一个点。在这个巨大的空间里，绝大多数点只是杂乱的噪声，"看起来像向日葵"的图像只占极小的一块区域。

![向日葵图像示例](/deep-learning/assets/diffusion-study/Untitled.png)

我们可以把真实图像看成从某个分布 $$p_\text{data}(x)$$ 中采样得到的样本。如果知道这个分布的概率密度函数并且能从中采样，就能直接生成新图像；问题在于这个分布未知，而且极其复杂，无法写出解析式。**生成模型**要做的，就是只根据一批样本 $$x^{(1)}, \dots, x^{(N)}$$，学到一个能采样的模型 $$p_\theta(x) \approx p_\text{data}(x)$$。

常见的几类生成模型思路不同（图源 [[3]](#ref-3)）：

![几类生成模型的对比](/deep-learning/assets/diffusion-study/Untitled%201.png)

- **GAN**：生成器把噪声 $$z$$ 直接映射成图像，用判别器对抗训练，见 [GAN 篇](/deep-learning/generative-models-2-gan/)。
- **VAE**：编码器把 $$x$$ 压缩成隐变量 $$z$$，解码器再还原，训练时最大化证据下界 ELBO，见 [VAE 篇](/deep-learning/generative-models-1-vae/)。
- **Flow**：用可逆变换把数据分布变成简单分布。
- **扩散模型**：把"数据 → 噪声"拆成很多个小步，每一步只加一点噪声；网络只需学会"去掉一点噪声"这件简单的事，反复做很多次就能从噪声得到数据。

扩散模型和 VAE 的结构很像，都可以看成"编码到隐变量、再解码回来"，ELBO 的推导思路也相同。区别在于：扩散模型的"编码器"（加噪过程）是固定的、没有参数；隐变量 $$x_1, \dots, x_T$$ 和原图尺寸相同；编码和解码都分成很多步。

### 需要的概率工具

下面几条在推导里会反复用到，更详细的推导见 [数学基础篇](/deep-learning/math-foundation/)。

**1. 高斯分布与重参数化。** 若 $$x \sim \mathcal{N}(\mu, \sigma^2 I)$$，则可以写成

$$
x = \mu + \sigma \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

这样"从分布中采样"就变成了"对一个标准噪声做确定的线性变换"。

**2. 独立高斯相加仍是高斯，方差相加。** 若 $$\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, I)$$ 独立，则

$$
a\,\epsilon_1 + b\,\epsilon_2 \sim \mathcal{N}\big(0, (a^2 + b^2) I\big),
$$

也就是说它等价于一个噪声 $$\sqrt{a^2 + b^2}\,\epsilon$$。

**3. 方差的缩放。** $$\mathrm{Var}(a x) = a^2 \mathrm{Var}(x)$$；独立变量之和的方差等于方差之和。

**4. 贝叶斯公式。** $$p(a \mid b) = \dfrac{p(b \mid a)\,p(a)}{p(b)}$$，加上条件 $$c$$ 时同样成立：$$p(a \mid b, c) = \dfrac{p(b \mid a, c)\,p(a \mid c)}{p(b \mid c)}$$。

**5. 马尔可夫链。** 若 $$x_t$$ 只依赖 $$x_{t-1}$$，联合分布可以写成相邻两步条件分布的连乘：$$q(x_{0:T}) = q(x_0)\prod_{t=1}^{T} q(x_t \mid x_{t-1})$$。这里 $$x_{0:T}$$ 是 $$x_0, x_1, \dots, x_T$$ 的简写。

**6. KL 散度。** 衡量两个分布 $$q$$、$$p$$ 的差异，$$\mathrm{KL}(q \,\Vert\, p) = \mathbb{E}_{q}\big[\log \frac{q}{p}\big] \ge 0$$，两者相等时为 0。两个高斯之间的 KL 有闭式解；当方差固定时，它只和两个均值之差的平方成正比。

### 一个数值例子：一个像素的加噪

把一张图归一化到 $$[-1, 1]$$ 后，看其中一个像素值 $$x_0 = 0.8$$。设每一步 $$\beta = 0.1$$（真实模型里 $$\beta_t$$ 要小得多，这里只为看清变化），每一步做

$$
x_t = \sqrt{1 - \beta}\, x_{t-1} + \sqrt{\beta}\, \epsilon_t .
$$

如果某次采样的噪声依次是 $$\epsilon_1 = 0.5,\ \epsilon_2 = -1.2,\ \epsilon_3 = 0.3$$，则

| 步 | 计算 | 结果 |
|---|---|---|
| $$x_1$$ | $$0.949 \times 0.8 + 0.316 \times 0.5$$ | 0.917 |
| $$x_2$$ | $$0.949 \times 0.917 - 0.316 \times 1.2$$ | 0.491 |
| $$x_3$$ | $$0.949 \times 0.491 + 0.316 \times 0.3$$ | 0.561 |

原信号 0.8 前面的系数每步乘以 0.949，三步后剩 $$0.949^3 \approx 0.854$$；噪声的方差逐步累积，三步后是 $$1 - 0.9^3 = 0.271$$。步数足够多时原信号的系数趋近 0，噪声方差趋近 1，$$x_T$$ 就变成标准高斯噪声。第 2 节会说明，这个结果可以一步算出，不用真的循环。

## 1. 整体结构：前向过程与反向过程

扩散模型由两个方向相反的马尔可夫链组成 [[1]](#ref-1)：

- **前向过程（forward / diffusion process）**：$$x_0 \to x_1 \to \dots \to x_T$$，每一步给上一步加一点高斯噪声，记为 $$q(x_t \mid x_{t-1})$$。这一过程是人为固定的，**没有可训练参数**。
- **反向过程（reverse process）**：$$x_T \to x_{T-1} \to \dots \to x_0$$，每一步去掉一点噪声，记为 $$p_\theta(x_{t-1} \mid x_t)$$，由神经网络学习。生成新图像就是运行反向过程，所以它也叫**采样过程**。

![前向过程与反向过程](/deep-learning/assets/diffusion-study/image.png)

图中下方一行是前向过程 $$q$$，上方一行是反向过程 $$p$$。两个过程的联合分布分别是

$$
q(x_{0:T}) = q(x_0) \prod_{t=1}^{T} q(x_t \mid x_{t-1}), \qquad
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t),
$$

其中 $$q(x_0)$$ 就是真实数据分布，$$p(x_T) = \mathcal{N}(0, I)$$ 是反向过程的起点。下面的推导里 $$t$$ 取 $$1, 2, \dots, T$$（论文约定），$$t = 0$$ 是原图；代码里时间步通常是 $$0, \dots, T-1$$，差一位，见 [DDPM 从零实现](/deep-learning/ddpm-from-scratch/) 的记号表。

## 2. 前向过程：逐步加噪

### 2.1 单步定义

DDPM 把每一步定义为一个线性高斯变换 [[1]](#ref-1)：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\ \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t I\big),
$$

用重参数化写成采样的形式就是

$$
x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, I).
$$

- $$\beta_t \in (0, 1)$$ 是第 $$t$$ 步加入噪声的方差，是人为设定的超参数，称为 **noise schedule**。通常随 $$t$$ 增大：前期加噪少，后期加噪多。DDPM 用 $$T = 1000$$，$$\beta_t$$ 从 $$10^{-4}$$ 线性增加到 $$0.02$$。
- 均值 $$\sqrt{1 - \beta_t}\, x_{t-1}$$ 是上一步按比例缩小，方差 $$\beta_t$$ 和 $$x_{t-1}$$ 无关。

为了书写简洁，定义

$$
\alpha_t = 1 - \beta_t, \qquad \bar\alpha_t = \prod_{i=1}^{t} \alpha_i .
$$

$$\bar\alpha_t$$ 是累计乘积，从 $$\bar\alpha_1 \approx 1$$ 单调减小到 $$\bar\alpha_T \approx 0$$。

### 2.2 任意时刻的闭式解

前向过程有一个重要性质：**不需要一步步迭代，可以从 $$x_0$$ 直接算出任意 $$x_t$$** [[1]](#ref-1)[[3]](#ref-3)。把单步公式反复代入：

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1 - \alpha_t}\, \epsilon_{t} \\
&= \sqrt{\alpha_t}\big(\sqrt{\alpha_{t-1}}\, x_{t-2} + \sqrt{1 - \alpha_{t-1}}\, \epsilon_{t-1}\big) + \sqrt{1 - \alpha_t}\, \epsilon_{t} \\
&= \sqrt{\alpha_t \alpha_{t-1}}\, x_{t-2} + \underbrace{\sqrt{\alpha_t (1 - \alpha_{t-1})}\, \epsilon_{t-1} + \sqrt{1 - \alpha_t}\, \epsilon_{t}}_{\text{两个独立高斯之和}} \\
&= \sqrt{\alpha_t \alpha_{t-1}}\, x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}}\, \bar\epsilon \\
&= \dots \\
&= \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon .
\end{aligned}
$$

第 3 到 4 行用了背景里的性质 2：两个噪声项的方差之和是 $$\alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t \alpha_{t-1}$$，合并成一个噪声 $$\bar\epsilon$$。继续展开到 $$x_0$$，系数的规律不变。于是

$$
\boxed{\ q(x_t \mid x_0) = \mathcal{N}\big(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t) I\big), \qquad x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon\ }
$$

这个式子是训练的基础：训练时随机抽一个 $$t$$，直接用它构造 $$x_t$$，不用把前面 $$t-1$$ 步都跑一遍。$$\sqrt{\bar\alpha_t}$$ 叫信号系数，$$\sqrt{1 - \bar\alpha_t}$$ 叫噪声系数，两者的平方和恒为 1。

下图是前向过程的几步示意（图源 [[5]](#ref-5)，Kemal Erdem）：

![前向过程示意](/deep-learning/assets/diffusion-study/image%201.png)

<div class="note-callout" markdown="1">
**前向过程小结**

$$
\begin{aligned}
q(x_t \mid x_{t-1}) &= \mathcal{N}\big(x_t;\ \sqrt{\alpha_t}\, x_{t-1},\ (1 - \alpha_t) I\big) \\
q(x_t \mid x_0) &= \mathcal{N}\big(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t) I\big) \\
x_t &= \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I) \\
\alpha_t &= 1 - \beta_t, \qquad \bar\alpha_t = \textstyle\prod_{i=1}^{t} \alpha_i
\end{aligned}
$$

</div>

### 2.3 为什么是这样的系数

**为什么 $$x_{t-1}$$ 前面要乘一个小于 1 的系数？** 我们希望 $$T$$ 步之后 $$x_T$$ 和原图无关，并且服从标准高斯 $$\mathcal{N}(0, I)$$，这样生成时才能从标准高斯出发。只有每一步都把上一步缩小一点，原图的成分 $$\sqrt{\bar\alpha_t}\, x_0$$ 才会随 $$t$$ 增大逐渐趋于 0。

**为什么系数是 $$\sqrt{1 - \beta_t}$$，而不是 $$1 - \beta_t$$？** 为了让方差保持不变（variance preserving）。假设 $$x_{t-1}$$ 的方差是 1（数据归一化后近似成立），则

$$
\mathrm{Var}(x_t) = (1 - \beta_t)\,\mathrm{Var}(x_{t-1}) + \beta_t = 1 - \beta_t + \beta_t = 1 .
$$

每一步方差都是 1，最后 $$x_T$$ 正好是方差为 1 的标准高斯。如果去掉根号，写成 $$x_t = (1 - \beta_t)\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t$$，方差就变成 $$(1 - \beta_t)^2 + \beta_t < 1$$，每一步都在缩小。用 DDPM 的线性 schedule 实际算一下，1000 步后方差只剩约 0.50，$$x_T$$ 不再是标准高斯，和采样起点 $$\mathcal{N}(0, I)$$ 对不上。同时信号系数从 $$\sqrt{\bar\alpha_t}$$ 变成了 $$\bar\alpha_t$$：例如 $$\bar\alpha_t = 0.01$$ 时，带根号的信号系数是 0.1，不带根号只有 0.01，原图的信息消失得过快，后面大部分时间步都只剩噪声，可学习的信号很少。

### 2.4 noise schedule 的选择

DDPM 的 $$\beta_t$$ 是在 $$[10^{-4}, 0.02]$$ 上线性均匀取 $$T = 1000$$ 个值 [[1]](#ref-1)：

```python
import torch

T = 1000
betas = torch.linspace(1e-4, 0.02, T)          # beta_1 ... beta_T（代码下标 0 ... T-1）
alphas = 1.0 - betas                            # alpha_t
alpha_bars = torch.cumprod(alphas, dim=0)       # bar alpha_t
```

Stable Diffusion 用的是另一种写法，称为 scaled linear：先在 $$\sqrt{\beta}$$ 上线性取值，再平方。

```python
beta_start, beta_end, T = 0.00085, 0.012, 1000
betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T, dtype=torch.float64) ** 2
```

两端的值不变，中间的 $$\beta_t$$ 是 $$t$$ 的二次函数：前期增长慢、后期增长快，也就是前期每一步加的噪声更少，原图的细节保留得更久。下图对比了这两种取法在 $$[0.00085, 0.012]$$ 上的曲线：

![线性与 scaled linear 的 beta 曲线对比](/deep-learning/assets/diffusion-study/Untitled%202.png)

后来 Improved DDPM 提出了 cosine schedule，让 $$\bar\alpha_t$$ 按余弦曲线下降，避免线性 schedule 在最后几百步几乎全是噪声的问题；Stable Diffusion 3、FLUX 等则改用 [Flow Matching](/deep-learning/flow-matching/) 的线性插值路径。schedule 的选择会影响训练效率和画质，但不改变本文的推导。

## 3. 反向过程与训练目标

### 3.1 反向过程用高斯近似

生成时要从 $$x_T$$ 一步步得到 $$x_0$$，需要知道真实的反向分布 $$q(x_{t-1} \mid x_t)$$。按贝叶斯公式它等于 $$q(x_t \mid x_{t-1})\, q(x_{t-1}) / q(x_t)$$，其中 $$q(x_{t-1})$$、$$q(x_t)$$ 要对整个数据分布积分，无法计算。

所以我们用一个神经网络来近似它。DDPM 假设每一步反向分布也是高斯 [[1]](#ref-1)：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\big).
$$

这个假设的依据是：当每一步加的噪声足够小（$$\beta_t$$ 很小）时，真实的反向分布 $$q(x_{t-1} \mid x_t)$$ 也非常接近高斯 [[2]](#ref-2)。这也是扩散模型要分成很多小步的原因：步子小，每一步的反向分布才简单。

下图来自 [[2]](#ref-2)，在二维数据上展示了前向过程（上行，螺旋逐渐变成噪声）、学到的反向过程（中行）以及网络学到的漂移方向（下行）：

![二维螺旋数据上的前向与反向轨迹](/deep-learning/assets/diffusion-study/image%203.png)

### 3.2 证据下界 ELBO

训练的目标是最大化数据的对数似然 $$\log p_\theta(x_0)$$。只有 $$x_0$$ 是观测到的，$$x_{1:T}$$ 都是隐变量，所以

$$
p_\theta(x_0) = \int p_\theta(x_{0:T})\, dx_{1:T} ,
$$

这个积分无法直接计算。和 [VAE](/deep-learning/generative-models-1-vae/) 一样，引入前向过程 $$q(x_{1:T} \mid x_0)$$ 作为隐变量的分布，用 Jensen 不等式得到一个下界 [[4]](#ref-4)：

$$
\begin{aligned}
\log p_\theta(x_0)
&= \log \int q(x_{1:T} \mid x_0)\, \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\, dx_{1:T} \\
&= \log \mathbb{E}_{q(x_{1:T} \mid x_0)}\Big[\frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\Big] \\
&\ge \mathbb{E}_{q(x_{1:T} \mid x_0)}\Big[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\Big] \;=:\; \text{ELBO} .
\end{aligned}
$$

第 3 行用了 Jensen 不等式：$$\log$$ 是凹函数，所以 $$\log \mathbb{E}[X] \ge \mathbb{E}[\log X]$$。最大化 ELBO 就间接地最大化了似然。接下来把 ELBO 展开成若干项，有两种分解方式。

#### 第一种分解：按前向的相邻两步

把两个联合分布按马尔可夫链展开：

$$
\begin{aligned}
\text{ELBO}
&= \mathbb{E}_{q}\Big[\log \frac{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)}{\prod_{t=1}^{T} q(x_t \mid x_{t-1})}\Big] \\
&= \mathbb{E}_{q}\Big[\log \frac{p(x_T)\, p_\theta(x_0 \mid x_1) \prod_{t=1}^{T-1} p_\theta(x_t \mid x_{t+1})}{q(x_T \mid x_{T-1}) \prod_{t=1}^{T-1} q(x_t \mid x_{t-1})}\Big] \\
&= \mathbb{E}_{q}\big[\log p_\theta(x_0 \mid x_1)\big]
 + \mathbb{E}_{q}\Big[\log \frac{p(x_T)}{q(x_T \mid x_{T-1})}\Big]
 + \sum_{t=1}^{T-1} \mathbb{E}_{q}\Big[\log \frac{p_\theta(x_t \mid x_{t+1})}{q(x_t \mid x_{t-1})}\Big] \\
&= \underbrace{\mathbb{E}_{q(x_1 \mid x_0)}\big[\log p_\theta(x_0 \mid x_1)\big]}_{\text{重建项}}
 - \underbrace{\mathbb{E}_{q(x_{T-1} \mid x_0)}\big[\mathrm{KL}\big(q(x_T \mid x_{T-1}) \,\Vert\, p(x_T)\big)\big]}_{\text{先验匹配项}} \\
&\quad - \sum_{t=1}^{T-1} \underbrace{\mathbb{E}_{q(x_{t-1}, x_{t+1} \mid x_0)}\big[\mathrm{KL}\big(q(x_t \mid x_{t-1}) \,\Vert\, p_\theta(x_t \mid x_{t+1})\big)\big]}_{\text{一致性项}} .
\end{aligned}
$$

第 2 行把分子分母的连乘各拆出一项，并把分子的下标从 $$t-1$$ 换成 $$t$$，使两个连乘的项一一对应。第 3 行把 $$\log$$ 的乘积拆成求和，并交换求和与期望。

第 4 行把每个期望的范围从 $$q(x_{1:T} \mid x_0)$$ 缩小到真正用到的变量上。以第一项为例，$$\log p_\theta(x_0 \mid x_1)$$ 只依赖 $$x_1$$，对其他变量积分时它们的概率之和为 1，可以直接积掉（边缘化）：

$$
\mathbb{E}_{q(x_{1:T} \mid x_0)}\big[\log p_\theta(x_0 \mid x_1)\big]
= \int q(x_1 \mid x_0) \log p_\theta(x_0 \mid x_1) \Big[\int q(x_{2:T} \mid x_1)\, dx_{2:T}\Big] dx_1
= \mathbb{E}_{q(x_1 \mid x_0)}\big[\log p_\theta(x_0 \mid x_1)\big].
$$

其余两项同理，并把 $$\mathbb{E}\big[\log \frac{p}{q}\big]$$ 写成 $$-\mathrm{KL}(q \,\Vert\, p)$$。

这个分解很直观，但有一个实际问题：一致性项要对 $$x_{t-1}$$ 和 $$x_{t+1}$$ **两个**随机变量求期望，用蒙特卡洛估计时方差很大，训练不稳定 [[4]](#ref-4)。DDPM 实际用的是下面第二种分解。

#### 第二种分解：以 $$x_0$$ 为条件

关键一步是利用马尔可夫性加上 $$x_0$$ 作为条件，再用贝叶斯公式把前向的单步"翻转"过来：

$$
q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)\; q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)} .
$$

把它代入 ELBO 中 $$t \ge 2$$ 的项，分母上的 $$q(x_t \mid x_0) / q(x_{t-1} \mid x_0)$$ 在连乘中逐项相消，只剩 $$q(x_T \mid x_0) / q(x_1 \mid x_0)$$。整理后得到 [[1]](#ref-1)[[4]](#ref-4)：

$$
\text{ELBO}
= \underbrace{\mathbb{E}_{q(x_1 \mid x_0)}\big[\log p_\theta(x_0 \mid x_1)\big]}_{-L_0\text{：重建项}}
- \underbrace{\mathrm{KL}\big(q(x_T \mid x_0) \,\Vert\, p(x_T)\big)}_{L_T\text{：先验匹配项}}
- \sum_{t=2}^{T} \underbrace{\mathbb{E}_{q(x_t \mid x_0)}\Big[\mathrm{KL}\big(q(x_{t-1} \mid x_t, x_0) \,\Vert\, p_\theta(x_{t-1} \mid x_t)\big)\Big]}_{L_{t-1}\text{：去噪匹配项}} .
$$

三项的含义：

- **$$L_T$$** 比较前向过程的终点和标准高斯，两者都没有可训练参数，是常数，训练时忽略。
- **$$L_{t-1}$$** 要求网络的反向一步 $$p_\theta(x_{t-1} \mid x_t)$$ 去逼近"已知 $$x_0$$ 时的真实反向一步" $$q(x_{t-1} \mid x_t, x_0)$$。现在每一项只对**一个**随机变量 $$x_t$$ 求期望，而 $$x_t$$ 可以用 2.2 节的闭式解一步采样。这是训练的主体。
- **$$L_0$$** 是最后一步的重建项，DDPM 用一个离散的解码器处理；简化损失里它和 $$L_{t-1}$$ 合并成同一个形式。

$$q(x_{t-1} \mid x_t)$$ 无法计算，但加上条件 $$x_0$$ 后的 $$q(x_{t-1} \mid x_t, x_0)$$ 可以算出闭式解，这正是第二种分解可行的原因。下一节推导它。

### 3.3 后验 $$q(x_{t-1} \mid x_t, x_0)$$ 的均值和方差

由上面的贝叶斯公式，右边三个分布都已知而且都是高斯：

$$
q(x_{t-1} \mid x_t, x_0) = q(x_t \mid x_{t-1})\, \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} ,
$$

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\, x_{t-1},\ \beta_t I), \quad
q(x_{t-1} \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}}\, x_0,\ (1 - \bar\alpha_{t-1}) I), \quad
q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t) I).
$$

高斯密度 $$\propto \exp\big(-\frac{(x - \mu)^2}{2\sigma^2}\big)$$，三者相乘除就是指数相加减（$$e^a e^b = e^{a+b}$$）。只保留和 $$x_{t-1}$$ 有关的项 [[3]](#ref-3)：

$$
\begin{aligned}
q(x_{t-1} \mid x_t, x_0)
&\propto \exp\Big(-\frac{1}{2}\Big(\frac{(x_t - \sqrt{\alpha_t}\, x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}}\, x_0)^2}{1 - \bar\alpha_{t-1}} - \frac{(x_t - \sqrt{\bar\alpha_t}\, x_0)^2}{1 - \bar\alpha_t}\Big)\Big) \\
&= \exp\Big(-\frac{1}{2}\Big(\frac{x_t^2 - 2\sqrt{\alpha_t}\, x_t {\color{blue}{x_{t-1}}} + \alpha_t {\color{red}{x_{t-1}^2}}}{\beta_t} + \frac{ {\color{red}{x_{t-1}^2}} - 2\sqrt{\bar\alpha_{t-1}}\, x_0 {\color{blue}{x_{t-1}}} + \bar\alpha_{t-1} x_0^2}{1 - \bar\alpha_{t-1}} - \frac{(x_t - \sqrt{\bar\alpha_t}\, x_0)^2}{1 - \bar\alpha_t}\Big)\Big) \\
&= \exp\Big(-\frac{1}{2}\Big({\color{red}{\Big(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar\alpha_{t-1}}\Big)}}\, x_{t-1}^2 - {\color{blue}{\Big(\frac{2\sqrt{\alpha_t}}{\beta_t}\, x_t + \frac{2\sqrt{\bar\alpha_{t-1}}}{1 - \bar\alpha_{t-1}}\, x_0\Big)}}\, x_{t-1} + C(x_t, x_0)\Big)\Big)
\end{aligned}
$$

其中 $$C(x_t, x_0)$$ 和 $$x_{t-1}$$ 无关。这是 $$x_{t-1}$$ 的二次式，所以后验仍是高斯，记为 $$\mathcal{N}(x_{t-1};\ \tilde\mu_t,\ \tilde\beta_t I)$$。把标准形式 $$\exp\big(-\frac{1}{2}\big(\frac{1}{\tilde\beta_t} x_{t-1}^2 - \frac{2\tilde\mu_t}{\tilde\beta_t} x_{t-1} + \dots\big)\big)$$ 和上式对比系数：

**方差**（对比 $$x_{t-1}^2$$ 的系数，并用 $$\alpha_t + \beta_t = 1$$）：

$$
\tilde\beta_t = 1 \Big/ \Big(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar\alpha_{t-1}}\Big)
= \frac{\beta_t (1 - \bar\alpha_{t-1})}{\alpha_t - \bar\alpha_t + \beta_t}
= \color{blue}{\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\, \beta_t} .
$$

**均值**（对比 $$x_{t-1}$$ 的系数）：

$$
\begin{aligned}
\tilde\mu_t(x_t, x_0)
&= \Big(\frac{\sqrt{\alpha_t}}{\beta_t}\, x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1 - \bar\alpha_{t-1}}\, x_0\Big)\, \tilde\beta_t \\
&= \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\, x_t + \frac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1 - \bar\alpha_t}\, x_0 .
\end{aligned}
$$

均值是 $$x_t$$ 和 $$x_0$$ 的加权组合。再用 2.2 节的闭式解把 $$x_0$$ 用 $$x_t$$ 和噪声表示：

$$
x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\big(x_t - \sqrt{1 - \bar\alpha_t}\, \epsilon\big),
$$

代入后化简得到

$$
\tilde\mu_t = \color{red}{\frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon\Big)} .
$$

这里的 $$\epsilon$$ 就是从 $$x_0$$ 加噪得到 $$x_t$$ 时用的那个噪声。

<div class="note-callout" markdown="1">
**后验小结**

$$
\begin{aligned}
q(x_{t-1} \mid x_t, x_0) &= \mathcal{N}\big(x_{t-1};\ \tilde\mu_t,\ \tilde\beta_t I\big) \\
\tilde\mu_t &= \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon\Big)
= \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\, x_t + \frac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1 - \bar\alpha_t}\, x_0 \\
\tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\, \beta_t
\end{aligned}
$$

</div>

### 3.4 从 KL 到"预测噪声"

$$L_{t-1}$$ 是两个高斯之间的 KL。DDPM 把反向过程的方差固定为常数 $$\Sigma_\theta = \sigma_t^2 I$$（不学习），这时 KL 只剩均值之差 [[1]](#ref-1)：

$$
L_{t-1} = \mathbb{E}_{q}\Big[\frac{1}{2\sigma_t^2}\big\Vert \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\big\Vert^2\Big] + C .
$$

网络可以直接输出 $$\mu_\theta$$，但 3.3 节已经说明真实均值的形式是 $$\frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon\big)$$，其中 $$x_t$$ 是网络的输入、已知，唯一未知的是噪声 $$\epsilon$$。所以让网络 $$\epsilon_\theta(x_t, t)$$ 预测噪声，再按同样的形式组装均值：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t)\Big).
$$

代入后 $$x_t$$ 相消，均值之差只剩噪声之差：

$$
L_{t-1} = \mathbb{E}_{x_0, \epsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\, \alpha_t\, (1 - \bar\alpha_t)}\big\Vert \epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon,\ t\big)\big\Vert^2\Big] + C .
$$

DDPM 进一步发现，**去掉前面随 $$t$$ 变化的权重、并对 $$t$$ 均匀采样**，训练更简单，生成质量反而更好 [[1]](#ref-1)。这就是最终的训练目标：

$$
\boxed{\ L_\text{simple} = \mathbb{E}_{t \sim \mathcal{U}\{1, T\},\ x_0,\ \epsilon \sim \mathcal{N}(0, I)}\Big[\big\Vert \epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon,\ t\big)\big\Vert^2\Big]\ }
$$

也就是**加噪时用的真实噪声和网络预测的噪声之间的均方误差**。注意它是差的平方（MSE），不是差本身。去掉权重相当于给噪声较大的时间步更多权重，让网络把精力放在更难的去噪任务上。

关于方差 $$\sigma_t^2$$ 的取值，DDPM 试了 $$\sigma_t^2 = \beta_t$$ 和 $$\sigma_t^2 = \tilde\beta_t$$ 两种，结果相近 [[1]](#ref-1)。论文 3.2 节指出，前者在 $$x_0 \sim \mathcal{N}(0, I)$$ 时最优，后者在 $$x_0$$ 固定为一个点时最优，真实数据介于两者之间。Improved DDPM 后来让网络学习在两者之间插值。

除了预测噪声 $$\epsilon$$，网络也可以预测 $$x_0$$ 或速度 $$v = \sqrt{\bar\alpha_t}\, \epsilon - \sqrt{1 - \bar\alpha_t}\, x_0$$，三者可以互相换算，见 [DDPM 从零实现](/deep-learning/ddpm-from-scratch/) 第 10 节。

## 4. 训练与采样

### 4.1 两个算法

DDPM 论文的算法 1 和算法 2 [[1]](#ref-1)：

**训练**（重复直到收敛）：

1. 从数据集取一张图 $$x_0$$；
2. 随机取时间步 $$t \sim \mathcal{U}\{1, \dots, T\}$$；
3. 采样噪声 $$\epsilon \sim \mathcal{N}(0, I)$$；
4. 构造 $$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$$；
5. 对 $$\big\Vert \epsilon - \epsilon_\theta(x_t, t)\big\Vert^2$$ 做一步梯度下降。

**采样**：

1. 采样 $$x_T \sim \mathcal{N}(0, I)$$；
2. 对 $$t = T, T-1, \dots, 1$$：
   - 若 $$t > 1$$，采样 $$z \sim \mathcal{N}(0, I)$$，否则 $$z = 0$$；
   - $$x_{t-1} = \dfrac{1}{\sqrt{\alpha_t}}\Big(x_t - \dfrac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t)\Big) + \sigma_t z$$；
3. 返回 $$x_0$$。

采样的每一步就是"用网络算出后验均值 $$\mu_\theta$$，再加上方差为 $$\sigma_t^2$$ 的随机噪声"。最后一步（$$t = 1$$）不再加噪声，直接取均值：

$$
x_0 = \frac{1}{\sqrt{\alpha_1}}\Big(x_1 - \frac{\beta_1}{\sqrt{1 - \bar\alpha_1}}\, \epsilon_\theta(x_1, 1)\Big).
$$

下图是反向过程的几步示意（图源 [[5]](#ref-5)，Kemal Erdem），每一步网络预测出噪声，从当前图像中减去其中一小部分，再加回一点随机噪声：

![反向过程示意](/deep-learning/assets/diffusion-study/image%204.png)

**每一步去掉多少噪声？** 网络预测的 $$\epsilon_\theta$$ 是从 $$x_0$$ 到 $$x_t$$ 累积的**全部**噪声，但每一步只减去它的一小部分，系数是 $$\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar\alpha_t}}$$。用 DDPM 的线性 schedule 计算，这个系数在 $$t = 1000$$ 时约 0.020，在 $$t = 500$$ 时约 0.011，在 $$t = 100$$ 到 $$t = 10$$ 之间约 0.006 到 0.007，在 $$t = 1$$ 时约 0.010。也就是每一步只去掉约 1% 到 2% 的噪声估计，所以需要上千步；直接 $$x_t - \epsilon_\theta$$ 并不会得到 $$x_0$$，因为 $$x_t$$ 里的噪声乘了系数 $$\sqrt{1 - \bar\alpha_t}$$，信号也乘了 $$\sqrt{\bar\alpha_t}$$。从 $$x_t$$ 一步估计 $$x_0$$ 的正确公式是 3.3 节的 $$\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\, \epsilon_\theta) / \sqrt{\bar\alpha_t}$$，[DDIM](/deep-learning/ddim/) 正是利用这个估计实现了跳步采样。

### 4.2 一个可运行的小例子

下面在二维数据（单位圆上的点）上完整地训练并采样一个扩散模型，用一个小 MLP 代替 U-Net，CPU 上约 15 秒。代码和上面两个算法逐行对应。

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)

# 数据：单位圆上的点，x0 形状 (N, 2)
def sample_data(n):
    theta = torch.rand(n) * 2 * math.pi
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

# noise schedule（代码下标 0..T-1 对应公式里的 t = 1..T）
T = 200
betas = torch.linspace(1e-4, 0.05, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# 噪声预测网络 eps_theta(x_t, t)：输入 x_t 和归一化的 t，输出与 x_t 同形状的噪声
class EpsMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x_t, t):
        t_in = (t.float() / T).unsqueeze(1)          # (B, 1)
        return self.net(torch.cat([x_t, t_in], dim=1))

model = EpsMLP()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)

# 训练（算法 1）
for step in range(4000):
    x0 = sample_data(512)                              # 1. 取数据
    t = torch.randint(0, T, (512,))                    # 2. t ~ U{1..T}（代码 0..T-1）
    eps = torch.randn_like(x0)                         # 3. eps ~ N(0, I)
    ab = alpha_bars[t].unsqueeze(1)                    # (B, 1)，按 batch 广播
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps       # 4. 闭式加噪
    loss = ((eps - model(x_t, t)) ** 2).mean()         # 5. L_simple
    opt.zero_grad()
    loss.backward()
    opt.step()

# 采样（算法 2），sigma_t^2 取 beta_t
@torch.no_grad()
def sample(n):
    x = torch.randn(n, 2)                              # x_T ~ N(0, I)
    for i in reversed(range(T)):                       # t = T..1
        t = torch.full((n,), i)
        eps_hat = model(x, t)
        mean = (x - betas[i] / (1 - alpha_bars[i]).sqrt() * eps_hat) / alphas[i].sqrt()
        z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)   # 最后一步不加噪声
        x = mean + betas[i].sqrt() * z
    return x

samples = sample(2000)
radius = samples.norm(dim=1)
print(f"final loss {loss.item():.3f}, sample radius mean {radius.mean():.3f}, std {radius.std():.3f}")
```

真实数据的半径恒为 1。训练后采样点的平均半径接近 1（一次运行的结果：平均半径 0.999，标准差 0.031），说明模型从纯噪声中还原出了圆环分布。

<blockquote markdown="1">
**易错点**

- **时间步下标差一位**：公式里 $$t = 1, \dots, T$$，代码里是 $$0, \dots, T-1$$；`alpha_bars[i]` 对应公式的 $$\bar\alpha_{i+1}$$。采样循环里"最后一步不加噪声"对应的是代码 `i == 0`。
- **广播**：`alpha_bars[t]` 形状是 `(B,)`，要 `unsqueeze` 到能和 `x0` 相乘的形状；图像数据是 `(B, 1, 1, 1)`。
- **损失是 MSE**：$$\Vert \epsilon - \epsilon_\theta \Vert^2$$ 取平均，不是 $$\epsilon - \epsilon_\theta$$ 本身。
- **数据范围**：图像要先归一化到 $$[-1, 1]$$，和 $$\mathcal{N}(0, I)$$ 的尺度一致；生成后再映射回 $$[0, 1]$$ 或 $$[0, 255]$$。
- **采样系数**：均值里噪声的系数是 $$\beta_t / \sqrt{1 - \bar\alpha_t}$$，分母是累计乘积 $$\bar\alpha_t$$，外面再除以 $$\sqrt{\alpha_t}$$（单步的 $$\alpha_t$$），两个 $$\alpha$$ 容易写混。

</blockquote>

## 5. 总结

| 步骤 | 公式 | 说明 |
|---|---|---|
| noise schedule | $$\beta_t$$；$$\alpha_t = 1 - \beta_t$$；$$\bar\alpha_t = \prod_{i \le t} \alpha_i$$ | 人为设定，DDPM 线性 $$10^{-4} \to 0.02$$，$$T = 1000$$ |
| 前向单步 | $$x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t$$ | 方差保持为 1 |
| 前向闭式解 | $$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$$ | 训练时一步得到任意 $$x_t$$ |
| 后验 | $$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde\mu_t, \tilde\beta_t I)$$ | 已知 $$x_0$$ 时反向一步的真值 |
| 后验方差 | $$\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\, \beta_t$$ | 采样时可取 $$\sigma_t^2 = \tilde\beta_t$$ 或 $$\beta_t$$ |
| 反向均值 | $$\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t)\big)$$ | 网络只预测噪声 |
| 训练目标 | $$\mathbb{E}\Vert \epsilon - \epsilon_\theta(x_t, t)\Vert^2$$ | ELBO 去掉权重后的 $$L_\text{simple}$$ |
| 采样一步 | $$x_{t-1} = \mu_\theta + \sigma_t z$$，$$t = 1$$ 时 $$z = 0$$ | 从 $$x_T \sim \mathcal{N}(0, I)$$ 循环 $$T$$ 步 |

## 6. 要点回顾

- **扩散模型做什么？** 把"噪声 → 数据"拆成很多小步，网络每一步只学去掉一点噪声。
- **前向过程为什么要 $$\sqrt{1 - \beta_t}$$？** 保持方差为 1，使 $$x_T$$ 正好是标准高斯，同时原图信息不会过早消失。
- **为什么能一步得到 $$x_t$$？** 每步的噪声都是独立高斯，可以合并成一个，得到 $$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$$。
- **为什么反向过程可以用高斯？** 每步噪声很小时，真实的反向一步近似高斯；已知 $$x_0$$ 时，后验 $$q(x_{t-1} \mid x_t, x_0)$$ 严格是高斯，有闭式解。
- **训练目标怎么来的？** 最大化 ELBO；按 $$x_0$$ 条件分解后，主体是一系列"网络反向一步 vs 后验"的 KL，化简成噪声的 MSE，再去掉权重得到 $$L_\text{simple}$$。
- **为什么采样需要很多步？** 每一步只减去约 1% 到 2% 的噪声估计并重新加一点随机噪声；少步采样需要 [DDIM](/deep-learning/ddim/) 这样的确定性采样器，或 [Flow Matching](/deep-learning/flow-matching/) 这样更直的路径。
