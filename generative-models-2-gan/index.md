---
date: 2026-01-01 00:00:00 -0800
keywords: >-
  GAN, minimax objective, optimal discriminator, Jensen-Shannon divergence, non-saturating loss, mode collapse, DCGAN, conditional GAN, WGAN, WGAN-GP, spectral normalization, hinge loss, PatchGAN, Pix2Pix, CycleGAN, StyleGAN, VQGAN.
summary: >-
  从两人博弈的直观出发，完整推导 GAN：价值函数、最优判别器、全局最优时生成分布等于数据分布（JS 散度）、为什么原始生成器 loss 会梯度消失以及 non-saturating loss 的梯度分析，并讨论模式坍缩、不收敛等训练难点。代码部分逐行解释 MLP GAN 和 DCGAN 的实现与易错点，附 FashionMNIST 和二维 8 高斯上的实验。最后介绍 cGAN、WGAN / WGAN-GP、谱归一化与 hinge loss、Pix2Pix / PatchGAN / CycleGAN、StyleGAN 和 VQGAN，并给出 WGAN-GP 和 VQGAN 训练一步的简单实现。
layout: note
title: "GAN：原理、推导与实现"
description: "GAN from the minimax objective to the optimal discriminator and JS divergence, the non-saturating loss, training pitfalls, MLP GAN and DCGAN code with experiments, and variants: cGAN, WGAN-GP, spectral norm, PatchGAN, StyleGAN and VQGAN."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig svg .lane{font-size:14px;font-weight:600}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
.sd-img{margin:1.2em 0;text-align:center}
.sd-img img{max-width:100%;height:auto;border:1px solid #eaecf0;border-radius:6px}
.sd-img figcaption{color:#667085;font-size:.9em;margin-top:.4em}
</style>

# GAN：原理、推导与实现

GAN（Generative Adversarial Network，生成对抗网络）[[1]](#ref-1) 用两个网络的对抗来学习生成数据：生成器（Generator）负责"造假"，判别器（Discriminator）负责"鉴别"，两者交替训练、互相推动，最终生成器造出的数据让判别器也分辨不出真假。和 [VAE](/deep-learning/generative-models-1-vae/) 不同，GAN 不写出任何概率密度，也不最大化似然，而是通过一个两人博弈直接学会"采样"。

本文先介绍 GAN 的出发点（第 1 节），然后完整推导它的目标函数、最优判别器和全局最优解，并分析训练中的梯度问题和常见难点（第 2 节）；第 3 节给出 MLP GAN 和 DCGAN 的代码和实验；第 4 节介绍主要的后续变体，包括 WGAN、Pix2Pix、StyleGAN 和 VQGAN。需要的前置知识：

| 内容 | 用在哪里 | 详细讲解 |
|---|---|---|
| 生成模型的分类（显式密度 vs 隐式） | 第 1、2 节的出发点 | [生成模型概览](/deep-learning/generative-models/) |
| 交叉熵、KL 散度 | 第 2 节的推导 | [数学基础](/deep-learning/math-foundation/) |
| 卷积、转置卷积、BatchNorm | 第 3 节 DCGAN | [数学基础](/deep-learning/math-foundation/) 中的 Transposed Convolution 部分 |
| VQ-VAE | 第 4.6 节 VQGAN | [VAE 与 VQ-VAE](/deep-learning/generative-models-1-vae/) 第 6 节 |

## 参考资料

本文的推导主要参考 GAN 原论文 [1]、Goodfellow 的 NIPS 2016 教程 [2] 和 Lilian Weng 的博客 [4]，代码参考 PyTorch-GAN [8] 和 PyTorch 官方 DCGAN 教程 [9]。正文中用 [n] 标注具体出处。

- <a id="ref-1"></a>**[1] GAN 原论文**：Goodfellow et al. *Generative Adversarial Nets*. NeurIPS 2014. [arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)。价值函数、最优判别器（Proposition 1）、全局最优（Theorem 1）和训练算法都出自这篇。
- <a id="ref-2"></a>**[2] 作者本人的教程**：Goodfellow. *NIPS 2016 Tutorial: Generative Adversarial Networks*. [arxiv.org/abs/1701.00160](https://arxiv.org/abs/1701.00160)。non-saturating loss、不收敛的例子、模式坍缩都讲得很清楚。
- <a id="ref-3"></a>**[3] Stanford CS231n Lecture 12**：[视频（从 GAN 部分开始）](https://www.youtube.com/watch?v=5WoItGTWV54&t=1592s)。第 1、2 节的几张图来自这份课件。
- <a id="ref-4"></a>**[4] 推导完整的博客**：Lilian Weng. *From GAN to WGAN*. [lilianweng.github.io/posts/2017-08-20-gan/](https://lilianweng.github.io/posts/2017-08-20-gan/)。中文版可参考 [GAN 原理及公式推导](https://www.cnblogs.com/jins-note/p/10848248.html)。
- <a id="ref-5"></a>**[5] 入门博客**：*GANs N' Roses*. [towardsdatascience.com](https://towardsdatascience.com/gans-n-roses-c6652d513260)。框架和例子比较直观，但没有数学部分；第 1 节的示意图来自这里。
- <a id="ref-6"></a>**[6] DCGAN**：Radford, Metz, Chintala. *Unsupervised Representation Learning with Deep Convolutional GANs*. ICLR 2016. [arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)。卷积 GAN 的架构准则。
- <a id="ref-7"></a>**[7] 训练不稳定的理论分析**：Arjovsky, Bottou. *Towards Principled Methods for Training GANs*. ICLR 2017. [arxiv.org/abs/1701.04862](https://arxiv.org/abs/1701.04862)。
- <a id="ref-8"></a>**[8] 代码合集**：eriklindernoren, [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)。几乎所有常见 GAN 都有单文件实现，非常适合初学者；第 3.2 节的代码来自其中的 [gan.py](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py)。
- <a id="ref-9"></a>**[9] PyTorch 官方 DCGAN 教程**：[DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)。强烈推荐，每一部分都有贴合代码的解释，使用 CelebA 数据集。
- <a id="ref-10"></a>**[10] 课程教材**：University of Amsterdam, [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/)，内容详细，也收录在 PyTorch Lightning 的教程中。
- <a id="ref-11"></a>**[11] 条件 GAN**：Mirza, Osindero. *Conditional Generative Adversarial Nets*. 2014. [arxiv.org/abs/1411.1784](https://arxiv.org/abs/1411.1784)。
- <a id="ref-12"></a>**[12] WGAN**：Arjovsky, Chintala, Bottou. *Wasserstein GAN*. ICML 2017. [arxiv.org/abs/1701.07875](https://arxiv.org/abs/1701.07875)。
- <a id="ref-13"></a>**[13] WGAN-GP**：Gulrajani et al. *Improved Training of Wasserstein GANs*. NeurIPS 2017. [arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)。第 3.5 节 8 高斯实验的设置参考了这篇。
- <a id="ref-14"></a>**[14] 谱归一化**：Miyato et al. *Spectral Normalization for Generative Adversarial Networks*. ICLR 2018. [arxiv.org/abs/1802.05957](https://arxiv.org/abs/1802.05957)。其他 loss：LSGAN [arxiv.org/abs/1611.04076](https://arxiv.org/abs/1611.04076)；R1 正则 [arxiv.org/abs/1801.04406](https://arxiv.org/abs/1801.04406)。
- <a id="ref-15"></a>**[15] Pix2Pix 和 CycleGAN**：Isola et al. *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR 2017. [arxiv.org/abs/1611.07004](https://arxiv.org/abs/1611.07004)；Zhu et al. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ICCV 2017. [arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593)。
- <a id="ref-16"></a>**[16] StyleGAN**：Karras, Laine, Aila. *A Style-Based Generator Architecture for GANs*. CVPR 2019. [arxiv.org/abs/1812.04948](https://arxiv.org/abs/1812.04948)。之前的 ProGAN：[arxiv.org/abs/1710.10196](https://arxiv.org/abs/1710.10196)。
- <a id="ref-17"></a>**[17] VQGAN**：Esser, Rombach, Ommer. *Taming Transformers for High-Resolution Image Synthesis*. CVPR 2021. [arxiv.org/abs/2012.09841](https://arxiv.org/abs/2012.09841)，官方代码 [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)。自适应权重 λ 的实现见其中的 `vqperceptual.py`。
- <a id="ref-18"></a>**[18] 评价指标 FID**：Heusel et al. *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS 2017. [arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)。
- <a id="ref-19"></a>**[19] 资源合集**：[the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)（收集了非常全的 GAN 论文）；[gans-awesome-applications](https://github.com/nashory/gans-awesome-applications)（GAN 的有趣应用）；[GAN Playground](https://reiinakano.github.io/gan-playground)（在浏览器里在线训练 GAN，可以自定义网络层，支持 MNIST 和 CIFAR10）。

## 0. 全文路线与符号约定

| 节 | 内容 |
|---|---|
| 1 | 出发点：两人博弈，生成器和判别器各做什么 |
| 2 | 推导：价值函数、最优判别器、JS 散度与全局最优、生成器 loss 的梯度、训练难点 |
| 3 | 代码：架构技巧、MLP GAN、完整训练循环、DCGAN、两组实验 |
| 4 | 变体：cGAN、WGAN / WGAN-GP、其他 loss 与谱归一化、Pix2Pix / CycleGAN、StyleGAN、VQGAN |
| 5 | GAN、VAE、扩散模型的对比；要点回顾 |

**符号约定：**

| 符号 | 含义 | 代码 | shape（第 3 节） |
|---|---|---|---|
| $$x \sim p_{\text{data}}$$ | 真实数据 | `x` / `real_imgs` | `[B, 1, 28, 28]`，范围 $$[-1, 1]$$ |
| $$z \sim p_z = \mathcal{N}(0, I)$$ | 噪声（latent） | `z` | `[B, 100]` |
| $$G(z)$$，参数 $$\theta_g$$ | 生成器 | `G` / `generator` | 输出同 $$x$$ |
| $$p_g$$ | 生成器输出的分布：$$z \sim p_z$$ 时 $$G(z)$$ 的分布 | — | — |
| $$D(x) \in (0, 1)$$，参数 $$\theta_d$$ | 判别器：$$x$$ 是真实数据的概率 | `torch.sigmoid(D(x))` | `[B, 1]` |
| $$a(x)$$ | 判别器的 logit，$$D(x) = \sigma(a(x))$$ | `D(x)`（代码里网络直接输出 logit） | `[B, 1]` |
| $$V(D, G)$$ | 价值函数（value function） | — | 标量 |

注意代码里的 `D(x)` 输出的是 logit，经过 sigmoid 才是公式中的概率 $$D(x)$$，第 3.3 节会解释为什么这样写。

## 1. 背景和初衷

GAN 的核心是一个两人博弈（two-player game）的流程 [[5]](#ref-5)，包含生成器（Generator）和判别器（Discriminator）：

- 生成器的作用是生成尽可能和真实数据一样的数据，用来"欺骗"判别器；
- 判别器的作用是尽可能正确地区分真实数据和生成器生成的数据：把真实数据判为真，把生成器的数据判为假。

![生成器和判别器的两人博弈](/deep-learning/assets/generative-models/Untitled%2030.png)

GAN 的训练中，生成器和判别器交替训练，两者的能力不断提升。最终期望生成器能生成以假乱真的结果，即判别器也难以区分的结果。

![判别器把真实数据判为 1、生成数据判为 0](/deep-learning/assets/generative-models/Untitled%2031.png)

生成器的输入是一个随机向量，输出一张图。它的结构和 AutoEncoder 中的 Decoder 几乎一样，区别在于训练方式：Decoder 靠重建 loss 训练，生成器则完全靠判别器给出的信号训练，它从来不直接看真实数据。

![生成器把随机向量映射成图像](/deep-learning/assets/generative-models/508e1ea2-330a-44d1-a859-c9a40e35088a.png)

一个常用的类比是造假币的人和警察：造假币的人（生成器）不断改进工艺，警察（判别器）不断提高鉴别能力；警察每次识破假币，造假者就知道该往哪个方向改进。当警察只能靠猜（判对的概率是 50%）时，假币就和真币没有区别了。第 2 节会把这个类比变成严格的数学结论：在理想情况下，这个博弈的平衡点正是 $$p_g = p_{\text{data}}$$、$$D(x) = \frac12$$。

## 2. GAN 原理和推导

### 2.1 目标函数

从建模方式上看，[生成模型概览](/deep-learning/generative-models/) 中介绍的 FVBN 和 [VAE](/deep-learning/generative-models-1-vae/) 都在优化一个显式的密度函数（FVBN 精确计算似然，VAE 优化似然的下界）。GAN 的出发点不同：它不定义任何密度函数，而是通过两人博弈，直接学习"从随机噪声生成像训练数据一样的样本" [[3]](#ref-3)。

![显式密度模型与 GAN 的区别](/deep-learning/assets/generative-models/Untitled%2032.png)

具体来说：从随机噪声 $$z$$ 出发，生成器生成假数据 $$G(z)$$；判别器是一个二分类器，要把 $$G(z)$$ 分类为假（0），把真实数据 $$x$$ 分类为真（1）。

判别器的训练目标就是普通的二分类交叉熵。对真实样本（标签 1），交叉熵是 $$-\log D(x)$$；对生成样本（标签 0），交叉熵是 $$-\log(1 - D(G(z)))$$。判别器要最小化交叉熵，等价于最大化

<span id="eq-gan-1"></span>

$$
V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p_z}\big[\log\big(1 - D(G(z))\big)\big] \tag{1}
$$

而生成器希望判别器犯错，即让 $$V$$ 尽可能小。于是 GAN 是一个**极小极大（minimax）博弈** [[1]](#ref-1)：

<span id="eq-gan-2"></span>

$$
\min_{G}\ \max_{D}\ V(D, G) \tag{2}
$$

![CS231n 中的 minimax 目标函数](/deep-learning/assets/generative-models/Untitled%2033.png)

目标函数包含判别器和生成器的两个相反的需求：

- 对于判别器，它要**最大化** $$V$$：让真实数据的判别结果 $$D(x)$$ 接近 1，同时让生成数据的判别结果 $$D(G(z))$$ 接近 0。即把真实数据判为真、把虚假数据判为假，这样才是好的分类器；
- 对于生成器，它要**最小化** $$V$$。生成器只影响第二项，所以它要让 $$D(G(z))$$ 接近 1，即让判别器把生成的数据判为真实。这正是生成器的最终目标：生成非常真实、判别器无法区分的数据。

标准的 GAN 训练是交替进行的：

1. 固定生成器，对判别器做梯度**上升**，最大化 $$V$$ 的两项之和；
2. 固定判别器，对生成器做梯度**下降**，最小化 $$V$$ 中只和生成器有关的第二项 $$\mathbb{E}_z\log(1 - D(G(z)))$$。

接下来的 2.2 到 2.4 节回答两个问题：这个博弈的解是什么？为什么解是 $$p_g = p_{\text{data}}$$？2.5 节再回到实际训练，解释为什么第 2 步要换一个写法。

### 2.2 最优判别器

先固定生成器 $$G$$，求使 $$V(D, G)$$ 最大的判别器 $$D^*$$。

第一步，把 <a href="#eq-gan-1">Eq. (1)</a> 中对 $$z$$ 的期望改写成对 $$x$$ 的期望。生成器把 $$z \sim p_z$$ 映射成 $$G(z)$$，记 $$G(z)$$ 的分布为 $$p_g$$，那么对任意函数 $$f$$ 有 $$\mathbb{E}_{z \sim p_z}[f(G(z))] = \mathbb{E}_{x \sim p_g}[f(x)]$$（这只是换了一个记号：先采 $$z$$ 再算 $$G(z)$$，就是从 $$p_g$$ 中采一个 $$x$$）。于是

$$
\begin{aligned}
V(D, G) &= \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{x \sim p_g}\big[\log(1 - D(x))\big] \\
&= \int_x \Big[\, p_{\text{data}}(x)\log D(x) + p_g(x)\log\big(1 - D(x)\big) \Big]\, dx
\end{aligned}
$$

第二步，逐点最大化。判别器 $$D$$ 可以是任意函数，它在每个 $$x$$ 处的取值互不影响，所以只要让积分号里的被积函数在每个 $$x$$ 处都最大即可。固定一个 $$x$$，记 $$a = p_{\text{data}}(x)$$、$$b = p_g(x)$$、$$y = D(x) \in (0, 1)$$，要最大化

$$
f(y) = a \log y + b \log(1 - y)
$$

求导并令其为 0：

$$
f'(y) = \frac{a}{y} - \frac{b}{1 - y} = 0 \quad\Longrightarrow\quad y = \frac{a}{a + b}
$$

二阶导 $$f''(y) = -\frac{a}{y^2} - \frac{b}{(1-y)^2} < 0$$，所以这是最大值点。因此，**固定 $$G$$ 时的最优判别器**为（论文 Proposition 1）[[1]](#ref-1)：

<span id="eq-gan-3"></span>

$$
D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \tag{3}
$$

它的含义很直观：在 $$x$$ 处，真实数据的密度占两者之和的比例，就是"这个 $$x$$ 来自真实数据"的概率。如果某处只有真实数据（$$p_g = 0$$），$$D^* = 1$$；只有生成数据，$$D^* = 0$$；两者密度相同，$$D^* = \frac12$$。

### 2.3 全局最优：生成器的目标是 JS 散度

把最优判别器代回价值函数，得到生成器真正在最小化的目标 $$C(G) = \max_D V(D, G) = V(D^*_G, G)$$：

$$
C(G) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\log\frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log\frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)}\right]
$$

引入两个分布的平均 $$m = \frac{p_{\text{data}} + p_g}{2}$$。两个对数里的分母都写成 $$2m$$，各拆出一个 $$\log\frac12$$：

$$
\begin{aligned}
C(G) &= \mathbb{E}_{x \sim p_{\text{data}}}\left[\log\frac{p_{\text{data}}(x)}{m(x)}\right] + \log\frac{1}{2} + \mathbb{E}_{x \sim p_g}\left[\log\frac{p_g(x)}{m(x)}\right] + \log\frac12\\
&= -\log 4 + KL\big(p_{\text{data}} \,\Vert\, m\big) + KL\big(p_g \,\Vert\, m\big)
\end{aligned}
$$

两个 KL 之和正好是 **Jensen-Shannon 散度**（JS 散度）的 2 倍。JS 散度的定义是 $$JSD(p \Vert q) = \frac12 KL(p \Vert m) + \frac12 KL(q \Vert m)$$，它是对称的，取值在 $$[0, \log 2]$$ 之间，当且仅当 $$p = q$$ 时为 0。于是（论文 Theorem 1）[[1]](#ref-1)：

<span id="eq-gan-4"></span>

$$
C(G) = -\log 4 + 2\cdot JSD\big(p_{\text{data}} \,\Vert\, p_g\big) \tag{4}
$$

结论：

- $$C(G) \ge -\log 4$$，当且仅当 $$p_g = p_{\text{data}}$$ 时取到最小值 $$-\log 4$$；
- 此时最优判别器 $$D^* = \frac{1}{2}$$ 处处成立，判别器只能靠猜；
- 也就是说，**在判别器总是最优的前提下，训练生成器就是在最小化生成分布和数据分布之间的 JS 散度**。这就是 GAN 能学会数据分布的理论依据。

下面用一个离散分布的小例子验证 <a href="#eq-gan-3">Eq. (3)</a> 和 <a href="#eq-gan-4">Eq. (4)</a>：

```python
import numpy as np

p_data = np.array([0.5, 0.3, 0.2, 0.0])     # 真实分布，4 个取值
p_g    = np.array([0.1, 0.2, 0.3, 0.4])     # 生成分布

D = p_data / (p_data + p_g)                 # Eq. (3)：最优判别器
print(D)                                    # [0.833 0.6 0.4 0.]

def xlogy(x, y):                            # 约定 0·log(0) = 0
    return np.where(x > 0, x * np.log(np.where(x > 0, y, 1)), 0.0)

V = xlogy(p_data, D).sum() + xlogy(p_g, 1 - D).sum()        # V(D*, G)
m = (p_data + p_g) / 2
jsd = 0.5 * xlogy(p_data, p_data / m).sum() + 0.5 * xlogy(p_g, p_g / m).sum()
print(V, -np.log(4) + 2 * jsd)              # -0.9433 -0.9433，两者相等
```

### 2.4 理论保证和实际训练的差距

论文还证明了（Proposition 2）：如果 $$G$$ 和 $$D$$ 的容量足够大，并且每一步都先把 $$D$$ 训练到最优，再沿 $$C(G)$$ 的梯度更新 $$p_g$$，那么 $$p_g$$ 会收敛到 $$p_{\text{data}}$$。原因是 $$C(G)$$ 作为 $$p_g$$ 的函数是凸的。

但这个保证有三个前提，在实际训练中都不成立：

1. **优化的是网络参数 $$\theta_g$$，而不是分布 $$p_g$$ 本身。** $$C$$ 对 $$p_g$$ 凸，不代表对 $$\theta_g$$ 凸；
2. **判别器不会被训练到最优。** 实际中每次只对 $$D$$ 做 $$k$$ 步（常常 $$k = 1$$）梯度更新；
3. **期望是用 mini-batch 估计的**，梯度带噪声。

所以 2.3 节的结论只说明了"博弈的平衡点在哪里"，并不保证梯度下降一定能走到那里。2.5、2.7 节讨论由此带来的实际问题。

### 2.5 生成器 loss：为什么要换成 non-saturating

![原始生成器 loss 在训练初期梯度很小](/deep-learning/assets/generative-models/Untitled%2034.png)

按 <a href="#eq-gan-2">Eq. (2)</a>，生成器要最小化 $$\log(1 - D(G(z)))$$。上图右下角是这个函数关于 $$D(G(z))$$ 的曲线：$$D(G(z))$$ 较小时曲线很平，梯度很小；只有 $$D(G(z))$$ 接近 1 时梯度才变大。而训练初期，生成器输出的几乎是噪声，判别器很容易识破，$$D(G(z))$$ 接近 0，正好落在梯度很小的区域：**样本越差，生成器得到的学习信号反而越弱**。这个现象叫做饱和（saturating）。

改进方法是把生成器的目标从"最小化 $$\log(1 - D(G(z)))$$"换成"最大化 $$\log D(G(z))$$"，称为 **non-saturating loss** [[1]](#ref-1) [[2]](#ref-2)。两者的目标一致，都是让 $$D(G(z))$$ 变大，但 $$\log D(G(z))$$ 在 $$D(G(z))$$ 接近 0 时梯度很大，如下图右下角的绿色曲线。

![non-saturating loss 在训练初期梯度很大](/deep-learning/assets/generative-models/Untitled%2035.png)

**用 logit 看得更清楚。** 判别器最后一层通常是 sigmoid：$$D = \sigma(a)$$，$$a$$ 是 logit。利用 $$\sigma'(a) = \sigma(a)(1 - \sigma(a))$$，两种生成器 loss 对 logit 的梯度分别是：

$$
\frac{\partial}{\partial a}\log\big(1 - \sigma(a)\big) = -\sigma(a) = -D, \qquad\qquad \frac{\partial}{\partial a}\Big(-\log \sigma(a)\Big) = -\big(1 - \sigma(a)\big) = -(1 - D)
$$

| | 生成器最小化 | 对 logit 的梯度 | $$D(G(z)) \approx 0$$ 时（训练初期） | $$D(G(z)) \approx 1$$ 时 |
|---|---|---|---|---|
| 原始（saturating） | $$\log(1 - D(G(z)))$$ | $$-D$$ | $$\approx 0$$，几乎学不动 | $$\approx -1$$ |
| non-saturating | $$-\log D(G(z))$$ | $$-(1 - D)$$ | $$\approx -1$$，学习信号最强 | $$\approx 0$$ |

梯度的方向相同（都是让 logit 增大），大小正好反过来。non-saturating 版本在样本差时给出最强的信号，这正是我们想要的，所以它是实践中的标准做法。在代码里，它就是"把生成样本的标签设为 1，计算 BCE"（3.2 节）。

需要注意的是，换了生成器 loss 之后，2.3 节"最小化 JS 散度"的结论不再严格成立。Arjovsky 和 Bottou [[7]](#ref-7) 证明，在判别器最优时，non-saturating loss 的梯度对应于最小化 $$KL(p_g \Vert p_{\text{data}}) - 2\,JSD(p_{\text{data}} \Vert p_g)$$。其中 $$KL(p_g \Vert p_{\text{data}})$$ 对"生成了数据中没有的样本"惩罚很重，对"漏掉了数据中的某些模式"惩罚很轻。这是 GAN 容易出现模式坍缩（2.7 节）的一个解释。

### 2.6 总结：GAN 的训练算法

![GAN 训练算法（论文 Algorithm 1）](/deep-learning/assets/generative-models/Untitled%2036.png)

| 步骤 | 公式 | 代码（3.3 节） |
|---|---|---|
| 0 | 初始化 $$G$$、$$D$$；数据归一化到 $$[-1, 1]$$ | `weights_init`；`x / 127.5 - 1` |
| 1 | 采 $$m$$ 个噪声 $$z^{(i)} \sim \mathcal{N}(0, I)$$ 和 $$m$$ 个真实样本 $$x^{(i)}$$ | `torch.randn(B, z_dim)`；取一个 batch |
| 2 | 判别器梯度**上升**：$$\nabla_{\theta_d}\frac1m\sum_i\big[\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))\big]$$，重复 $$k$$ 次 | 真实标签 1、生成标签 0 的 BCE 之和，`fake.detach()` |
| 3 | 重新采 $$m$$ 个噪声 | 实践中常直接复用第 1 步的 `fake` |
| 4 | 生成器梯度**上升**（non-saturating）：$$\nabla_{\theta_g}\frac1m\sum_i\log D(G(z^{(i)}))$$ | 生成样本标签设为 1 的 BCE，不 detach |
| 5 | 重复 1 到 4；训练结束后只保留 $$G$$，生成时 $$x = G(z)$$ | `G(torch.randn(n, z_dim))` |

其中 $$k$$ 的选择没有定论：原论文实验用 $$k = 1$$，有些工作发现 $$k > 1$$ 更稳定。WGAN 一类方法通常取 $$k = 5$$（4.2 节）。训练结束后只需要用生成器来生成数据。

### 2.7 训练中的难点

GAN 的生成质量很高，但出名地难训练。常见问题有四类：

**（1）判别器太强时梯度消失。** 图像这类高维数据实际集中在一个低维流形上，训练初期 $$p_g$$ 和 $$p_{\text{data}}$$ 几乎不重叠。两个分布不重叠时，$$JSD = \log 2$$ 是一个常数，与 $$p_g$$ 离 $$p_{\text{data}}$$ 多远无关，于是没有梯度告诉生成器"往哪边走更近" [[7]](#ref-7)。此时判别器可以做到完美分类，用原始 loss 的生成器梯度趋于 0；non-saturating loss 虽然有梯度，但方差很大、训练不稳定。这正是 WGAN（4.2 节）要解决的问题。

**（2）模式坍缩（mode collapse）。** 生成器只生成少数几种样本（例如只会画某一类衣服），因为只要这几种能骗过判别器，loss 就足够低。判别器发现后开始惩罚这几种，生成器又整体跳到另外几种，如此往复，始终覆盖不了全部模式。3.5 节的二维实验会直接看到这个现象。

**（3）不收敛、振荡。** 极小极大博弈用梯度下降求解时，不一定收敛到平衡点。Goodfellow 的教程 [[2]](#ref-2) 给了一个最简单的例子：$$\min_x\max_y\ xy$$，平衡点是 $$(0, 0)$$。两个玩家同时做梯度更新时，轨迹绕着平衡点转圈并越转越远：

```python
x, y, lr = 1.0, 1.0, 0.1
for _ in range(100):
    x, y = x - lr * y, y + lr * x          # 同时更新：x 最小化 xy（梯度 y），y 最大化 xy（梯度 x）
print((x ** 2 + y ** 2) ** 0.5)            # 2.33：离平衡点越来越远（初始距离 1.41）

x, y = 1.0, 1.0
for _ in range(100):
    x = x - lr * y                         # 交替更新：y 用的是更新后的 x
    y = y + lr * x
print((x ** 2 + y ** 2) ** 0.5)            # 1.39：一直绕圈，既不发散也不收敛
```

GAN 的交替训练就是第二种。实际中的 loss 曲线因此经常上下振荡，不像普通监督学习那样单调下降，**loss 的大小也不能说明生成质量的好坏**。

**（4）难以评价。** GAN 没有似然，不能像 VAE 那样用测试集上的 ELBO 来比较模型。常用的指标是 FID [[18]](#ref-18)：用预训练的 Inception 网络提取真实图像和生成图像的特征，各自拟合一个高斯分布，计算两个高斯之间的 Fréchet 距离，越小越好。FID 同时反映了样本质量和多样性。

## 3. 代码实现

### 3.1 架构技巧

根据 GAN 的流程可以看出 [[3]](#ref-3)：

- 生成器通常由一些上采样（upsampling）模块组成，例如分数步长卷积（fractionally-strided convolution，即转置卷积 / Deconvolution），通过设计好 stride 等参数的卷积层来放大尺寸；
- 判别器通常就是标准的卷积分类网络。

![DCGAN 的架构准则](/deep-learning/assets/generative-models/Untitled%2037.png)

上图是 DCGAN [[6]](#ref-6) 总结的稳定训练卷积 GAN 的架构准则：

1. 不用池化层：判别器用步长卷积下采样，生成器用转置卷积上采样；
2. 生成器和判别器都使用 BatchNorm（但生成器的输出层和判别器的输入层不用）；
3. 深层网络去掉全连接隐层；
4. 生成器除输出层外都用 ReLU，输出层用 Tanh（所以数据要归一化到 $$[-1, 1]$$）；
5. 判别器全部使用 LeakyReLU。

论文中还有两条训练上的经验：权重用 $$\mathcal{N}(0, 0.02)$$ 初始化；Adam 的学习率取 $$2 \times 10^{-4}$$，$$\beta_1$$ 从默认的 0.9 降到 0.5，以减少振荡。下面是 DCGAN 论文中的生成器样例（用于 $$64 \times 64$$ 的 LSUN 图像）：

![DCGAN 生成器结构](/deep-learning/assets/generative-models/Untitled%2038.png)

100 维的 $$z$$ 先经过全连接层投影、reshape 成 $$4 \times 4 \times 1024$$，再经过 4 个步长为 2 的转置卷积，每次边长翻倍、通道减半，得到 $$64 \times 64 \times 3$$ 的图像。

### 3.2 一个最简单的 GAN：逐行解读

以 PyTorch-GAN 中的 [gan.py](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py) [[8]](#ref-8) 为例，生成器和判别器都是全连接网络。

**（1）生成器**

- 输入噪声 $$z$$ 的 shape 是 `(batch_size, latent_dim)`，从**标准正态分布** $$\mathcal{N}(0, 1)$$ 中采样，`latent_dim` 默认是 100；
- 整个生成器是 4 个 Linear + BatchNorm1d + LeakyReLU 的模块，最后接一个 Linear，输出维度是图像 shape 中 $$C \times H \times W$$ 的乘积，经过 Tanh 限制在 $$[-1, 1]$$ 之间，再 reshape 成图像的 shape。

```python
img_shape = (opt.channels, opt.img_size, opt.img_size)      # (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                # 注意：第二个位置参数是 eps，不是 momentum。原代码写成 0.8 很可能是笔误，
                # 本意应该是 momentum=0.8；eps=0.8 会让归一化明显偏离标准 BatchNorm
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),        # 100 -> 128，第一层不加 BN
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),            # 1024 -> 784
            nn.Tanh()                                # 输出 [-1, 1]，与归一化后的真实图像一致
        )

    def forward(self, z):                                        # z: [B, 100]
        img = self.model(z)                                      # [B, 784]
        img = img.view(img.size(0), *img_shape)                  # [B, 1, 28, 28]
        return img
```

```python
# 原代码使用了旧版 PyTorch 的 Variable 写法。PyTorch 0.4 之后 Variable 和 Tensor 已经合并，
# 直接用 tensor 即可。真实图像不需要梯度：real_imgs = imgs.to(device)
real_imgs = Variable(imgs.type(Tensor))

# 噪声：标准正态分布 N(0, 1)，shape [B, latent_dim]。等价的新写法：z = torch.randn(B, opt.latent_dim)
z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

gen_imgs = generator(z)                                          # G(z): [B, 1, 28, 28]
```

**（2）判别器**

判别器也是几层 Linear + LeakyReLU，最终输出一个标量。这里最后接了 Sigmoid，把输出限制在 $$[0, 1]$$ 之间，表示"是真实数据的概率" $$D(x)$$。

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),                                        # D(x) ∈ (0, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)                     # 先把图像拉平成 [B, 784]，再进入 Linear
        validity = self.model(img_flat)                          # [B, 1]
        return validity
```

**（3）训练的 loss**

交替训练时，loss 分为两类：

- 生成器只有一个 loss：判别器对生成样本的输出要"像真的"，即和标签 1 之间的 BCE 最小。这就是 2.5 节的 non-saturating loss：$$\operatorname{BCE}(D(G(z)), 1) = -\log D(G(z))$$；
- 判别器的 loss 包括两部分：
    - 输入生成样本，输出要判为假，即和标签 0 之间的 BCE 最小：$$-\log(1 - D(G(z)))$$；
    - 输入真实数据，输出要判为真，即和标签 1 之间的 BCE 最小：$$-\log D(x)$$。

    两部分相加，就是 <a href="#eq-gan-1">Eq. (1)</a> 的相反数。

```python
adversarial_loss = torch.nn.BCELoss()

# 标签：imgs 的维度是 (B, C, H, W)，因此 valid 和 fake 都是 (B, 1)
valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)   # 真：1
fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)    # 假：0

# ---------------- 训练生成器 ----------------
optimizer_G.zero_grad()
z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
gen_imgs = generator(z)
# 生成器的能力：骗过判别器。标签设为 1，即 -log D(G(z))（non-saturating）
g_loss = adversarial_loss(discriminator(gen_imgs), valid)
g_loss.backward()
optimizer_G.step()

# ---------------- 训练判别器 ----------------
optimizer_D.zero_grad()
real_loss = adversarial_loss(discriminator(real_imgs), valid)              # -log D(x)
# 注意这里要用 detach()：训练判别器时，阻止梯度经过 gen_imgs 传回生成器
fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)       # -log(1 - D(G(z)))
d_loss = (real_loss + fake_loss) / 2
d_loss.backward()
optimizer_D.step()
```

代码注意：

- 用 BCELoss 计算判别器的输出和对应标签之间的 loss：生成样本的标签是 0，真实图像的标签是 1。
- **计算判别器在生成样本上的 loss（`fake_loss`）时，要用 `gen_imgs.detach()`。** `gen_imgs` 是生成器的输出，梯度可以一路回传到生成器。训练判别器时生成器应当固定，所以要切断这条路径。`real_imgs` 不需要 `detach()`，因为它和生成器无关。这个细节很关键：如果不 detach，`d_loss.backward()` 会把梯度也算到生成器的参数上（方向是让生成器"更容易被识破"），虽然这里 `optimizer_D.step()` 不会更新生成器，但这些梯度会留在生成器参数的 `.grad` 里，并且多做了一次没用的反向传播。
- 反过来，训练生成器时**不能** detach：`g_loss` 的梯度要经过判别器传回生成器。这一步也会在判别器的参数上留下梯度，所以下一次训练判别器前必须 `optimizer_D.zero_grad()`。
- 判别器的最终 loss `d_loss` 是 `real_loss` 和 `fake_loss` 之和的一半。除以 2 只是让数值和生成器的 loss 处于同一量级，不影响最优解。
- 这份代码先训练生成器、再训练判别器，和 2.6 节的算法顺序相反。两种顺序都可以，因为是交替进行的。

### 3.3 完整的训练循环

下面是一个更现代、更稳妥的写法，也是第 3.5 节实验用的代码：判别器直接输出 logit（不加 Sigmoid），loss 用 `binary_cross_entropy_with_logits`。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):                                       # DCGAN：权重 ~ N(0, 0.02)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0, 0.02)
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1, 0.02); nn.init.zeros_(m.bias)

G, D = DCGenerator(), DCDiscriminator()                    # 见 3.4 节；D 输出 logit a(x)
G.apply(weights_init); D.apply(weights_init)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(10):
    for x, _ in train_loader:                              # x: [B, 1, 28, 28]，已归一化到 [-1, 1]
        B = x.shape[0]
        fake = G(torch.randn(B, 100))                      # G(z)

        # ---- 判别器：最大化 log D(x) + log(1 - D(G(z)))，即最小化两项 BCE
        logit_real, logit_fake = D(x), D(fake.detach())    # detach：这一步不更新 G
        loss_D = F.binary_cross_entropy_with_logits(logit_real, torch.ones_like(logit_real)) \
               + F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # ---- 生成器（non-saturating）：最小化 -log D(G(z))
        logit_gen = D(fake)                                # 不 detach，并且用的是更新后的 D
        loss_G = F.binary_cross_entropy_with_logits(logit_gen, torch.ones_like(logit_gen))
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

# 生成
G.eval()
with torch.no_grad():
    samples = G(torch.randn(64, 100))          # [-1, 1]；显示前 (samples + 1) / 2
```

> **易错点**
>
> 1. **用 logit + `binary_cross_entropy_with_logits`，不要 Sigmoid + `BCELoss`。** 判别器很自信时 sigmoid 输出会变成精确的 0 或 1，`BCELoss` 里的 $$\log 0$$ 会被截断，梯度随之消失或变成 NaN。`with_logits` 版本把 sigmoid 和 log 合并计算（$$-\log\sigma(a) = \operatorname{softplus}(-a)$$），数值上是稳定的。
> 2. **数据归一化要和生成器输出层一致。** 生成器用 Tanh 输出 $$[-1, 1]$$，真实图像就要归一化到 $$[-1, 1]$$（`Normalize((0.5,), (0.5,))`）。如果真实图像在 $$[0, 1]$$，判别器只看像素范围就能分出真假。
> 3. **`detach` 只在判别器那一步用。** 判别器步用 `fake.detach()`；生成器步用不 detach 的 `fake`，否则 `loss_G` 传不回生成器。
> 4. **每一步之前都要 `zero_grad`。** 生成器步的反向传播会在判别器参数上留下梯度；如果判别器步之前不清零，这些梯度会被累加进去。
> 5. **生成器 loss 的标签是 1，不是 0。** 标签设为 0 再取负（`-BCE(D(G(z)), 0)`）是原始的 saturating loss，训练初期几乎学不动（2.5 节、3.5 节）。
> 6. **生成时切换到 `G.eval()`。** BatchNorm 在训练和推理时的行为不同；用 `train()` 模式、batch 很小时生成，结果会受 batch 内其他样本影响。
> 7. **不要用 loss 判断训练好坏。** 健康的训练中 `loss_D` 和 `loss_G` 都在某个值附近振荡；判别器 loss 很快降到接近 0，通常说明判别器太强、生成器学不动了。要定期看生成的样本。

### 3.4 DCGAN

把全连接换成卷积，就是 DCGAN [[6]](#ref-6)。下面是适配 $$28 \times 28$$ 灰度图的小版本，遵守 3.1 节的准则：

```python
class DCGenerator(nn.Module):
    def __init__(self, z_dim=100, ch=64):
        super().__init__()
        self.ch = ch
        # 投影并 reshape：100 -> 128×7×7
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 2 * ch * 7 * 7, bias=False), nn.BatchNorm1d(2 * ch * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1, bias=False),         # 7 -> 14
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, 1, 4, 2, 1),                          # 14 -> 28
            nn.Tanh())                                                   # 输出层：Tanh，不加 BN

    def forward(self, z):                                                # z: [B, 100]
        return self.net(self.fc(z).view(-1, 2 * self.ch, 7, 7))          # [B, 1, 28, 28]


class DCDiscriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2),                # 28 -> 14，输入层不加 BN
            nn.Conv2d(ch, 2 * ch, 4, 2, 1, bias=False),                  # 14 -> 7
            nn.BatchNorm2d(2 * ch), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(2 * ch * 7 * 7, 1))                                # 输出 logit，不加 Sigmoid

    def forward(self, x):                                                # x: [B, 1, 28, 28]
        return self.net(x)                                               # [B, 1]
```

`ConvTranspose2d(k=4, s=2, p=1)` 正好把边长放大 2 倍，`Conv2d(k=4, s=2, p=1)` 正好缩小 2 倍，两者一一对应。后面紧跟 BatchNorm 的卷积层不需要 bias，因为 BatchNorm 会减去均值，bias 被抵消。

### 3.5 实验

**实验一：FashionMNIST 上的 MLP GAN 和 DCGAN。** 用 3.3 节的训练循环（non-saturating loss，Adam $$2 \times 10^{-4}$$、$$\beta_1 = 0.5$$，batch 128），分别训练 3.2 节的 MLP GAN（BatchNorm 用默认参数）和 3.4 节的 DCGAN，各 10 个 epoch。

<figure class="sd-img">
<img src="/deep-learning/assets/generative-models-2-gan/fmnist_samples.png" alt="MLP GAN 和 DCGAN 在不同 epoch 的生成样本">
<figcaption>同一组固定的 z 在第 1、3、10 个 epoch 的生成结果。MLP GAN 第 10 个 epoch 已经能画出上衣、裤子、鞋的轮廓，但满是椒盐状的噪点：全连接网络没有"相邻像素相关"的先验，判别器也很难发现这种逐像素的噪声。DCGAN 第 3 个 epoch 就已经有清楚的类别，第 10 个 epoch 的样本平滑、细节更多，类别也很多样。两者参数量相近（MLP GAN 的生成器约 150 万参数，DCGAN 的生成器约 77 万），差别来自卷积的结构先验。</figcaption>
</figure>

训练过程中判别器的输出（每 50 步记录一次 batch 平均值，再做 5 点平滑）：

<figure class="sd-img">
<img src="/deep-learning/assets/generative-models-2-gan/fmnist_curves.png" alt="训练中 D(x) 和 D(G(z)) 的变化">
<figcaption>理想的平衡点是两条线都等于 0.5（2.3 节）。MLP GAN 的两条线很快贴近 0.5，这里并不说明生成器已经完美，而是判别器太弱、已经分不出带噪点的假图。DCGAN 的判别器更强，D(x) 在 0.55 到 0.8 之间、D(G(z)) 在 0.2 到 0.5 之间来回振荡（2.7 节第 3 点），但生成质量在稳步提升。所以这些曲线只能用来发现异常（例如 D(G(z)) 长期接近 0，说明生成器学不动），不能代替看样本。</figcaption>
</figure>

在 CPU 上，两个模型各训练 10 个 epoch 共约 9 分钟。

**实验二：二维 8 高斯上的三种 loss。** 为了直接看到 2.5 节和 2.7 节的现象，用一个二维的玩具分布：8 个高斯排成一个圆环。生成器和判别器都是两层 MLP，三种设置使用完全相同的网络、优化器（Adam $$10^{-4}$$）和每轮 5 次判别器更新，只有 loss 不同：原始的 saturating loss、non-saturating loss、WGAN-GP（4.2 节）。"覆盖的模式数"统计的是：落在某个高斯中心 0.3 范围内的样本超过 2% 时，该模式算被覆盖。

<figure class="sd-img">
<img src="/deep-learning/assets/generative-models-2-gan/toy_8gaussians.png" alt="二维 8 高斯上三种 loss 的训练过程">
<figcaption>灰色是真实数据，红色是生成样本（同一组 2000 个固定的 z）。第 1 行，原始 loss：判别器每轮更新 5 次后很容易识破生成样本，生成器梯度很小，2000 步时样本仍然散成一片，6000 步才覆盖一半的模式。第 2 行，non-saturating loss：学得很快，但 2000 步起就坍缩到 2 个模式，之后一直没有恢复，这就是模式坍缩。第 3 行，WGAN-GP：2000 步时已覆盖全部 8 个模式。</figcaption>
</figure>

| loss | 第 500 步 | 第 2000 步 | 第 6000 步 | 第 6000 步时落在某个模式附近的样本比例 |
|---|---|---|---|---|
| 原始（saturating） | 0 / 8 | 2 / 8 | 4 / 8 | 38% |
| non-saturating | 2 / 8 | 2 / 8 | 2 / 8 | 96% |
| WGAN-GP | 0 / 8 | 8 / 8 | 8 / 8 | 64% |

non-saturating 版本有 96% 的样本落在模式附近，单看"样本像不像真的"它是最好的，但它只覆盖了 2 个模式。这正好对应 2.5 节的分析：它更怕生成"不像真的"样本，而不怕漏掉模式。需要说明的是，这是一个小实验：判别器每轮只更新 1 次、或者把学习率调到 $$10^{-3}$$ 时，前两种 loss 在这个简单分布上也能覆盖全部模式。它展示的是这些问题在判别器较强时**会**出现，而不是一定出现。

## 4. GAN 的变体

GAN 的后续工作非常多（[[19]](#ref-19) 收集了数百种），这里按"解决什么问题"介绍最主要的几类。

### 4.1 条件 GAN（cGAN）

原始 GAN 无法控制生成哪一类。条件 GAN [[11]](#ref-11) 把条件 $$y$$（类别、文本、另一张图……）同时输入生成器和判别器：

$$
\min_G\max_D\ \mathbb{E}_{x, y}\big[\log D(x \vert y)\big] + \mathbb{E}_{z, y}\big[\log\big(1 - D(G(z \vert y) \vert y)\big)\big]
$$

判别器要判断的变成了"$$x$$ 是否是**符合条件 $$y$$** 的真实数据"，所以生成器不但要生成真实的图，还要生成**和 $$y$$ 匹配**的图。最简单的实现是把类别的嵌入拼接到输入上：

```python
class CondGenerator(nn.Module):
    def __init__(self, z_dim=100, n_classes=10, emb=10):
        super().__init__()
        self.embed = nn.Embedding(n_classes, emb)                # 类别 -> 10 维向量
        self.net = nn.Sequential(
            nn.Linear(z_dim + emb, 256), nn.LeakyReLU(0.2),      # 输入是 [z, embed(y)]
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 784), nn.Tanh())

    def forward(self, z, y):                                     # z: [B, 100]，y: [B] 类别
        return self.net(torch.cat([z, self.embed(y)], dim=1)).view(-1, 1, 28, 28)


class CondDiscriminator(nn.Module):
    def __init__(self, n_classes=10, emb=10):
        super().__init__()
        self.embed = nn.Embedding(n_classes, emb)
        self.net = nn.Sequential(
            nn.Linear(784 + emb, 512), nn.LeakyReLU(0.2),        # 输入是 [x, embed(y)]
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1))                                   # logit：x 是否是"符合 y 的真实数据"

    def forward(self, x, y):
        return self.net(torch.cat([x.flatten(1), self.embed(y)], dim=1))

# 训练时真实样本用它自己的标签 y；生成样本用随机标签 y_fake 同时输入 G 和 D：
# logit_fake = D(G(z, y_fake).detach(), y_fake)
```

后来更常用的是投影判别器（projection discriminator）：判别器最后的输出加上 $$\langle \text{embed}(y), \phi(x) \rangle$$，其中 $$\phi(x)$$ 是判别器倒数第二层的特征。生成器则常用条件 BatchNorm，让 BN 的缩放和偏移由 $$y$$ 决定。

### 4.2 WGAN 和 WGAN-GP

**问题。** 2.7 节说过，$$p_g$$ 和 $$p_{\text{data}}$$ 不重叠时，JS 散度恒为 $$\log 2$$，提供不了"距离还有多远"的信息。看一个例子 [[12]](#ref-12)：二维平面上，$$p_{\text{data}}$$ 是 $$x = 0$$ 这条竖线上的均匀分布，$$p_g$$ 是 $$x = \theta$$ 这条竖线上的均匀分布。只要 $$\theta \ne 0$$，两者完全不重叠，$$JSD = \log 2$$，对 $$\theta$$ 的梯度为 0，生成器不知道该把线往哪边移。

**Wasserstein 距离。** WGAN 改用 Wasserstein-1 距离（又叫推土机距离，Earth Mover's distance）：

$$
W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x, y) \sim \gamma}\big[\lVert x - y \rVert\big]
$$

$$\Pi$$ 是所有边缘分布分别为 $$p_{\text{data}}$$ 和 $$p_g$$ 的联合分布。直观地说，把 $$p_g$$ 看成一堆土，$$p_{\text{data}}$$ 看成要填的坑，$$\gamma$$ 是一种搬运方案，$$W$$ 就是最省力的方案要搬的"土量 × 距离"。上面的例子里 $$W = \lvert\theta\rvert$$，它随 $$\theta$$ 连续变化，处处有梯度。

**对偶形式。** 上面的定义无法直接计算。根据 Kantorovich-Rubinstein 对偶：

$$
W(p_{\text{data}}, p_g) = \sup_{\lVert f \rVert_L \le 1}\ \mathbb{E}_{x \sim p_{\text{data}}}\big[f(x)\big] - \mathbb{E}_{x \sim p_g}\big[f(x)\big]
$$

其中 $$\lVert f \rVert_L \le 1$$ 表示 $$f$$ 是 1-Lipschitz 函数，即 $$\lvert f(x_1) - f(x_2) \rvert \le \lVert x_1 - x_2 \rVert$$。把 $$f$$ 用一个神经网络 $$D$$ 表示，就得到 WGAN 的目标：

$$
\min_G\ \max_{D:\ 1\text{-Lipschitz}}\ \mathbb{E}_{x \sim p_{\text{data}}}\big[D(x)\big] - \mathbb{E}_{z \sim p_z}\big[D(G(z))\big]
$$

这里的 $$D$$ 不再是分类器，输出不经过 sigmoid、没有 log，是一个任意实数的"打分"，所以论文改称它为 critic（评论家）。critic 训练得越好，它给出的值越接近真实的 Wasserstein 距离，因此 **WGAN 的 critic loss 和生成质量相关**，可以用来监控训练，这是原始 GAN 做不到的。

**怎么保证 Lipschitz 约束。** 原始 WGAN 用权重裁剪（weight clipping）：每次更新后把 critic 的所有权重截断到 $$[-c, c]$$（$$c = 0.01$$）。这很粗糙：$$c$$ 太大训练慢，太小梯度消失，而且会让 critic 退化成过于简单的函数。

WGAN-GP [[13]](#ref-13) 改用**梯度惩罚**（gradient penalty）。可以证明，最优 critic 在 $$p_{\text{data}}$$ 和 $$p_g$$ 之间（按最优搬运方案配对的点的连线上）梯度的模恰好为 1。于是在真实样本和生成样本的随机插值点上，惩罚梯度的模偏离 1：

$$
L_D = \underbrace{\mathbb{E}_{\tilde x \sim p_g}\big[D(\tilde x)\big] - \mathbb{E}_{x \sim p_{\text{data}}}\big[D(x)\big]}_{\text{原 critic loss}} + \lambda\, \mathbb{E}_{\hat x}\Big[\big(\lVert \nabla_{\hat x} D(\hat x) \rVert_2 - 1\big)^2\Big], \qquad \hat x = \epsilon x + (1 - \epsilon)\tilde x,\ \ \epsilon \sim U[0, 1]
$$

论文取 $$\lambda = 10$$。生成器的 loss 是 $$L_G = -\mathbb{E}_z\big[D(G(z))\big]$$。代码（3.5 节实验二用的就是这段）：

```python
def wgan_gp_d_loss(D, x_real, x_fake, lam=10.0):
    # critic 要让真实样本的分高、生成样本的分低
    loss = D(x_fake).mean() - D(x_real).mean()
    eps = torch.rand(x_real.shape[0], *([1] * (x_real.dim() - 1)))   # 每个样本一个 ε，[B, 1, 1, 1]
    x_hat = (eps * x_real + (1 - eps) * x_fake).requires_grad_(True)  # 插值点
    grad = torch.autograd.grad(D(x_hat).sum(), x_hat, create_graph=True)[0]   # ∇_x̂ D(x̂)
    gp = ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()          # (||∇||₂ - 1)²
    return loss + lam * gp

def wgan_g_loss(D, x_fake):
    return -D(x_fake).mean()

# 训练：每更新 1 次生成器，更新 n_critic = 5 次 critic；Adam(lr=1e-4, betas=(0.5, 0.9))
```

> **易错点**
>
> 1. **`create_graph=True`。** 梯度惩罚本身要对 critic 的参数求导，是"梯度的梯度"，必须保留计算图。
> 2. **critic 里不要用 BatchNorm。** 梯度惩罚是对**每个样本**单独施加的，BatchNorm 让一个样本的输出依赖于同一 batch 的其他样本，破坏了这个前提。论文建议改用 LayerNorm 或不用归一化。
> 3. **critic 最后不加 sigmoid，loss 里没有 log。**
> 4. **`x_fake` 要 detach。** 训练 critic 时生成样本来自 `G(z).detach()`；插值点 `x_hat` 自己需要梯度（`requires_grad_(True)`）。
> 5. **Adam 的 $$\beta_1$$ 取 0 到 0.5，并且 critic 多更新几步。** 动量太大在这种博弈中容易振荡。

### 4.3 其他 loss 和谱归一化

除了 WGAN-GP，还有几种常用的 loss 和正则 [[14]](#ref-14)：

| 方法 | 判别器 loss | 生成器 loss | 特点 |
|---|---|---|---|
| 原始 GAN（non-saturating） | $$-\log\sigma(a_r) - \log(1 - \sigma(a_f))$$ | $$-\log\sigma(a_f)$$ | 2.5 节 |
| LSGAN | $$(a_r - 1)^2 + a_f^2$$ | $$(a_f - 1)^2$$ | 用平方误差代替交叉熵，离决策边界远的样本也有梯度 |
| WGAN | $$a_f - a_r$$ + Lipschitz 约束 | $$-a_f$$ | 4.2 节 |
| Hinge | $$\max(0, 1 - a_r) + \max(0, 1 + a_f)$$ | $$-a_f$$ | SN-GAN、BigGAN、VQGAN 使用；判别器对已经分得很开的样本不再用力 |

表中 $$a_r = a(x)$$、$$a_f = a(G(z))$$ 是判别器的原始输出（logit），各项都对 batch 取平均。

**谱归一化（spectral normalization）** 是另一种控制判别器 Lipschitz 常数的方法：把每一层权重除以它的最大奇异值（谱范数）$$\sigma(W)$$，即 $$\bar W = W / \sigma(W)$$，使每层都是 1-Lipschitz，整个网络的 Lipschitz 常数也就不超过 1（激活函数如 ReLU、LeakyReLU 本身是 1-Lipschitz 的）。$$\sigma(W)$$ 用幂迭代（power iteration）估计，每步迭代一次，开销很小。PyTorch 中一行即可：

```python
from torch.nn.utils import spectral_norm

D = nn.Sequential(
    spectral_norm(nn.Conv2d(1, 64, 4, 2, 1)), nn.LeakyReLU(0.2),
    spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2),
    nn.Flatten(), spectral_norm(nn.Linear(128 * 7 * 7, 1)))

def hinge_d_loss(logit_real, logit_fake):
    return F.relu(1 - logit_real).mean() + F.relu(1 + logit_fake).mean()

def hinge_g_loss(logit_fake):
    return -logit_fake.mean()
```

谱归一化 + hinge loss 不需要额外计算梯度惩罚，比 WGAN-GP 快，是后来很多大规模 GAN 的默认选择。StyleGAN 系列则常用 R1 正则：只在真实样本上惩罚判别器的梯度 $$\frac{\gamma}{2}\mathbb{E}_{x\sim p_{\text{data}}}\lVert\nabla_x D(x)\rVert^2$$。

### 4.4 图像翻译：Pix2Pix、PatchGAN 和 CycleGAN

**Pix2Pix** [[15]](#ref-15) 是条件 GAN 在"图像到图像"任务上的应用：输入一张图 $$y$$（例如边缘图、语义分割图、黑白照片），输出对应的另一张图 $$x$$（照片、彩色照片）。生成器是 U-Net，loss 是条件 GAN loss 加上 L1 重建：

$$
G^* = \arg\min_G\max_D\ \mathcal{L}_{\text{cGAN}}(G, D) + \lambda\,\mathbb{E}_{x, y}\big[\lVert x - G(y) \rVert_1\big], \qquad \lambda = 100
$$

两项分工明确：L1 负责低频（整体的颜色和结构对不对），但单独用 L1 结果会很模糊（它倾向于输出所有可能结果的中位数）；GAN loss 负责高频（纹理、边缘是否清晰）。

为了让判别器专注于高频细节，Pix2Pix 提出了 **PatchGAN** 判别器：它是一个全卷积网络，不输出一个标量，而是输出一个 $$N \times N$$ 的网格，每个位置判断输入图像中对应的一个局部 patch（例如 $$70 \times 70$$ 的感受野）是真是假，loss 对所有位置取平均。这样判别器参数更少、可以处理任意尺寸的图，并且只惩罚局部纹理，全局结构交给 L1。VQGAN（4.6 节）和 Stable Diffusion 的 VAE 训练都沿用了 PatchGAN 判别器。

**CycleGAN** [[15]](#ref-15) 解决没有成对数据的情况（例如马 ↔ 斑马、照片 ↔ 莫奈画）。它训练两个生成器 $$G: X \to Y$$、$$F: Y \to X$$ 和两个判别器，再加上**循环一致性**约束：转过去再转回来要还原，

$$
\mathcal{L}_{\text{cyc}} = \mathbb{E}_x\big[\lVert F(G(x)) - x \rVert_1\big] + \mathbb{E}_y\big[\lVert G(F(y)) - y \rVert_1\big]
$$

没有这一项，生成器只要生成任意一张"像斑马"的图就能骗过判别器，和输入的马没有关系。

### 4.5 StyleGAN

StyleGAN [[16]](#ref-16) 是生成高分辨率人脸等图像的代表性工作，主要改动在生成器的结构：

1. **映射网络（mapping network）。** 一个 8 层 MLP 把 $$z \in \mathcal{Z}$$ 映射成中间向量 $$w \in \mathcal{W}$$。$$z$$ 必须服从固定的高斯分布，而 $$w$$ 的分布可以是任意形状，因此更容易"解耦"，即 $$w$$ 的不同方向对应不同的属性；
2. **风格注入。** 生成器从一个**学到的常数张量**出发（而不是从 $$z$$ 出发），在每个分辨率上用 $$w$$ 经过仿射变换得到的"风格"去调制特征。第一版用 AdaIN：$$\operatorname{AdaIN}(h, w) = \gamma(w)\frac{h - \mu(h)}{\sigma(h)} + \beta(w)$$，即先把每个通道归一化，再用 $$w$$ 决定的缩放和偏移；StyleGAN2 改成直接调制卷积权重；
3. **噪声输入。** 每一层加入逐像素的随机噪声，负责头发丝、毛孔这类随机细节；
4. **风格混合和截断技巧。** 低分辨率层用一个 $$w$$、高分辨率层用另一个 $$w$$，可以分别控制姿态脸型和颜色细节；生成时把 $$w$$ 向平均值 $$\bar w$$ 拉近（$$w' = \bar w + \psi(w - \bar w)$$，$$\psi < 1$$），用多样性换质量。

它继承了 ProGAN 的渐进式训练（从 $$4 \times 4$$ 逐步加层到 $$1024 \times 1024$$；StyleGAN2 去掉了渐进式训练）。

### 4.6 VQGAN

VQGAN [[17]](#ref-17) 把 GAN 用在了 [VQ-VAE](/deep-learning/generative-models-1-vae/) 的训练上，目标是一个**压缩率很高、但重建依然清晰**的图像 tokenizer。它分两个阶段：

**第一阶段：训练带对抗 loss 的 VQ-VAE。** 结构和 VQ-VAE 相同（Encoder、码本、Decoder），loss 增加了两项：

$$
\mathcal{L} = \underbrace{\lVert x - \hat x \rVert_1 + \mathcal{L}_{\text{LPIPS}}(x, \hat x)}_{\text{重建 + 感知}} + \underbrace{\lVert \operatorname{sg}[z_e] - z_q \rVert^2 + \beta\lVert z_e - \operatorname{sg}[z_q] \rVert^2}_{\text{VQ（同 VQ-VAE）}} + \lambda\,\mathcal{L}_{\text{GAN}}(\hat x)
$$

- **感知损失（LPIPS）**：在预训练 VGG 网络的特征空间里比较 $$x$$ 和 $$\hat x$$，比像素误差更符合人眼感受；
- **对抗损失**：PatchGAN 判别器（4.4 节）+ hinge loss，逼 Decoder 生成清晰的纹理，而不是 L1/L2 带来的模糊；
- **自适应权重 $$\lambda$$**：对抗 loss 和重建 loss 的梯度大小可能相差很多，固定权重很难调。VQGAN 让两者在 Decoder **最后一层** $$G_L$$ 上的梯度大小相当：

$$
\lambda = \frac{\lVert \nabla_{G_L}\mathcal{L}_{\text{rec}} \rVert}{\lVert \nabla_{G_L}\mathcal{L}_{\text{GAN}} \rVert + \delta}, \qquad \delta = 10^{-4}
$$

- **延迟启动判别器**：先只用重建和 VQ loss 训练若干步（官方代码中的 `disc_start`），等重建大致可用了再加入对抗 loss，否则判别器一开始就轻易获胜，干扰 Decoder 的学习。

**第二阶段：Transformer 先验。** 和 VQ-VAE 用 PixelCNN 不同，VQGAN 在 token 序列上训练一个 GPT 式的 Transformer，做下一个 token 预测；条件生成（类别、分割图等）时把条件 token 放在序列前面。生成高分辨率图像时用滑动窗口逐块采样。

因为对抗和感知损失保证了重建质量，VQGAN 可以把 $$256 \times 256$$ 的图像压成 $$16 \times 16$$ 个 token（每边下采样 16 倍），远比 VQ-VAE 激进，Transformer 的序列长度因此可以接受。它的第一阶段后来被 [Latent Diffusion](/deep-learning/latent-diffusion/) 直接沿用：LDM 的 VQ 正则和 KL 正则两种自编码器，都用这套"重建 + LPIPS + PatchGAN"的 loss 训练。

下面是一个简化的第一阶段训练步骤：在 [VQ-VAE 篇](/deep-learning/generative-models-1-vae/) 第 6.3 节的 `VQVAE` 上加入 PatchGAN 判别器、hinge loss 和自适应权重（省略了需要预训练 VGG 权重的 LPIPS）：

```python
class PatchDiscriminator(nn.Module):
    """输出一个 logit 网格，每个位置判断一个局部 patch 的真假（最后不做池化）"""
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2),                                   # 28 -> 14
            nn.Conv2d(ch, 2 * ch, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * ch), nn.LeakyReLU(0.2),
            nn.Conv2d(2 * ch, 1, 3, 1, 1))                         # [B, 1, 7, 7]

    def forward(self, x):
        return self.net(x)


def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (F.relu(1 - logits_real).mean() + F.relu(1 + logits_fake).mean())


def adaptive_weight(rec_loss, g_loss, last_layer, delta=1e-4):
    """λ = ||∇_{G_L} L_rec|| / (||∇_{G_L} L_GAN|| + δ)，G_L 是 Decoder 最后一层的权重"""
    g_rec = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    g_gan = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    return (g_rec.norm() / (g_gan.norm() + delta)).clamp(0, 1e4).detach()


def train_step(model, disc, opt_ae, opt_d, x, use_gan, disc_weight=0.5):
    """x: [B, 1, 28, 28]，范围 [0, 1]；use_gan 在 disc_start 步之后才为 True"""
    # ---- 自编码器（相当于生成器）
    logits, vq_loss, _ = model(x)
    x_rec = torch.sigmoid(logits)
    rec_loss = (x_rec - x).abs().mean()                    # L1（完整的 VQGAN 还要加 LPIPS）
    loss = rec_loss + vq_loss
    if use_gan:
        g_loss = -disc(x_rec * 2 - 1).mean()               # hinge 的生成器 loss；判别器输入用 [-1, 1]
        lam = adaptive_weight(rec_loss, g_loss, model.decoder[-1].weight)
        loss = loss + disc_weight * lam * g_loss
    opt_ae.zero_grad(); loss.backward(); opt_ae.step()

    # ---- 判别器
    if use_gan:
        d_loss = hinge_d_loss(disc(x * 2 - 1), disc(x_rec.detach() * 2 - 1))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
```

为了验证这段代码可以运行，从 VQ-VAE 篇训练好的模型出发（相当于已经过了 `disc_start` 阶段），用 batch 64 跑了 40 步：判别器的 hinge loss 从 1.00 缓慢下降，自适应权重 $$\lambda$$ 在 0.15 到 0.44 之间变化，即对抗 loss 的梯度在 Decoder 最后一层上比重建 loss 的梯度大几倍，$$\lambda$$ 自动把它缩小到与重建 loss 相当。这里只是运行检查，没有做完整训练；在 $$28 \times 28$$、每边只下采样 4 倍的设置下，VQ-VAE 的重建本来就已经比较清晰，对抗 loss 的作用要在高压缩率、大图像上才明显。

> **易错点**
>
> 1. **自适应权重要 `detach()`。** $$\lambda$$ 只是一个缩放系数，不应该再对它求导；`retain_graph=True` 是因为同一张图后面还要用于 `loss.backward()`。
> 2. **`last_layer` 是 Decoder 输出层的权重张量**（官方代码中是 `decoder.conv_out.weight`），不是整个模块。用更前面的层，梯度比值的含义就变了。
> 3. **判别器训练时 `x_rec` 要 detach，生成器步不能 detach**，和 3.3 节完全相同。
> 4. **不要一开始就加对抗 loss。** 官方配置的 `disc_start` 通常是几千到几万步。

## 5. GAN、VAE 与扩散模型

| | VAE | GAN | 扩散模型 |
|---|---|---|---|
| 训练目标 | 最大化似然的下界 ELBO | 两人博弈，理想情况下最小化 JS 散度（或 Wasserstein 距离） | 预测噪声的 MSE（也是 ELBO 的简化） |
| 训练稳定性 | 稳定 | 不稳定，需要很多技巧 | 稳定 |
| 样本质量 | 偏模糊 | 清晰 | 清晰 |
| 多样性 | 好 | 容易模式坍缩 | 好 |
| 采样速度 | 一次前向 | 一次前向 | 多步迭代（DDIM 等可减少步数） |
| 能否估计似然 | 能（下界） | 不能 | 能（下界） |

如今在高质量图像生成上，扩散模型已经取代了 GAN 的主导地位，但对抗损失仍然随处可见：[Stable Diffusion](/deep-learning/stable-diffusion/) 的 VAE、VQGAN 等 tokenizer 都用 PatchGAN 判别器训练，以保证重建清晰；扩散模型的少步蒸馏（例如对抗扩散蒸馏）也用判别器来提升一步或几步生成的质量。

### 要点回顾

- **GAN 在优化什么？** $$\min_G\max_D\ \mathbb{E}_x\log D(x) + \mathbb{E}_z\log(1 - D(G(z)))$$。判别器在做二分类（BCE），生成器在让判别器犯错。
- **最优判别器是什么？** $$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$，由逐点最大化 $$a\log y + b\log(1-y)$$ 得到。
- **为什么能学到数据分布？** 代入 $$D^*$$ 后，生成器的目标是 $$-\log 4 + 2\,JSD(p_{\text{data}}\Vert p_g)$$，当且仅当 $$p_g = p_{\text{data}}$$ 时最小，此时 $$D^* = \frac12$$。但这要求判别器每步最优、在分布空间里优化，实际训练不满足。
- **为什么生成器用 $$-\log D(G(z))$$？** 原始的 $$\log(1 - D(G(z)))$$ 对 logit 的梯度是 $$-D$$，样本差（$$D \approx 0$$）时梯度消失；换成 $$-\log D$$ 后梯度是 $$-(1 - D)$$，样本越差信号越强。代码里就是"生成样本的标签设为 1"。
- **代码最容易错在哪？** 判别器步要 `fake.detach()`、生成器步不能；每步前 `zero_grad`；数据归一化到 $$[-1, 1]$$ 配合 Tanh；用 logit + `BCEWithLogits`。
- **WGAN 解决什么？** 分布不重叠时 JS 散度是常数、没有梯度；Wasserstein 距离随分布的移动连续变化。critic 要满足 1-Lipschitz，用梯度惩罚（WGAN-GP）或谱归一化实现。
- **VQGAN 和 GAN 的关系？** VQGAN = VQ-VAE + 感知损失 + PatchGAN 对抗损失 + 自适应权重，再用 Transformer 学 token 先验；对抗损失让高压缩率下的重建依然清晰。
