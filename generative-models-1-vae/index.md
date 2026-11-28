---
date: 2025-09-01 00:00:00 -0800
tags: [VAE]
keywords: >-
  VAE, ELBO derivation, reparameterization trick, VAE loss scaling, posterior collapse, Conv-VAE, VQ-VAE, vector quantization, straight-through estimator, codebook, autoregressive prior, VQGAN, FSQ.
summary: >-
  从 AutoEncoder 的局限出发，推导 VAE 的证据下界 ELBO、重建项和 KL 项，并说明从 ELBO 到实际 loss 的三个关键细节（单样本近似、似然的系数、按像素求和）。给出完整的 PyTorch 实现和 FashionMNIST 实验，演示 loss 尺度写错时的后验坍缩。后半部分介绍 VQ-VAE：向量量化、码本和 commitment loss、梯度直通、第二阶段的自回归先验，以及 VQ-VAE-2、VQGAN、FSQ 等后续工作。
layout: note
title: "VAE 与 VQ-VAE：从推导到实现"
description: "VAE derivation (ELBO, reconstruction and KL terms), how the ELBO becomes the training loss, PyTorch implementation and experiments, Conv-VAE, and VQ-VAE with codebook, straight-through gradients and an autoregressive prior."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig svg .lane{font-size:14px;font-weight:600}
.ddpm-fig svg .it{font-style:italic;font-family:"Times New Roman",serif;font-size:16px}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
.sd-img{margin:1.2em 0;text-align:center}
.sd-img img{max-width:100%;height:auto;border:1px solid #eaecf0;border-radius:6px}
.sd-img figcaption{color:#667085;font-size:.9em;margin-top:.4em}
</style>

# VAE 与 VQ-VAE：从推导到实现

VAE 全称是 Variational AutoEncoder（变分自编码器），是针对普通 AutoEncoder 的改进：它让 latent code 服从一个已知的分布，从而既能重建输入，又能从这个分布里采样、生成新的数据。VQ-VAE 把连续的高斯 latent 换成一个离散的码本，是 VQGAN、Latent Diffusion 的 VQ 版本以及很多图像、视频 tokenizer 的基础。

本文先从直观例子讲 VAE 的出发点（第 1、2 节），再推导证据下界 ELBO 和最终的 loss（第 3 节），然后给出完整的代码和实验（第 4、5 节），最后详细介绍 VQ-VAE（第 6 节）和后续工作（第 7 节）。需要的前置知识：

| 内容 | 用在哪里 | 详细讲解 |
|---|---|---|
| 生成模型的分类、FVBN | 第 2 节的出发点 | [生成模型概览](/deep-learning/generative-models/) |
| 高斯分布、KL 散度、两个高斯之间的 KL | 第 3.2 节 | [数学基础](/deep-learning/math-foundation/) |
| 卷积、转置卷积 | 第 5、6 节 | [数学基础](/deep-learning/math-foundation/) 中的 Transposed Convolution 部分 |

## 参考资料

本文的推导主要参考 VAE 原论文 [1]、Doersch 的教程 [2] 和 Stanford CS231n 课程 [3]，VQ-VAE 部分主要参考原论文 [17]。正文中用 [n] 标注具体出处。

- <a id="ref-1"></a>**[1] VAE 原论文**：Kingma, Welling. *Auto-Encoding Variational Bayes*. ICLR 2014. [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)。ELBO、重参数化、Appendix B 的高斯 KL 闭式解都出自这篇。
- <a id="ref-2"></a>**[2] 最清楚的教程论文**：Doersch. *Tutorial on Variational Autoencoders*. 2016. [arxiv.org/abs/1606.05908](https://arxiv.org/abs/1606.05908)。很长，但前半部分把 VAE 的出发点和推导讲得非常清楚。
- <a id="ref-3"></a>**[3] Stanford CS231n Lecture 12**：[视频（从 VAE 部分开始）](https://youtu.be/5WoItGTWV54?t=26m32s)，[课件](http://cs231n.stanford.edu/slides/2021/lecture_12.pdf)。第 2、3 节的几张图来自这份课件。
- <a id="ref-4"></a>**[4] 推导很细的视频**：Ali Ghodsi, *Deep Learning, Variational Autoencoder*. [YouTube](https://www.youtube.com/watch?v=gErs0bAM63E)。
- <a id="ref-5"></a>**[5] 总结全面的博客**：Lilian Weng. *From Autoencoder to Beta-VAE*. [lilianweng.github.io/posts/2018-08-12-vae/](https://lilianweng.github.io/posts/2018-08-12-vae/)。β-VAE、VQ-VAE、VQ-VAE-2 都有涉及。
- <a id="ref-6"></a>**[6] 中文教程**：张振虎，变分自编码器。[zhangzhenhu.com](https://www.zhangzhenhu.com/aigc/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8.html)。总结全面，后续还会引出扩散模型，第 3.1 节的推导部分参考了它。
- <a id="ref-7"></a>**[7] 变分推断入门博客**：Eric Jang. *A Beginner's Guide to Variational Methods: Mean-Field Approximation*. [blog.evjang.com](https://blog.evjang.com/2016/08/variational-bayes.html)。图文展示了 KL 散度和 ELBO 的推导。
- <a id="ref-8"></a>**[8] 博客**：*The Variational Autoencoder*. [dfdazac.github.io/01-vae.html](https://dfdazac.github.io/01-vae.html)。
- <a id="ref-9"></a>**[9] 直观例子多的博客**：Jeremy Jordan. *Variational autoencoders*. [jeremyjordan.me/variational-autoencoders/](https://www.jeremyjordan.me/variational-autoencoders/)。第 1 节的属性分布图来自这里。
- <a id="ref-10"></a>**[10] 博客**：*Intuitively Understanding Variational Autoencoders*. [towardsdatascience.com](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)。第 1 节 latent 分布的例子来自这里。
- <a id="ref-11"></a>**[11] 博客**：*A Wizard's Guide to Adversarial Autoencoders, Part 2*. [towardsdatascience.com](https://towardsdatascience.com/a-wizards-guide-to-adversarial-autoencoders-part-2-exploring-latent-space-with-adversarial-2d53a6f8a4f9)。讲 GAN 的文章，开头部分介绍了 VAE 的原理。
- <a id="ref-12"></a>**[12] 中文教程**：变分自编码器介绍、推导及实现。[zhuanlan.zhihu.com/p/83865427](https://zhuanlan.zhihu.com/p/83865427)。基本是 [4] 的推导整理，符号不太统一。
- <a id="ref-13"></a>**[13] 重建项为什么是 MSE / BCE**：Stack Exchange 上的两个问答：[Help understanding reconstruction loss in VAE](https://stats.stackexchange.com/questions/323568/help-understanding-reconstruction-loss-in-variational-autoencoder)，[Why reconstruction term is same to square loss?](https://stats.stackexchange.com/questions/347378/variational-autoencoder-why-reconstruction-term-is-same-to-square-loss)。另有两个高斯 KL 的推导：[KL divergence between two univariate Gaussians](https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians)。
- <a id="ref-14"></a>**[14] 多元伯努利分布**：Dai et al. *Multivariate Bernoulli distribution*. [arxiv.org/abs/1206.1874](https://arxiv.org/abs/1206.1874)。灰度图用 BCE 的严格处理见 Loaiza-Ganem, Cunningham. *The continuous Bernoulli*. NeurIPS 2019. [arxiv.org/abs/1907.06845](https://arxiv.org/abs/1907.06845)。
- <a id="ref-15"></a>**[15] 代码**：PyTorch 官方示例 [pytorch/examples/vae](https://github.com/pytorch/examples/tree/main/vae)，对 VAE 原论文的复现，第 4.1 节的代码来自这里；另一个实现 [LiUzHiAn/VAE-Pytorch](https://github.com/LiUzHiAn/VAE-Pytorch)，与官方示例几乎一致。
- <a id="ref-16"></a>**[16] Conv-VAE 博客**：*Building a Convolutional VAE in PyTorch*. [towardsdatascience.com](https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71)。
- <a id="ref-17"></a>**[17] VQ-VAE 原论文**：van den Oord, Vinyals, Kavukcuoglu. *Neural Discrete Representation Learning*. NeurIPS 2017. [arxiv.org/abs/1711.00937](https://arxiv.org/abs/1711.00937)。量化、三项 loss、梯度直通、PixelCNN 先验、EMA 码本更新（附录）都出自这篇。
- <a id="ref-18"></a>**[18] VQ-VAE-2**：Razavi, van den Oord, Vinyals. *Generating Diverse High-Fidelity Images with VQ-VAE-2*. NeurIPS 2019. [arxiv.org/abs/1906.00446](https://arxiv.org/abs/1906.00446)。
- <a id="ref-19"></a>**[19] VQGAN**：Esser, Rombach, Ommer. *Taming Transformers for High-Resolution Image Synthesis*. CVPR 2021. [arxiv.org/abs/2012.09841](https://arxiv.org/abs/2012.09841)。
- <a id="ref-20"></a>**[20] 后续的视觉 tokenizer**：VideoGPT [arxiv.org/abs/2104.10157](https://arxiv.org/abs/2104.10157)；MAGVIT-v2（*Language Model Beats Diffusion: Tokenizer is Key to Visual Generation*）[arxiv.org/abs/2310.05737](https://arxiv.org/abs/2310.05737)；FSQ（*Finite Scalar Quantization: VQ-VAE Made Simple*）[arxiv.org/abs/2309.15505](https://arxiv.org/abs/2309.15505)。
- <a id="ref-21"></a>**[21] β-VAE**：Higgins et al. *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR 2017. [openreview.net/forum?id=Sy2fzU9gl](https://openreview.net/forum?id=Sy2fzU9gl)。

## 0. 全文路线与符号约定

| 节 | 内容 |
|---|---|
| 1 | 出发点：普通 AutoEncoder 的 latent 有什么问题 |
| 2 | 数学引子：$$p(x)$$ 的积分为什么算不出来，为什么需要 Encoder |
| 3 | ELBO 的推导、重建项和 KL 项、从 ELBO 到实际 loss、重参数化 |
| 4 | 代码实现、完整训练、实验（含 loss 尺度写错的后果） |
| 5 | Conv-VAE |
| 6 | VQ-VAE：离散 latent、三项 loss、梯度直通、先验、实验 |
| 7 | 后续工作：β-VAE、VQ-VAE-2、VQGAN、FSQ、视频 tokenizer；要点回顾 |

**符号约定**（第 3、4 节的 VAE；第 6 节的 VQ-VAE 符号在 6.2 节另列）：

| 符号 | 含义 | 代码 | shape（第 4 节的 MLP VAE） |
|---|---|---|---|
| $$x$$ | 一张图像，像素值在 $$[0, 1]$$ | `x` | `[B, 784]` |
| $$z$$ | latent 向量，维度 $$d$$ | `z` | `[B, 20]` |
| $$p(z) = \mathcal{N}(0, I)$$ | 先验 | `torch.randn` | — |
| $$q_\phi(z \mid x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$$ | Encoder，参数 $$\phi$$，近似后验 | `encode` | — |
| $$\mu,\ \log\sigma^2$$ | Encoder 输出的均值和对数方差 | `mu`, `logvar` | `[B, 20]` |
| $$p_\theta(x \mid z)$$ | Decoder，参数 $$\theta$$，似然 | `decode` | 输出 `[B, 784]` |
| $$\hat x$$ | Decoder 输出的均值（伯努利时是每个像素为 1 的概率） | `recon_x` | `[B, 784]` |
| $$\epsilon \sim \mathcal{N}(0, I)$$ | 重参数化用的标准噪声 | `eps` | `[B, 20]` |
| $$\mathcal{L}(q, \theta)$$ | 证据下界 ELBO，训练时最小化 $$-\mathcal{L}$$ | `loss_function` 的返回值取负 | 标量 |

注意第 2 节沿用了 CS231n 课件的写法：Decoder 的参数是 $$\theta$$，Encoder 的参数是 $$\phi$$，全文一致。

## 1. 背景和初衷 Background

首先用例子来介绍 VAE 的初衷 [[9]](#ref-9)。回顾普通的 AutoEncoder：如下例，Encoder 可以把输入图片编码成 latent code。一种"理想"的 latent code 应该能够表示这张图片的**核心属性**，其中每个值表达一个属性，例如下图中的 Smile、Skin、Gender、Beard、Glasses、Hair Color 等。这些属性值再通过 Decoder 还原出原始图片。

![AutoEncoder 用固定值表示每个属性](/deep-learning/assets/generative-models/Untitled%2016.png)

上图是用一个固定的值来记录一种属性。但通常情况下，对于一个属性，我们更倾向于用**一个范围**的值来表示它，而不是单独的一个值。以 Smile 属性为例，不同的原始图片显然应该对应不同的值，这些值应该服从某种概率分布才比较合理。如下图：

![用一个分布表示每个属性](/deep-learning/assets/generative-models/Untitled%2017.png)

如果用一个范围的值来表达 latent code 中的每个属性，那么 Decoder 做的就是从 latent code 的分布中采样出 latent vector，再进行还原。

![从每个属性的分布中采样再解码](/deep-learning/assets/generative-models/Untitled%2018.png)

因此，我们期待的 Encoder 应该能够输出一个服从某种分布要求（例如高斯分布）的 latent distribution，这样才能更好地对输入数据进行区分，并且更可控地进行重建。

![Encoder 输出每个属性的分布](/deep-learning/assets/generative-models/Untitled%2019.png)

再看一个例子 [[10]](#ref-10)。下图是普通 AutoEncoder 的 Encoder 在 MNIST 数据集上得到的某一个 latent 维度的分布（换一次训练，分布可能又会不同）：

![AutoEncoder 的 latent 分布随训练的变化](/deep-learning/assets/generative-models/Untitled%2020.png)

右图可以看出，这个分布在训练初期大致以 0 为中心，但负数值已经明显偏多。随着训练进行，分布越来越向负数方向偏移。左图给出了分布的直方图：负数值多很多，而且整个分布集中在负数的一个很小的区域，使得整个分布区间中出现了很大的空白（gap）。

这种分布的问题在于：如果 latent code 的分布过于集中且不均匀，出现很大的 gap，那么给定一个落在 gap 中的 latent code，Decoder 很可能重建出一个奇怪的结果。我们通常希望：相似的输入数据，latent code 也比较接近，形成类似聚类的效果；而全部数据的 latent code 最好能连续地占据整个空间。如下图：

![希望的 latent 分布：相似样本聚在一起、整体连续](/deep-learning/assets/generative-models/Untitled%2021.png)

因此，我们希望得到的 latent code **不但能够满足重建的需求**（即 Decoder 还原出的数据和原始输入尽可能相似，这是普通 AutoEncoder 能做到的），同时还希望 latent code 的分布满足一个先验要求，例如服从标准正态分布（这是 VAE 要做到的）。这就是 VAE 的初衷和要解决的问题。满足了这一点，就可以**从先验分布里随便采一个 $$z$$ 交给 Decoder，生成一张新的图**，AutoEncoder 就变成了生成模型。

下图从另一个角度说明 VAE 学到的东西：一个映射，把服从正态分布的一批数据（左图）映射到服从数据规律的另一种分布（右图）。

![把正态分布映射成数据分布](/deep-learning/assets/generative-models/Untitled%2022.png)

## 2. 数学引子 Intuitive in Math

和 [生成模型概览](/deep-learning/generative-models/) 中介绍的 FVBN 类似，VAE 同样想**估计显式的概率分布** $P(X)$，只是出发点不同 [[3]](#ref-3)。

首先的出发点是一个合理的假设：**训练数据** $X$ **是由一个潜在的、未观测到的 latent representation 生成的**，记为 $Z$（下图中用了小写；**本节用大写表示随机变量，小写表示它的一个取值**）。即存在一个真实的 latent code $Z$。例如 $X$ 是图片时，$Z$ 可以是用来生成 $X$ 的属性值、方向等信息，或者人脸图片对应的头发、表情、年龄等。如下图以及第 1 节中的例子。

![数据由未观测的 latent z 生成](/deep-learning/assets/generative-models/Untitled%2023.png)

不妨假设 $Z$ 有一个先验分布 $P(Z)$。一个合理的选择是标准正态分布 $\mathcal{N}(0, I)$，用来描述头发长短、年龄大小、笑容大小这类连续属性。那么如何生成 $X$？如下图，可以设计一个 Decoder，记它的参数为 $\theta$：先从 $Z$ 的分布中采样得到一个取值 $z$，再将它输入 Decoder 得到 $X$。

![用 Decoder 网络表示 p(x|z)](/deep-learning/assets/generative-models/Untitled%2024.png)

我们的最终目标始终是：**让模型生成的 X 尽可能像真实数据**。用数学语言表示，目标和 FVBN 一致：最大化 $P(X)$ 这个似然，只是此时它由 Decoder 网络 $\theta$ 决定：

$$
P(X)=\int P(X,z)\,dz = \int P(z)\,P(X\vert z; \theta)\,dz
$$

这里要积分，是因为只观测到了 $X$ 而不知道 $Z$，只能对所有可能的 $z$ 求积分（边缘化）。第二个等号是概率的乘法公式 $P(X, z) = P(z) P(X \vert z)$。

问题在于：虽然已经假定 $P(Z)$ 是高斯分布，也可以用 Decoder 计算 $P(X\vert z;\theta)$，**但这个积分无法计算**：$z$ 是连续的高维变量，不可能遍历所有取值。即便用 Monte Carlo 方法从 $P(Z)$ 中采很多个 $z$ 来近似，对绝大多数 $z$ 来说 $P(X \vert z)$ 几乎为 0（随便采一个 $z$ 解码出来的图和给定的 $X$ 几乎无关），需要的样本数大得不可接受。

换一种思路，用贝叶斯公式：

$$
P(X)=\frac{P(X\vert Z)P(Z)}{P(Z\vert X)}
$$

其中各个量的含义：

- $P(X)$：数据的概率分布，又称为**证据（Evidence）**。我们想最大化它的似然。
- $P(X\vert Z)$：似然（likelihood），即给定 latent code $Z$ 时生成 $X$ 的概率，由 Decoder 计算。
- $P(Z)$：先验分布，VAE 中直接假设为 $\mathcal{N}(0, I)$。
- $P(Z\vert X)$：后验分布（posterior），即给定 $X$ 时 $Z$ 的分布。它也算不出来，因为分母里正好需要 $P(X)$。如何近似它是 VAE 的重点。

做法是**再设计一个 Encoder**，从 $X$ 预测 $Z$ 的分布，记为 $Q(Z\vert X;\phi)$，参数为 $\phi$。VAE 的一个目标就是用 $Q(Z\vert X)$ 来近似真实的后验 $P(Z\vert X)$。

![Encoder 近似后验、Decoder 表示似然](/deep-learning/assets/generative-models/Untitled%2025.png)

接下来证明：有了这个 Encoder，就能写出 $\log P(X)$ 的一个可以计算的**下界**，VAE 实际最大化的就是这个下界。下一节给出推导。

## 3. VAE 原理推导

### 3.1 VAE 的推导以及证据下界 ELBO

本节推导主要参考 [[6]](#ref-6)，也可以对照 [[1]](#ref-1) 第 2 节和 [[2]](#ref-2)。

根据前面介绍的原理，我们已知 $X$ 的一批观测值 $\mathcal{D}$，要最大化它的概率 $P(X)$。VAE 的出发点是把 $X$ 编码为 latent 变量 $Z$，再从 $Z$ 还原出 $X$。$x$ 的概率是联合分布 $p(x, z)$ 对所有 $z$ 的积分：

$$
p(x) = \int_z p(x,z)\, dz
$$

在推导前，先定义：

- Decoder 的参数是 $\theta$。Decoder 从 $Z$ 解码到 $X$，对应似然 $p_\theta(x\vert z)$；
- Encoder 的参数是 $\phi$。Encoder 从 $X$ 编码到 $Z$，对应近似后验 $q_\phi(z\vert x)$。

根据最大似然理论，求解模型参数可以通过**极大化观测样本的出现概率**实现。如果 $X,Z$ 都能观测到，即数据集是 $\mathcal{D}=\{(z^{(1)},x^{(1)}),\cdots,(z^{(N)},x^{(N)})\}$，那么要极大化的目标函数是

$$
\ell(\theta;\mathcal{D}) = \sum_{i=1}^{N} \log p_{\theta}(x^{(i)},z^{(i)})
$$

这里 $\theta$ 写在下标，表示这是由模型参数 $\theta$ 决定的概率。

但实际上我们不知道模型参数 $\theta$，也观测不到 $Z$（它是**隐变量**），样本集只有 $\mathcal{D}=\{x^{(1)},\cdots,x^{(N)}\}$。所以要最大化的概率变成了对全部可能的 $Z$ 做积分：

<span id="eq-vae-1"></span>

$$
\ell(\theta;\mathcal{D}) = \sum_{i=1}^{N} \log p_{\theta}(x^{(i)})
= \sum_{i=1}^{N} \log  \int_z p_{\theta}(x^{(i)},z)\, dz \tag{1}
$$

这个式子中有积分，直接最大化很难。一个思路是找一个可以计算的替代函数，使它始终小于等于上式，然后最大化这个替代函数。因为 $p(x)$ 被称为证据（Evidence），这个下界函数就被称为 **Evidence Lower Bound (ELBO)，即证据下界**。

下面推导这个下界。为了公式简洁，**只看 <a href="#eq-vae-1">Eq. (1)</a> 中的单个样本** $x$（对所有样本求和不影响过程和结论）。推导的核心是 Jensen 不等式（[Jensen's inequality](https://zh.wikipedia.org/wiki/%E7%B0%A1%E6%A3%AE%E4%B8%8D%E7%AD%89%E5%BC%8F)）。

要最大化的是 $x$ 的对数概率（对 latent $z$ 边缘化）：

$$
 \ell(\theta;x) = \log p_\theta(x)
$$

于是有：

$$
\begin{aligned}
\ell(\theta;x) &=  \log  p_{\theta}(x)\\ &=  \log \int_{z} p_{\theta}(x,z)\,dz\\&=  \log \int_{z} q_{\phi}(z\vert x) \frac{p_{\theta}(x,z)}{q_{\phi}(z\vert x)}\,dz  \ \ \ ;\text{同时乘除 } q_{\phi}(z\vert x)\text{，值不变}\\&=  \log \mathbb{E}_{q_{\phi}(z\vert x) } \left [ \frac{p_{\theta}(x,z)}{q_{\phi}(z\vert x)} \right ] \ \ \ ;\text{期望的定义}\\
& \ge  \mathbb{E}_{q_{\phi}(z\vert x) } \log \left [ \frac{p_{\theta}(x,z)}{q_{\phi}(z\vert x)} \right ]
\ \ \ ;\text{Jensen 不等式}\\
&=  \int_{z} q_{\phi}(z\vert x) \log \left [ \frac{p_{\theta}(x,z)}{q_{\phi}(z\vert x)} \right ]dz\\&=    \int_{z} q_{\phi}(z\vert x) \log  p_{\theta}(x,z)\,dz
-   \int_{z} q_{\phi}(z\vert x) \log q_{\phi}(z\vert x)\,dz \\&\triangleq \mathcal{L}(q,\theta)
\end{aligned}
$$

其中 Jensen 不等式是：如果 $\varphi$ 是**凸函数**，那么

$$
\varphi(\mathbb{E}[X]) \le \mathbb{E}[\varphi(X)]
$$

如果 $\varphi$ 是**凹函数**，不等号反过来，变成 $\ge$。

回到上面的推导，第 5 行就是对凹函数 $\log$ 应用 Jensen 不等式得到的。整个式子的结果就是 **ELBO 下界函数**，记为 $\mathcal{L}(q,\theta)$：

<span id="eq-vae-2"></span>

$$
\begin{aligned}\mathcal{L}(q,\theta) &= \int_{z} q_{\phi}(z\vert x) \log  p_{\theta}(x,z)\,dz-   \int_{z} q_{\phi}(z\vert x) \log q_{\phi}(z\vert x)\,dz\\&=   \mathbb{E}_{z \sim q_{\phi}} [ \log  p_{\theta}(x,z) ]- \mathbb{E}_{z \sim q_{\phi}}[ \log q_{\phi}(z\vert x) ]\end{aligned}\tag{2}
$$

这里把期望的下标 $z \sim q_{\phi}(z\vert x)$ 简写为 $z \sim q_{\phi}$。

根据乘法公式，联合概率 $p(x,z)$ 有两种分解方式

$$
p(x,z) = p(z) p(x\vert z)=p(x)p(z\vert x)
$$

下面分别用这两种分解来展开 ELBO，即 <a href="#eq-vae-2">Eq. (2)</a>。

**（1）第一种形式：使用 $p(x,z) = p(x) p(z\vert x)$**

$$
\begin{aligned}    
\mathcal{L}(q,\theta) &= \mathbb{E}_{z \sim q_{\phi} } [ \log  p_{\theta}(x,z) ]    - \mathbb{E}_{z \sim q_{\phi} }[ \log q_{\phi}(z\vert x)]\\    &= \mathbb{E}_{z \sim q_{\phi}} \left [ \log  p_{\theta}(x) + \log  p_{\theta}(z\vert x) \right ]        - \mathbb{E}_{z \sim q_{\phi}}[ \log q_{\phi}(z\vert x)]\\    &= \underbrace{\mathbb{E}_{z \sim q_{\phi}} [ \log  p_{\theta}(x) ]}_{\text{与}z\text{无关，期望符号可以去掉}}        + \mathbb{E}_{z \sim q_{\phi}} [\log  p_{\theta}(z\vert x) ]        - \mathbb{E}_{z \sim q_{\phi}}[ \log q_{\phi}(z\vert x)]\\    &= \underbrace{\log  p_{\theta}(x)}_{\text{观测数据对数似然/证据}}        +        \underbrace{        \mathbb{E}_{z \sim q_{\phi}} [ \log  p_{\theta}(z\vert x) ]        - \mathbb{E}_{z \sim q_{\phi}}[ \log q_{\phi}(z\vert x)] }_{=\,-KL(q_\phi(z\vert x)\,\Vert\, p_\theta(z\vert x))}\\    &= \ell(\theta;x)        - KL( q_{\phi}(z\vert x) \Vert p_{\theta}(z\vert x)) \end{aligned}
$$

最后一行的第二项直接套用了 KL 散度的定义（可以参考 [数学基础](/deep-learning/math-foundation/) 中的 KL Divergence 部分）：$KL(q \Vert p) = \mathbb{E}_{q}[\log q - \log p]$。倒数第二行括号里是 $\mathbb{E}_q[\log p - \log q]$，正好差一个负号，所以写成 $-KL$。

移项整理后：

$$
\underbrace{ \ell(\theta;x)  }_{\text{观测数据对数似然/证据}}= \underbrace{ \mathcal{L}(q,\theta) }_{\text{证据下界函数 ELBO}}+  \underbrace{  KL(q_{\phi}(z\vert x)\Vert p_{\theta}( z \vert x ) )}_{ \text{KL散度}\ \ge\ 0}
$$

所以我们想最大化的 $\ell(\theta;x)$ 等于 ELBO 加上一个 KL 散度，而 KL 散度始终非负，这就再次验证了 ELBO 是下界。这个式子还说明了**下界和真实值之间差了多少**：差的正好是近似后验 $q_\phi(z \vert x)$ 和真实后验 $p_\theta(z \vert x)$ 之间的 KL 散度。Encoder 越准，下界越紧。

**（2）第二种形式：使用 $p(x,z) = p(z)p(x\vert z)$**

展开 ELBO 表达式 <a href="#eq-vae-2">Eq. (2)</a> 得到

<span id="eq-vae-3"></span>

$$
\begin{aligned}    
\mathcal{L}(q,\theta) &= \mathbb{E}_{z \sim q_{\phi}} [ \log  p_{\theta}(x,z) ]    - \mathbb{E}_{z \sim q_{\phi}}[ \log q_{\phi}(z\vert x)]\\    &= \mathbb{E}_{z \sim q_{\phi}} [ \log  p_{\theta}(x\vert z) + \log p(z)]        - \mathbb{E}_{z \sim q_{\phi}}[ \log q_{\phi}(z\vert x)]\\        &= \mathbb{E}_{z \sim q_{\phi}} [\log  p_{\theta}(x\vert z) ] +        \underbrace{        \mathbb{E}_{z \sim q_{\phi}} [ \log  p(z) ]        - \mathbb{E}_{z \sim q_{\phi}}[ \log q_{\phi}(z\vert x)] }_{=\,-KL(q_\phi(z\vert x)\,\Vert\, p(z))}\\    &=   \underbrace{\mathbb{E}_{z \sim q_{\phi}} [\log  p_{\theta}(x\vert z) ]}_{重建项(\textbf{reconstruction term})}        - \underbrace{KL(q_{\phi}(z\vert x)\Vert p(z))}_{正则项(\mathbf{regularization \ term})}
\end{aligned}\tag{3}
$$

即，ELBO 由一个重建项和一个 KL 散度的正则项组成。重建项是 $z \sim q_\phi(z\vert x)$ 时 $\log p_\theta(x\vert z)$ 的期望；正则项是近似后验和先验之间的 KL 散度。第一种形式说明了 ELBO 为什么是下界，第二种形式则是**可以计算的**：重建项由 Decoder 给出，正则项在高斯假设下有闭式解（3.2 节），训练时用的就是这一种。

### 3.2 VAE Loss 计算

总结上一节：要最大化的 $\ell(\theta;x)$ 被转换成了最大化 ELBO $\mathcal{L}(q,\theta)$，即最大化 <a href="#eq-vae-3">Eq. (3)</a> 中的两项之和：重建项越大越好，KL 项越小越好。两项之间是此消彼长的关系，3.3 节会专门讨论。

下面分别计算 <a href="#eq-vae-3">Eq. (3)</a> 中的这两项。

**（1）第一项重建项的计算**

第一项是

$$
\mathbb{E}_{z \sim q_{\phi}} [\log p_{\theta}(x\vert z)]
$$

给定 $z$ 时，$X$ 仍然是一个随机变量，$p_\theta(x \vert z)$ 是它的分布。Decoder 的作用是根据 $z$ 输出这个分布的参数。通常 Decoder 只预测分布的**均值** $\mu_x$，不预测方差 $\Sigma_x$，而是把方差设为常数（例如单位方差 $I$），这样模型更简单；大部分回归模型也是这样假设的。即

$$
\mu_x = \mu_{\theta}(z) = \operatorname{decoder}_{\theta}(z)
$$

这里 $\mu_x$ 是 $X$ 的均值参数，$\mu_\theta(z)$ 是 $z$ 到 $\mu_x$ 的映射。（上一节 CS231n 的图里 Decoder 同时输出了 $\mu_{x\vert z}$ 和 $\Sigma_{x\vert z}$，是更一般的写法；实践中几乎都只输出均值。）因此，最大化 $\log p_\theta(x\vert z)$ 就相当于让 Decoder 的输出，即均值 $\hat{x}$，接近实际观测值 $x$，也就是最小化一个 Reconstruction Loss。

Loss 的具体形式取决于假设的似然分布。记 $\hat{x}$ 是 Decoder 的输出，$x$ 是观测值，$x_i$ 表示第 $i$ 个像素。VAE 论文 [[1]](#ref-1) 给出了两种常见形式：

(1) 如果 $X$ 是二值数据（例如二值图像），Decoder 通常设计成 MLP + Sigmoid。Sigmoid 把每个输出值 $\hat{x}_i$ 限制在 $[0,1]$ 之间，表示"第 $i$ 个像素为 1 的概率"。于是可以认为 $P(X\vert Z)$ 是**多元伯努利分布**（multivariate Bernoulli distribution）[[14]](#ref-14)，各像素相互独立：

$$
P(X\vert Z)=\prod_i \hat{x}_i^{x_i}(1-\hat{x}_i)^{1-x_i}
$$

取对数后，连乘变成求和：

$$
\log P(X\vert Z)=\sum_i \Big[ x_i \log \hat{x}_i + (1-x_i) \log (1-\hat{x}_i) \Big]
$$

这正是 **Binary Cross Entropy Loss（BCE Loss）对所有像素求和后的相反数**。因此，最大化这一项就相当于最小化 BCE Loss。BCE 的定义见 PyTorch 文档 [BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)。

MNIST、FashionMNIST 的像素其实是 $[0,1]$ 之间的灰度值，不是严格的 0/1。实践中照样用 BCE，把灰度值当作"软标签"，效果很好；但严格来说这时它不再是一个归一化的概率分布，严格的处理见 continuous Bernoulli [[14]](#ref-14)。

(2) 如果 $X$ 是普通的连续数据（如一般的信号、自然图像），通常假设给定 $z$ 时 $X$ 服从高斯分布，Decoder 预测这个高斯分布的**均值**，方差取一个固定的超参数 $\sigma^2$（不预测方差是为了模型简单）。即 $P(X\vert Z) = \mathcal{N}(\hat{X}, \sigma^2 I)$。这里的高斯分布是一个**建模假设**：Decoder 是非线性网络，输入 $z$ 服从高斯分布并不意味着输出也服从高斯分布，高斯只是对"给定 $z$ 时，真实数据在 $\hat x$ 附近的波动"的假设。根据 $k$ 维高斯分布的公式（$\Sigma = \sigma^2 I$，$k$ 是像素个数）：

$$
P(X\vert Z) = \frac{1}{((2\pi)^k \vert\Sigma\vert)^\frac{1}{2}} e^{-\frac{1}{2 \sigma^2}(X-\hat{X})^T(X-\hat{X})}
$$

取 log 后：

$$
\log P(X\vert Z) = {-\frac{1}{2\sigma^2}(X-\hat{X})^T(X-\hat{X})} + C, \qquad C = -\frac{k}{2}\log(2\pi\sigma^2)
$$

其中 $C$ 是与参数无关的常数项。这里写成了向量内积形式，$(X-\hat{X})^T(X-\hat{X}) = \sum_i (x_i - \hat x_i)^2$ 已经包含了对所有像素的求和，它就是 $X$ 和 $\hat{X}$ 之间的平方误差（L2 Loss / **MSE Loss** 的求和形式）。注意该项前面的负号。

于是，最大化这一项相当于最小化平方误差。注意前面还有一个系数 $\frac{1}{2\sigma^2}$，它在 3.3 节非常重要。

**（2）第二项 KL 散度的计算**

第二项 $KL(q_{\phi}(z\vert x)\Vert p(z))$ 是两个分布之间的 KL 散度。VAE 中先验已经假定为 $p(z) = \mathcal{N} (0,I)$。近似后验 $q_\phi(z\vert x)$ 的形式由我们选择，VAE 选择的也是高斯分布（这是一个设计选择，即"变分分布族"，并不是因为先验是高斯所以后验必然是高斯），并且各维相互独立：$q_\phi(z\vert x) =\mathcal{N} (\mu_z,\Sigma_z)$，$\Sigma_z = \operatorname{diag}(\sigma_{z,1}^2, \dots, \sigma_{z,d}^2)$，均值和方差由 Encoder 根据 $x$ 预测。

在传统的 AutoEncoder 中，$Z$ 是一个确定的数值，Encoder 直接输出 $Z$ 的值。而在 VAE 中，$Z$ 是一个随机变量，Encoder 不直接输出 $Z$，而是输出 $Z$ 的分布的均值和方差，再从这个分布中采样得到 $z$。

已知 $q_\phi(z\vert x) = \mathcal{N} (\mu_z,\Sigma_z)$ 和 $p(z) = \mathcal{N} (0,I)$ 这两个高斯分布，代入 [数学基础](/deep-learning/math-foundation/) 中两个多维高斯分布之间 KL 散度的表达式 [[13]](#ref-13)，可以得到：

$$
\begin{aligned}
KL(q_{\phi}(z\vert x)\Vert p(z)) &= \frac{1}{2} \left[ \log \frac{\vert\Sigma_p\vert}{\vert\Sigma_q\vert} - d + \operatorname{tr}(\Sigma_p^{-1} \Sigma_q) + (\mu_p - \mu_q)^\top \Sigma_p^{-1} (\mu_p - \mu_q) \right] \ \ \ ;\ q\text{ 指 } q_\phi(z\vert x)\text{，}p\text{ 指 } p(z)\\
&= \frac{1}{2} \left( - \log \vert\Sigma_z\vert -d + \operatorname{tr}(\Sigma_z) + \mu_z^T\mu_z \right)
\end{aligned}
$$

其中 $d$ 是 $z$ 的维度。这是一般多维高斯的形式。

如果 $z$ 是一维变量，即 $q_\phi(z\vert x) = \mathcal{N} (\mu_z,\sigma_z^2)$、$p(z) = \mathcal{N} (0,1)$，KL 散度为：

$$
\begin{aligned}
KL(q_{\phi}(z\vert x)\Vert p(z)) &= \log \frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2\sigma_p^2}  - \frac{1}{2} \\
& =-\log \sigma_z+\frac{1}{2}(\sigma_z^2+\mu_z^2 - 1) \\
&=-\frac{1}{2}(1 + \log \sigma_z^2 - \sigma_z^2 - \mu_z^2)
\end{aligned}
$$

<div class="note-callout">

注意：上面多维高斯的公式对任意协方差矩阵都成立。VAE 中 $\Sigma_z$ 是对角矩阵（各维独立），此时 $\log\vert\Sigma_z\vert = \sum_j \log\sigma_{z,j}^2$，$\operatorname{tr}(\Sigma_z) = \sum_j \sigma_{z,j}^2$，多维公式正好化简为 $d$ 个一维结果之和：$KL = -\frac12\sum_{j=1}^{d}\left(1 + \log\sigma_{z,j}^2 - \sigma_{z,j}^2 - \mu_{z,j}^2\right)$。代码里就是对每一维套用一维公式，再对 $d$ 维求和。

</div>

### 3.3 VAE Loss 的最终形式

代码中总是最小化 loss，所以把最大化 ELBO 改写成最小化它的相反数：

<span id="eq-vae-4"></span>

$$
\mathcal{L}_{\text{VAE}}(x) = -\mathcal{L}(q,\theta) = \underbrace{-\mathbb{E}_{z \sim q_\phi(z\vert x)}\big[\log p_\theta(x\vert z)\big]}_{\text{重建 loss}} + \underbrace{KL\big(q_\phi(z\vert x)\,\Vert\, p(z)\big)}_{\text{KL loss}} \tag{4}
$$

所以 VAE 的 loss 就是**重建 loss + KL 散度**。但从这个式子到真正能写进代码的 loss，还有四处需要说清楚。很多实现在这里出错，导致"理论推出来的 loss"和"代码里的 loss"看起来对不上。

**（1）期望用一个样本近似。** 重建项是对 $q_\phi(z\vert x)$ 的期望，没有闭式解。VAE 论文 [[1]](#ref-1) 用 Monte Carlo 估计：从 $q_\phi(z \vert x)$ 中采 $L$ 个样本取平均，

$$
\mathbb{E}_{z \sim q_\phi(z\vert x)}\big[\log p_\theta(x\vert z)\big] \approx \frac{1}{L}\sum_{l=1}^{L}\log p_\theta(x\vert z^{(l)}), \qquad z^{(l)} = \mu + \sigma \odot \epsilon^{(l)},\ \ \epsilon^{(l)} \sim \mathcal{N}(0, I)
$$

论文指出 batch 足够大（例如 100）时 $L = 1$ 就够了，所以代码里每张图只采**一个** $z$，算一次重建 loss。采样用的是 3.4 节的重参数化。KL 项有闭式解，不需要采样。

**（2）重建 loss 的形式和系数由似然决定。** 由 3.2 节：

| 似然 $$p_\theta(x \vert z)$$ | 重建 loss $$-\log p_\theta(x\vert z)$$ | 代码 |
|---|---|---|
| 伯努利（每个像素为 1 的概率是 $$\hat x_i$$） | $$\sum_i \operatorname{BCE}(\hat x_i, x_i)$$，没有多余的系数 | `F.binary_cross_entropy(recon_x, x, reduction='sum')` |
| 高斯 $$\mathcal{N}(\hat x, \sigma^2 I)$$ | $$\frac{1}{2\sigma^2}\sum_i (x_i - \hat x_i)^2 + \text{常数}$$ | `((recon_x - x) ** 2).sum() / (2 * sigma2)` |

高斯情形下，常数可以丢掉，但**系数 $$\frac{1}{2\sigma^2}$$ 不能随便丢**：它决定了重建 loss 和 KL loss 的相对权重。把 loss 乘以 $$2\sigma^2$$（不改变最优解），得到

$$
\sum_i (x_i - \hat x_i)^2 + 2\sigma^2 \cdot KL\big(q_\phi(z\vert x)\,\Vert\, p(z)\big)
$$

也就是说，"平方误差 + KL"只是 $$\sigma^2 = \frac12$$ 时的特例。$$\sigma^2$$ 越大，相当于认为数据噪声越大，KL 的权重越大，模型越不愿意把信息写进 $$z$$。这正是 β-VAE [[21]](#ref-21) 的形式 $$\text{重建} + \beta \cdot KL$$，其中 $$\beta = 2\sigma^2$$。

**（3）对像素和 latent 维度求和，而不是求平均。** ELBO 是**一个样本**的对数似然的下界：重建项要对 $$x$$ 的全部像素（这里 784 个）求和，KL 项要对 $$z$$ 的全部 $$d$$ 维求和。对 batch 求和还是求平均都可以（只差一个常数倍，相当于调学习率），但重建项和 KL 项必须用同样的方式。PyTorch 的 `F.mse_loss`、`F.binary_cross_entropy` 默认 `reduction='mean'`，会对像素也求平均，重建项被缩小 784 倍，等价于 $$\sigma^2 = 784 / 2 = 392$$：模型认为图像噪声的方差是 392（而像素值只在 $$[0, 1]$$ 之间），于是完全放弃重建，只去满足 KL。4.3 节的实验会看到，这时 KL 降到 0，所有 latent 维度都不再携带信息，Decoder 对任何输入都输出同一张"平均图"。这种现象叫**后验坍缩（posterior collapse）**。

**（4）两种分解并不矛盾。** 3.1 节的第一种形式说，最大化 ELBO 会让 $$q_\phi(z\vert x)$$ 接近**真实后验** $$p_\theta(z\vert x)$$；第二种形式里却有一项要让 $$q_\phi(z\vert x)$$ 接近**先验** $$p(z)$$。两者看似冲突，其实第二种形式中的 KL 项从来不是单独最小化的：

- 如果只最小化 $$KL(q_\phi(z\vert x)\Vert p(z))$$，最优解是对所有 $$x$$ 都有 $$q_\phi(z\vert x) = p(z)$$，$$z$$ 和 $$x$$ 无关，重建项会变得很差。这正是上面的后验坍缩；
- 如果只最大化重建项，Encoder 会把方差压到接近 0，退化成普通的 AutoEncoder，latent 空间出现第 1 节说的空白，从先验里采的 $$z$$ 解码出来会很奇怪；
- 两项的和才等于第一种形式中的 $$\log p_\theta(x) - KL(q_\phi(z\vert x)\Vert p_\theta(z\vert x))$$。最优的 $$q_\phi(z\vert x)$$ 是真实后验：对每个 $$x$$ 来说比先验"窄"、带着 $$x$$ 的信息；但对所有数据平均后，$$\int q_\phi(z\vert x)\,p_{\text{data}}(x)\,dx$$ 接近先验 $$\mathcal{N}(0, I)$$，所以才能从先验采样生成。

**最终形式。** 对伯努利 Decoder（本文代码用的就是这一种），一个 batch 的 loss 是

<span id="eq-vae-5"></span>

$$
\mathcal{L}_{\text{VAE}} = \frac{1}{B}\sum_{b=1}^{B}\left[\ \sum_{i=1}^{784}\operatorname{BCE}\big(\hat x_i^{(b)}, x_i^{(b)}\big)\ \underbrace{-\ \frac{1}{2}\sum_{j=1}^{d}\Big(1 + \log\sigma_{j}^{(b)2} - \mu_{j}^{(b)2} - \sigma_{j}^{(b)2}\Big)}_{KL}\ \right],\qquad \hat x^{(b)} = \operatorname{decoder}_\theta(\mu^{(b)} + \sigma^{(b)} \odot \epsilon^{(b)}) \tag{5}
$$

PyTorch 官方示例 [[15]](#ref-15) 对 batch 也求和（不除以 $$B$$），两者只差一个常数倍。这个 loss 的数值本身也有意义：它是 $$-\log p_\theta(x)$$ 的一个上界，单位是 nats（自然对数）。

### 3.4 Reparameterization Trick 重参数化

代码中 VAE 的 Encoder + Decoder 的流程是：

1. 从 $X$ 出发，用 Encoder 预测正态分布 $q_\phi(z\vert x)$ 的参数，记为 $\mathcal{N} (\mu, \sigma^2)$；
2. 从正态分布 $\mathcal{N} (\mu, \sigma^2)$ 中采样，得到 $z$；
3. 将 $z$ 作为 Decoder 的输入，重建出 $\hat{X}$。

整个流程如下图：

![采样 z 的位置](/deep-learning/assets/generative-models/Untitled%2026.png)

这里有一个问题：$z$ 是从 $q_\phi(z\vert x)$ 中**随机采样**得到的，而"采样"这个操作对分布的参数 $\mu, \sigma$ 不可导。代码实现要求每一步都可导（differentiable），才能做反向传播，把重建 loss 的梯度传回 Encoder。

VAE 用 reparameterization trick 解决这个问题，利用的是高斯分布的性质 [[1]](#ref-1)：

- 先用 Encoder 网络预测参数 $\mu, \sigma^2$；
- 再令 $z=\mu + \sigma \odot\epsilon$，其中 $\epsilon \sim \mathcal{N} (0,I)$ 是从标准正态分布中采样的一个随机变量，它和 Encoder 的参数无关。

$\epsilon$ 和 $\phi$ 无关，所以 $z = \mu_\phi + \sigma_\phi \odot \epsilon$ 对 $\phi$ 可导：随机性全部来自一个"外部输入" $\epsilon$，$\mu$ 和 $\sigma$ 只参与普通的乘法和加法。这样既保证了 $z$ 服从 $\mathcal{N} (\mu, \sigma^2)$，又使这一步可导，问题就解决了。

实际上，从任意高斯分布 $\mathcal{N} (\mu,\sigma^2)$ 采样，标准做法本来就是先采 $\epsilon \sim \mathcal{N} (0,1)$ 再计算 $z=\mu + \sigma\epsilon$。重参数化只是把这一步显式地写进计算图里。

![重参数化：epsilon 作为计算图的输入](/deep-learning/assets/generative-models/Untitled%2027.png)

### 3.5 VAE 最终流程图

![VAE 整体流程](/deep-learning/assets/generative-models/Untitled%2028.png)

### 3.6 总结：VAE 的训练和生成

| | 训练（每个 batch） | 生成 |
|---|---|---|
| 1 | Encoder 输出 $$\mu,\ \log\sigma^2 = \operatorname{encoder}_\phi(x)$$ | 从先验采样 $$z \sim \mathcal{N}(0, I)$$ |
| 2 | 采 $$\epsilon \sim \mathcal{N}(0, I)$$，$$z = \mu + \sigma \odot \epsilon$$（重参数化，$$L = 1$$） | Decoder 输出 $$\hat x = \operatorname{decoder}_\theta(z)$$ |
| 3 | Decoder 输出 $$\hat x = \operatorname{decoder}_\theta(z)$$ | 把 $$\hat x$$（每个像素的概率或均值）直接当作生成的图 |
| 4 | 重建 loss：像素求和的 BCE（或 $$\frac{1}{2\sigma^2}$$ 倍的平方误差） | |
| 5 | KL loss：$$-\frac12\sum_j(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$ | |
| 6 | 两项相加取 batch 平均，反向传播，同时更新 $$\phi$$ 和 $$\theta$$ | |

生成时不需要 Encoder。Encoder 只在训练时用来近似后验，让 Decoder 学会把先验中的 $$z$$ 映射成像样的数据。

## 4. VAE 代码实现

### 4.1 核心代码

下面是 PyTorch 官方示例 [[15]](#ref-15) 中 VAE 类的核心代码以及 VAE Loss 代码，注释标出了每一行对应的公式：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)        # Encoder 隐层
        self.fc21 = nn.Linear(400, 20)        # 输出 μ，d = 20
        self.fc22 = nn.Linear(400, 20)        # 输出 log σ²
        self.fc3 = nn.Linear(20, 400)         # Decoder 隐层
        self.fc4 = nn.Linear(400, 784)        # 输出 x̂ 的 logits

    def encode(self, x):                      # q_φ(z|x) 的参数
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)   # μ, log σ²   [B, 20]

    def reparameterize(self, mu, logvar):     # z = μ + σ ⊙ ε
        std = torch.exp(0.5 * logvar)         # σ = exp(½ log σ²)
        eps = torch.randn_like(std)           # ε ~ N(0, I)
        return mu + eps * std

    def decode(self, z):                      # p_θ(x|z)：每个像素为 1 的概率 x̂
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))    # [B, 784]，范围 (0, 1)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)   # 每张图只采一个 z（L = 1）
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # 重建项：-log p_θ(x|z) = 对 784 个像素求和的 BCE（伯努利似然）
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KL 项：-½ Σ_j (1 + log σ_j² - μ_j² - σ_j²)，对 d 维和 batch 都求和
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD                          # = -ELBO（对 batch 求和）
```

几个注意要点：

- VAE 的 Encoder 用 Linear 或 Conv 等层把输入编码成均值和方差。实现中 Encoder 输出的通常是 $\mu$ 和 $\log{\sigma^2}$，而不是 $\sigma$ 或 $\sigma^2$ 本身，原因是：
    1. **方差必须为正，而网络的输出可以是任意实数。** 让网络输出 $\log\sigma^2 \in (-\infty, +\infty)$，再用 $\exp$ 变回 $\sigma^2$，自动保证为正，不需要额外的约束或截断；
    2. **数值更稳定，也方便计算 Loss。** KL 项里本来就有 $\log\sigma^2$，直接用网络输出即可，不必对一个可能非常接近 0 的 $\sigma^2$ 再取对数；需要 $\sigma^2$ 时用 `logvar.exp()` 即可。
- 当 Encoder 预测得到 $\log \sigma^2$ 后，标准差为 $e^{0.5\cdot \log{\sigma^2}}=e^{\log (\sigma^2)^{0.5}}=\sigma$，即代码中的 `torch.exp(0.5 * logvar)`。

### 4.2 完整的训练和生成

下面在 FashionMNIST 上训练这个 VAE。和 4.1 的唯一区别是 Decoder 返回 logits（不加 sigmoid），loss 用 `binary_cross_entropy_with_logits`，数值上更稳定（sigmoid 输出接近 0 或 1 时，`binary_cross_entropy` 内部的 $$\log$$ 容易溢出）：

```python
def vae_loss(logits, x, mu, logvar):
    """Eq. (5)：返回 batch 平均的 -ELBO 以及它的两部分。x: [B, 784]，范围 [0, 1]"""
    B = x.shape[0]
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")   # Σ_b Σ_i BCE
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())            # Σ_b Σ_j KL_j
    return (rec + kl) / B, rec / B, kl / B


model = VAE()                     # decode 改为返回 self.fc4(h3)，即 logits
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(15):
    for x, _ in train_loader:     # x: [B, 1, 28, 28]，ToTensor() 之后范围 [0, 1]
        x = x.view(-1, 784)
        logits, mu, logvar = model(x)
        loss, rec, kl = vae_loss(logits, x, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()

# 生成：从先验采样，只用 Decoder
with torch.no_grad():
    z = torch.randn(64, 20)                        # z ~ N(0, I)
    samples = torch.sigmoid(model.decode(z))       # [64, 784]，每个像素的概率，直接当灰度图
    samples = samples.view(64, 1, 28, 28)
```

> **易错点**
>
> 1. **`reduction` 要用 `'sum'`。** 重建项必须对像素求和（3.3 节第 3 点）。用默认的 `'mean'` 会让重建项缩小 784 倍，模型直接后验坍缩，见 4.3 节。如果想用平均，就让重建项和 KL 项都除以同一个数（例如 batch 大小），不要只对其中一项求平均。
> 2. **KL 对 latent 维度求和。** `torch.sum` 会把 $$d$$ 维和 batch 都加起来。写成 `torch.mean` 同样会改变两项的比例。
> 3. **`logvar` 是 $$\log\sigma^2$$，不是 $$\log\sigma$$。** 标准差是 `exp(0.5 * logvar)`。写成 `exp(logvar)` 不会报错，但采样的方差和 KL 里用的方差对不上。
> 4. **生成时不加噪声、不用 Encoder。** 从 $$\mathcal{N}(0, I)$$ 采 $$z$$ 直接解码，输出的 $$\hat x$$ 是伯努利分布的均值。有的实现再从伯努利分布里采一次像素，得到的是满是噪点的二值图，一般不需要。
> 5. **重建时用 $$\mu$$。** 想看"这张图被重建成什么样"时，用 `decode(mu)` 而不是 `decode(z)`，后者带有采样噪声，每次结果略有不同。
> 6. **`F.binary_cross_entropy` 要求输入在 $$[0, 1]$$。** 如果数据做了 `Normalize((0.5,), (0.5,))`，范围变成 $$[-1, 1]$$，BCE 的公式就不成立了（PyTorch 会直接报错）。VAE 用 BCE 时，只用 `ToTensor()`。

### 4.3 实验：正确的 loss 和尺度写错的 loss

在 FashionMNIST 上训练 4.2 节的模型（$$d = 20$$，batch 128，Adam 学习率 $$10^{-3}$$，15 个 epoch，CPU 上不到 1 分钟），对比两种 loss：

- **BCE(sum) + KL**：Eq. (5)，即正确的 $$-$$ELBO；
- **MSE(mean) + KL**：很多实现里的写法，`F.mse_loss(recon_x, x)` 用默认的 `reduction='mean'`，KL 照常求和。

在测试集上统一用 Eq. (5) 的两项来评估（单位：nats / 图像）：

| loss | 重建项（BCE） | KL 项 | 携带信息的 latent 维度 |
|---|---|---|---|
| BCE(sum) + KL | 227.0 | 16.1 | 14 / 20 |
| MSE(mean) + KL | 385.3 | 0.0 | 0 / 20 |

"携带信息的维度"指测试集上平均 KL 大于 0.01 的维度：KL 接近 0 的维度，$$q_\phi(z_j\vert x)$$ 对所有 $$x$$ 都等于 $$\mathcal{N}(0, 1)$$，这一维什么也没编码。正确的 loss 下 20 维里用到了 14 维，其余 6 维也"坍缩"到先验，这是 VAE 的正常现象：维度够用时，多余的维度没必要付出 KL 代价。尺度写错的 loss 下，**全部 20 维都坍缩了**：

<figure class="sd-img">
<img src="/deep-learning/assets/generative-models-1-vae/vae_recon_samples.png" alt="两种 loss 下的重建和从先验采样的结果">
<figcaption>第 1 行：测试图像。第 2、3 行：正确的 BCE(sum) + KL，重建保留了每件衣服的类别和轮廓，从 N(0, I) 采样能生成各种类别。第 4、5 行：MSE(mean) + KL，重建和采样都是同一张模糊的"平均图"，Decoder 完全忽略了 z。</figcaption>
</figure>

为了说明系数 $$\frac{1}{2\sigma^2}$$（等价地，$$\beta = 2\sigma^2$$）的作用，把 Decoder 换成固定方差的高斯似然，loss 为 $$\frac{1}{2\sigma^2}\lVert x - \hat x\rVert^2 + KL$$（对像素和维度求和），扫描 $$\sigma^2$$（每个 10 个 epoch）：

| $$\sigma^2$$ | 等价的 $$\beta = 2\sigma^2$$ | 重建平方误差（每张图） | KL（nats） | 携带信息的维度 |
|---|---|---|---|---|
| 0.01 | 0.02 | 8.2 | 54.5 | 20 / 20 |
| 0.1 | 0.2 | 9.6 | 21.9 | 19 / 20 |
| 0.5 | 1 | 13.8 | 8.5 | 8 / 20 |
| 2 | 4 | 20.3 | 3.6 | 4 / 20 |
| 392 | 784 | 68.0 | 0.0 | 0 / 20 |

$$\sigma^2$$ 越小，重建越准、KL 越大、用到的维度越多；$$\sigma^2$$ 越大，模型越倾向于让 $$q_\phi(z\vert x)$$ 贴近先验。$$\sigma^2 = 392$$ 正好对应"MSE 取平均"的写法，结果和上表一样完全坍缩。实践中 $$\sigma^2$$（或 $$\beta$$）是一个需要调的超参数：太小，latent 空间不连续，采样质量差；太大，重建模糊甚至坍缩。

最后把 latent 维度设为 $$d = 2$$，可以直接画出 latent 空间：

<figure class="sd-img">
<img src="/deep-learning/assets/generative-models-1-vae/vae_latent2d.png" alt="二维 latent 空间">
<figcaption>左：测试集每张图的编码均值 μ(x)，按类别着色。同类聚在一起，整体大致占据 N(0, I) 的高概率区域，没有第 1 节那样的大片空白。右：在 z 平面的网格上解码（坐标取标准正态的分位点）。相邻位置的图像平滑过渡：鞋类在下方，上衣在上方，裤子在左侧，包在右侧，和左图的类别分布一致。</figcaption>
</figure>

## 5. VAE 变种: 卷积 VAE (Convolutional VAE, or Conv-VAE)

Convolutional VAE（Conv-VAE）的构成和普通 VAE 非常像 [[16]](#ref-16)，区别只是专门针对图像：把 Encoder 中的 MLP 换成卷积层（Convolutional layers），把 Decoder 中的 MLP 换成转置卷积层（Transposed Convolutional layers，即 deconvolution）。其余部分都不变，例如 loss 的计算、latent code 维度、重参数化等。

![Conv-VAE 结构](/deep-learning/assets/generative-models/Untitled%2029.png)

以 $$28 \times 28$$ 的灰度图为例：

```python
class ConvVAE(nn.Module):
    def __init__(self, z_dim=20, ch=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.ReLU(),            # [B, 32, 14, 14]
            nn.Conv2d(ch, 2 * ch, 4, 2, 1), nn.ReLU(),       # [B, 64, 7, 7]
            nn.Flatten())                                    # [B, 3136]
        self.fc_mu = nn.Linear(2 * ch * 7 * 7, z_dim)        # μ
        self.fc_logvar = nn.Linear(2 * ch * 7 * 7, z_dim)    # log σ²
        self.fc_dec = nn.Linear(z_dim, 2 * ch * 7 * 7)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (2 * ch, 7, 7)), nn.ReLU(),
            nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1), nn.ReLU(),   # [B, 32, 14, 14]
            nn.ConvTranspose2d(ch, 1, 4, 2, 1))                   # [B, 1, 28, 28]，logits

    def forward(self, x):                                    # x: [B, 1, 28, 28]
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(self.fc_dec(z)), mu, logvar          # loss 与 4.2 节相同，像素求和时要先 flatten
```

`ConvTranspose2d(k=4, s=2, p=1)` 正好把边长放大 2 倍，和 `Conv2d(k=4, s=2, p=1)` 的缩小一一对应。这里 latent 仍然是一个 $$d$$ 维向量：卷积只负责提取特征，最后仍要展平再经过全连接层得到 $$\mu, \log\sigma^2$$。Stable Diffusion 等模型里的 VAE 则把 latent 保留成一个小的特征图（例如 $$4 \times 64 \times 64$$），每个位置都有自己的 $$\mu, \log\sigma^2$$，KL 对所有位置和通道求和，见 [Latent Diffusion 篇](/deep-learning/latent-diffusion/)。

## 6. VQ-VAE：离散的 latent

### 6.1 为什么要离散的 latent

VQ-VAE（Vector Quantised-VAE）[[17]](#ref-17) 把 VAE 的连续高斯 latent 换成了**离散**的：Encoder 输出的每个向量都被替换成一个有限码本（codebook）中离它最近的那个向量，于是一张图被表示成一个由整数索引组成的小网格，就像一段由"视觉单词"组成的文本。这样做的动机有三点：

1. **很多数据天然是离散的。** 语言是一个个词，图像也可以描述为"这里有一只狗、那里是草地"这样离散的概念。离散的 latent 更容易被下游模型使用。
2. **避免后验坍缩。** 3.3 节看到，Decoder 足够强（例如 PixelCNN、Transformer 这类自回归 Decoder）时，VAE 很容易忽略 $$z$$。VQ-VAE 的 loss 中没有 KL 项把 $$q(z\vert x)$$ 往先验拉（原因见 6.2 节），latent 总会被用上。
3. **生成可以交给强大的自回归模型。** 把图像变成 token 序列后，"生成图像"就变成"生成 token 序列"，可以直接用 PixelCNN、Transformer 这类擅长离散序列的模型，再用 Decoder 把 token 还原成图像。DALL·E（第一版）、VQGAN、视频生成中的 tokenizer 都沿用了这个思路。

和 VAE 的另一个不同是：**VQ-VAE 的先验不是固定的**。训练时假设先验是均匀分布；训练完成后，再单独训练一个自回归模型来学习 token 的真实分布，作为生成时的先验（6.4 节）。

### 6.2 结构和公式

<figure class="ddpm-fig">
<svg viewBox="0 0 760 330" role="img" aria-label="VQ-VAE 结构示意">
  <defs>
    <marker id="va" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker>
    <marker id="vr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#d92d20"/></marker>
  </defs>
  <!-- codebook -->
  <rect x="300" y="10" width="160" height="70" rx="6" fill="#fff7ed" stroke="#f79009"/>
  <text x="380" y="28" text-anchor="middle" class="lane">码本 e ∈ ℝ^(K×D)</text>
  <rect x="318" y="38" width="124" height="10" fill="#fdb022"/><rect x="318" y="50" width="124" height="10" fill="#fec84b"/><rect x="318" y="62" width="124" height="10" fill="#fedf89"/>
  <text x="470" y="40" class="sm">e₁, …, e_K</text>
  <text x="470" y="55" class="sm">每行一个 D 维码字</text>
  <!-- x -->
  <rect x="14" y="140" width="64" height="64" fill="#f2f4f7" stroke="#98a2b3"/>
  <text x="46" y="177" text-anchor="middle" class="it">x</text>
  <text x="46" y="222" text-anchor="middle" class="sm">28×28</text>
  <!-- encoder -->
  <polygon points="92,132 152,150 152,194 92,212" fill="#eef4ff" stroke="#2a6fdb"/>
  <text x="122" y="176" text-anchor="middle">E</text>
  <text x="122" y="230" text-anchor="middle" class="sm">Encoder</text>
  <!-- z_e -->
  <g stroke="#2a6fdb" fill="#dbe7ff">
    <rect x="172" y="146" width="52" height="52"/>
  </g>
  <path d="M172,159 h52 M172,172 h52 M172,185 h52 M185,146 v52 M198,146 v52 M211,146 v52" stroke="#2a6fdb" stroke-width=".7"/>
  <text x="198" y="138" text-anchor="middle" class="it">z_e(x)</text>
  <text x="198" y="216" text-anchor="middle" class="sm">7×7×D</text>
  <!-- nearest neighbour -->
  <line x1="222" y1="146" x2="304" y2="78" stroke="#475467" marker-end="url(#va)"/>
  <text x="240" y="96" text-anchor="end" class="sm">找最近的码字</text>
  <text x="240" y="111" text-anchor="end" class="sm">k = argmin‖z_e − e_j‖</text>
  <!-- indices -->
  <rect x="330" y="146" width="52" height="52" fill="#fff" stroke="#f79009"/>
  <path d="M330,159 h52 M330,172 h52 M330,185 h52 M343,146 v52 M356,146 v52 M369,146 v52" stroke="#f79009" stroke-width=".7"/>
  <text x="336.5" y="157" text-anchor="middle" class="sm">3</text><text x="349.5" y="157" text-anchor="middle" class="sm">7</text><text x="362.5" y="157" text-anchor="middle" class="sm">7</text><text x="375.5" y="157" text-anchor="middle" class="sm">1</text>
  <text x="336.5" y="170" text-anchor="middle" class="sm">5</text><text x="349.5" y="170" text-anchor="middle" class="sm">2</text><text x="362.5" y="170" text-anchor="middle" class="sm">9</text><text x="375.5" y="170" text-anchor="middle" class="sm">4</text>
  <text x="356" y="216" text-anchor="middle" class="sm">索引 k，7×7 个整数</text>
  <line x1="226" y1="172" x2="326" y2="172" stroke="#475467" marker-end="url(#va)"/>
  <!-- lookup -->
  <line x1="440" y1="80" x2="458" y2="142" stroke="#475467" marker-end="url(#va)"/>
  <text x="462" y="104" class="sm">查表 e_k</text>
  <line x1="384" y1="172" x2="442" y2="172" stroke="#475467" marker-end="url(#va)"/>
  <!-- z_q -->
  <rect x="446" y="146" width="52" height="52" fill="#fef0c7" stroke="#f79009"/>
  <path d="M446,159 h52 M446,172 h52 M446,185 h52 M459,146 v52 M472,146 v52 M485,146 v52" stroke="#f79009" stroke-width=".7"/>
  <text x="474" y="138" class="it">z_q(x)</text>
  <text x="472" y="216" text-anchor="middle" class="sm">7×7×D</text>
  <!-- decoder -->
  <polygon points="518,150 578,132 578,212 518,194" fill="#ecfdf3" stroke="#12b76a"/>
  <text x="548" y="176" text-anchor="middle">D</text>
  <text x="548" y="230" text-anchor="middle" class="sm">Decoder</text>
  <line x1="500" y1="172" x2="516" y2="172" stroke="#475467" marker-end="url(#va)"/>
  <line x1="154" y1="172" x2="170" y2="172" stroke="#475467" marker-end="url(#va)"/>
  <line x1="80" y1="172" x2="90" y2="172" stroke="#475467" marker-end="url(#va)"/>
  <line x1="580" y1="172" x2="600" y2="172" stroke="#475467" marker-end="url(#va)"/>
  <rect x="604" y="140" width="64" height="64" fill="#f2f4f7" stroke="#98a2b3"/>
  <text x="636" y="177" text-anchor="middle" class="it">x̂</text>
  <!-- straight-through -->
  <path d="M472,200 C472,262 198,262 198,202" fill="none" stroke="#d92d20" stroke-dasharray="5 4" marker-end="url(#vr)"/>
  <text x="335" y="276" text-anchor="middle" class="sm" style="fill:#d92d20">反向传播：∂L/∂z_q 原样复制给 z_e（straight-through）</text>
  <!-- losses -->
  <text x="690" y="150" class="sm">重建 loss</text>
  <text x="690" y="166" class="sm">−log p(x|z_q)</text>
  <text x="380" y="305" text-anchor="middle" class="sm">码本 loss ‖sg[z_e] − e‖² 只更新码本；commitment loss β‖z_e − sg[e]‖² 只更新 Encoder</text>
</svg>
<figcaption>按 VQ-VAE 论文图 1 的思想重画。Encoder 输出的每个 D 维向量被替换成码本中最近的那一个；argmin 不可导，梯度从 z_q 直接跳到 z_e（红色虚线）。</figcaption>
</figure>

| 符号 | 含义 | 代码 | shape（本文实验） |
|---|---|---|---|
| $$z_e(x)$$ | Encoder 输出的连续特征 | `z_e` | `[B, D, 7, 7]`，$$D = 16$$ |
| $$e \in \mathbb{R}^{K \times D}$$ | 码本，$$K$$ 个 $$D$$ 维码字 | `codebook.weight` | `[64, 16]` |
| $$k$$ | 每个位置选中的码字索引 | `idx` | `[B, 7, 7]`，取值 $$0..K-1$$ |
| $$z_q(x) = e_k$$ | 量化后的特征 | `z_q` | `[B, D, 7, 7]` |
| $$\operatorname{sg}[\cdot]$$ | stop-gradient，前向不变、梯度为 0 | `.detach()` | — |
| $$\beta$$ | commitment loss 的权重 | `beta` | 0.25 |

**量化。** 对 $$z_e(x)$$ 的每个空间位置，找码本里欧氏距离最近的码字：

$$
k = \arg\min_{j \in \{1,\dots,K\}} \lVert z_e(x) - e_j \rVert_2, \qquad z_q(x) = e_k
$$

用 VAE 的语言说，近似后验是一个 one-hot 分布：$$q(z = k \vert x) = 1$$，其余为 0。这是确定性的，没有采样，也就不需要重参数化。

**为什么没有 KL 项。** 训练时取均匀先验 $$p(z = k) = 1/K$$。对一个 one-hot 的 $$q$$：

$$
KL(q(z\vert x)\,\Vert\, p(z)) = \sum_k q(k\vert x)\log\frac{q(k\vert x)}{p(k)} = 1 \cdot \log\frac{1}{1/K} = \log K
$$

它是与参数无关的常数，对优化没有作用，所以从 loss 中去掉了。也就是说，VQ-VAE 的 ELBO 只剩下重建项。

**三项 loss。** [[17]](#ref-17) 的训练目标为

<span id="eq-vq-1"></span>

$$
\mathcal{L}_{\text{VQ}} = \underbrace{-\log p\big(x \vert z_q(x)\big)}_{\text{重建 loss}} + \underbrace{\big\lVert \operatorname{sg}[z_e(x)] - e \big\rVert_2^2}_{\text{码本 loss}} + \underbrace{\beta\,\big\lVert z_e(x) - \operatorname{sg}[e] \big\rVert_2^2}_{\text{commitment loss}} \tag{6}
$$

其中 $$e$$ 指每个位置被选中的码字 $$e_k$$。三项各管一件事：

- **重建 loss** 更新 Encoder 和 Decoder（通过梯度直通到达 Encoder，见下）。它**更新不到码本**：前向用的是 $$e_k$$，但梯度直通跳过了量化这一步，码本收不到重建 loss 的梯度。
- **码本 loss** 只更新码本：把选中的码字 $$e_k$$ 拉向 Encoder 的输出 $$z_e(x)$$。$$z_e$$ 被 stop-gradient，所以这一项不影响 Encoder。效果类似 k-means：每个码字移动到分配给它的那些 $$z_e$$ 的中心。
- **commitment loss** 只更新 Encoder：让 $$z_e(x)$$ 靠近它选中的码字，"承诺"使用这个码字。没有这一项，码本的移动速度跟不上 Encoder 时，$$z_e$$ 可能不断变大、在不同码字之间来回跳。论文中 $$\beta = 0.25$$，并发现结果对 $$\beta$$ 在 0.1 到 2.0 之间不敏感。

**梯度直通（straight-through estimator）。** $$\arg\min$$ 不可导，重建 loss 的梯度无法从 $$z_q$$ 传到 $$z_e$$。VQ-VAE 的做法是：前向传播用 $$z_q$$，反向传播时**假装量化这一步是恒等映射**，把 $$\partial\mathcal{L}/\partial z_q$$ 原样作为 $$\partial\mathcal{L}/\partial z_e$$。因为 $$z_e$$ 和 $$z_q$$ 在 $$D$$ 维空间中很接近（commitment loss 保证了这一点），这个近似梯度的方向是合理的：它告诉 Encoder"如果输出往这个方向移动，重建会变好"，下一步量化时就可能选到更合适的码字。代码中一行实现：

$$
z_q^{\text{st}} = z_e + \operatorname{sg}[z_q - z_e]
$$

前向时它的值等于 $$z_q$$；反向时 $$\operatorname{sg}[\cdot]$$ 部分梯度为 0，对 $$z_e$$ 的导数是单位矩阵。

### 6.3 代码实现

```python
class VectorQuantizer(nn.Module):
    def __init__(self, K=64, D=16, beta=0.25):
        super().__init__()
        self.K, self.D, self.beta = K, D, beta
        self.codebook = nn.Embedding(K, D)                    # e: [K, D]
        self.codebook.weight.data.uniform_(-1 / K, 1 / K)

    def forward(self, z_e):                                   # z_e: [B, D, H, W]
        B, D, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).reshape(-1, D)            # 每个位置一个向量 [BHW, D]
        e = self.codebook.weight                              # [K, D]
        # ||z - e_j||² = ||z||² - 2 z·e_j + ||e_j||²，一次算出所有位置到所有码字的距离
        dist = z.pow(2).sum(1, keepdim=True) - 2 * z @ e.T + e.pow(2).sum(1)   # [BHW, K]
        idx = dist.argmin(1)                                  # k = argmin_j ||z_e - e_j||
        z_q = e[idx].view(B, H, W, D).permute(0, 3, 1, 2)     # z_q = e_k  [B, D, H, W]

        codebook_loss = F.mse_loss(z_q, z_e.detach())         # ||sg[z_e] - e||²，只更新码本
        commit_loss = F.mse_loss(z_e, z_q.detach())           # ||z_e - sg[e]||²，只更新 Encoder
        z_q_st = z_e + (z_q - z_e).detach()                   # 梯度直通：前向 = z_q，反向 ∂/∂z_e = I
        return z_q_st, codebook_loss + self.beta * commit_loss, idx.view(B, H, W)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x + self.net(x)


class VQVAE(nn.Module):
    def __init__(self, K=64, D=16, ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.ReLU(),             # 28 -> 14
            nn.Conv2d(ch, ch, 4, 2, 1),                       # 14 -> 7
            ResBlock(ch), ResBlock(ch), nn.ReLU(),
            nn.Conv2d(ch, D, 1))                              # z_e: [B, D, 7, 7]
        self.vq = VectorQuantizer(K, D)
        self.decoder = nn.Sequential(
            nn.Conv2d(D, ch, 3, padding=1),
            ResBlock(ch), ResBlock(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.ReLU(),   # 7 -> 14
            nn.ConvTranspose2d(ch, 1, 4, 2, 1))               # 14 -> 28，输出 logits

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, idx = self.vq(z_e)
        return self.decoder(z_q), vq_loss, idx

    def decode_indices(self, idx):                            # 生成时用：索引 [B, 7, 7] -> 图像
        z_q = self.vq.codebook(idx).permute(0, 3, 1, 2)       # 查表 [B, D, 7, 7]
        return torch.sigmoid(self.decoder(z_q))


# 训练一步：Eq. (6)
logits, vq_loss, _ = model(x)                                  # x: [B, 1, 28, 28]，范围 [0, 1]
rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)   # 每张图的重建 loss
loss = rec + 784 * vq_loss       # vq_loss 是逐元素平均，乘回到"每张图"的尺度，与 rec 对齐
```

结构和论文一致：两层步长为 2 的卷积、两个残差块，Decoder 对称；码本大小 $$K$$、码字维度 $$D$$ 都远小于论文中 ImageNet 的设置（$$K = 512$$），因为 FashionMNIST 简单得多。

> **易错点**
>
> 1. **`detach` 的位置决定了谁被更新。** 码本 loss 里 detach 的是 `z_e`，commitment loss 里 detach 的是 `z_q`，写反了就变成码本去追自己、Encoder 去追自己，两边都学不到东西。
> 2. **梯度直通要写成 `z_e + (z_q - z_e).detach()`。** 写成 `z_q.detach()` 会让 Encoder 收不到任何重建梯度；直接用 `z_q` 则梯度停在码本上（`e[idx]` 的梯度只到码本），Encoder 同样学不到重建。
> 3. **`permute` 之后再 `reshape`。** `z_e` 是 `[B, D, H, W]`，要先把 $$D$$ 换到最后一维，变成"每个位置一个 $$D$$ 维向量"；直接 `view(-1, D)` 会把不同位置的通道混在一起，不报错但结果错误。
> 4. **各项 loss 的尺度要匹配。** `F.mse_loss` 默认对所有元素求平均，而重建 loss 是对像素求和的。两者相差的倍数会改变 commitment 和码本 loss 的实际权重，和 3.3 节同理。本文把 VQ 项乘以 784 对齐到"每张图"的尺度；也可以都用平均，关键是比例一致。
> 5. **注意码本利用率。** 一部分码字可能从来没有被选中（dead code），它们收不到码本 loss 的梯度，永远不会被更新。训练时要监控用到了多少个码字和 perplexity（6.5 节），利用率太低时可以用 EMA 更新码本，或把长期未用的码字重置为随机的 $$z_e$$（6.6 节）。
> 6. **码本初始化的尺度。** 码字初始化得离 $$z_e$$ 的分布太远，开始时所有位置都会选中同一两个码字。论文和多数实现用很小的均匀分布初始化，也可以用第一个 batch 的 $$z_e$$ 做 k-means 初始化。

### 6.4 第二阶段：学习先验

训练完 VQ-VAE 后，每张训练图都可以变成一个 $$7 \times 7$$ 的索引网格。此时码本和 Decoder 冻结，再单独训练一个**自回归模型**来学习这些索引的分布 $$p(k_1, k_2, \dots, k_{49})$$：

$$
p(k_1, \dots, k_{N}) = \prod_{n=1}^{N} p(k_n \vert k_1, \dots, k_{n-1})
$$

即按光栅顺序（从左到右、从上到下）把网格展开成长度为 49 的序列，每一步根据前面的 token 预测下一个 token 的类别分布，用交叉熵训练，和语言模型完全一样。论文 [[17]](#ref-17) 用的是 PixelCNN，后续工作多用 Transformer。生成时：

1. 用自回归模型逐个采样 49 个索引；
2. 查码本得到 $$z_q$$；
3. 用 Decoder 解码成图像。

本文用一个 3 层的小 Transformer（causal attention，和 GPT 结构相同）：

```python
class TinyGPT(nn.Module):
    """先验 p(k_1..k_49)：把 7x7 索引网格展平成长度 49 的序列，做下一个 token 预测"""
    def __init__(self, K=64, L=49, d=128, n_layer=3, n_head=4):
        super().__init__()
        self.K, self.L = K, L
        self.tok = nn.Embedding(K + 1, d)                     # 多一个 id K 作为起始符 <BOS>
        self.pos = nn.Parameter(torch.zeros(1, L, d))         # 可学习的位置编码
        layer = nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d, K)
        # causal mask：第 n 个位置只能看到它之前的位置
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, seq):                                   # seq: [B, 49]，取值 0..K-1
        # 右移一位：输入是 <BOS>, k_1..k_48
        inp = torch.cat([torch.full_like(seq[:, :1], self.K), seq[:, :-1]], 1)
        h = self.tok(inp) + self.pos
        return self.head(self.blocks(h, mask=self.mask))      # [B, 49, K]：第 n 个输出预测 k_n

    @torch.no_grad()
    def sample(self, n, temperature=1.0):
        seq = torch.zeros(n, self.L, dtype=torch.long)
        for i in range(self.L):                               # 逐个位置采样，共 49 次前向
            logits = self(seq)[:, i] / temperature
            seq[:, i] = torch.multinomial(logits.softmax(-1), 1).squeeze(1)
        return seq


# 训练：tokens 是用冻结的 VQ-VAE 编码训练集得到的索引 [N, 49]
loss = F.cross_entropy(prior(tokens).reshape(-1, K), tokens.reshape(-1))

# 生成
idx = prior.sample(64).view(64, 7, 7)
images = vqvae.decode_indices(idx)                            # [64, 1, 28, 28]
```

生成时第 $$i$$ 步只取第 $$i$$ 个位置的输出：由于 causal mask，它只依赖于已经采好的前 $$i-1$$ 个 token，后面还没填的位置（初始为 0）不影响结果。

### 6.5 实验

在 FashionMNIST 上训练 6.3 节的 VQ-VAE（$$K = 64$$，$$D = 16$$，latent $$7 \times 7$$，batch 128，Adam 学习率 $$10^{-3}$$，8 个 epoch），再在训练集的索引上训练 6.4 节的先验（6 个 epoch）。CPU 上总共约 16 分钟。

训练过程中的测试集指标：

| epoch | 重建 BCE（nats / 图像） | 用到的码字数 | perplexity |
|---|---|---|---|
| 1 | 222.0 | 42 / 64 | 25.0 |
| 2 | 212.4 | 63 / 64 | 44.9 |
| 4 | 207.5 | 64 / 64 | 46.2 |
| 8 | 205.1 | 64 / 64 | 48.2 |

perplexity 定义为 $$\exp\big(-\sum_k p_k \log p_k\big)$$，$$p_k$$ 是码字 $$k$$ 在测试集所有位置上被选中的频率。它可以理解为"等效地均匀使用了多少个码字"：如果 64 个码字被同样频繁地使用，perplexity 就是 64；如果总是只用一个，就是 1。这里第 1 个 epoch 只用到 42 个码字，之后全部用上，perplexity 约 48，没有出现死码字。

几组数字的对比：

- 一张 $$28 \times 28$$ 的 8 位灰度图原始大小是 6272 bit；VQ-VAE 把它表示成 49 个索引，每个索引 $$\log_2 64 = 6$$ bit，共 294 bit，约 21 倍压缩；
- 先验在测试集上的交叉熵是 1.80 nats / token，即每张图约 127 bit。也就是说，学到的先验把"索引的组合"进一步压缩到了均匀编码（294 bit）的一半以下，它学到了哪些索引组合才像一件衣服；
- 重建 BCE 205.1，低于第 4.3 节 VAE 的 227.0。但两者不是公平比较：VAE 的 20 维 latent 实际携带的信息只有 KL 项的 16 nats（约 23 bit），而 VQ-VAE 的 latent 最多能携带 294 bit。这里的重点是：离散的 token 网格保留了空间结构，能表示比单个向量多得多的细节。

<figure class="sd-img">
<img src="/deep-learning/assets/generative-models-1-vae/vq_recon.png" alt="VAE 和 VQ-VAE 的重建对比">
<figcaption>第 1 行：测试图像。第 2 行：第 4.3 节的 VAE 用编码均值 μ 重建，只保留了大致的类别和轮廓，凉鞋变成了实心的鞋、印花消失。第 3 行：VQ-VAE 用 7×7 个索引重建，镂空的凉鞋、毛衣上的图案、包的提手都保留了下来。</figcaption>
</figure>

<figure class="sd-img">
<img src="/deep-learning/assets/generative-models-1-vae/vq_samples.png" alt="VQ-VAE 的生成结果">
<figcaption>上：用 6.4 节的 Transformer 先验逐个采样 49 个索引，再查码本、解码，生成了裤子、上衣、包、运动鞋等不同类别。下：49 个索引各自从 64 个码字中均匀随机抽取，解码出来是没有意义的纹理拼贴。码本和 Decoder 完全相同，区别只在先验。这说明 VQ-VAE 本身不能直接生成，必须有第二阶段的先验。</figcaption>
</figure>

### 6.6 常用的改进

**EMA 更新码本。** 论文附录 [[17]](#ref-17) 给出了码本 loss 的一种替代：不用梯度，而是用指数滑动平均（EMA）直接把每个码字设为分配给它的 $$z_e$$ 的均值。对第 $$j$$ 个码字，设当前 batch 中有 $$n_j$$ 个向量 $$\{z_{e,1}, \dots, z_{e,n_j}\}$$ 被分配给它，衰减率 $$\gamma = 0.99$$：

$$
N_j \leftarrow \gamma N_j + (1-\gamma)\,n_j, \qquad m_j \leftarrow \gamma m_j + (1-\gamma)\sum_{i=1}^{n_j} z_{e,i}, \qquad e_j = \frac{m_j}{N_j}
$$

这就是在线版的 k-means。此时 loss 中只保留重建项和 commitment 项。EMA 收敛更快、更稳定，是大多数开源实现的默认选择：

```python
@torch.no_grad()
def ema_update(self, z, idx, gamma=0.99, eps=1e-5):          # z: [BHW, D]，idx: [BHW]
    onehot = F.one_hot(idx, self.K).type(z.dtype)            # [BHW, K]
    self.N.mul_(gamma).add_(onehot.sum(0), alpha=1 - gamma)  # N_j：每个码字被分到的向量数（滑动平均）
    self.m.mul_(gamma).add_(onehot.T @ z, alpha=1 - gamma)   # m_j：分到的向量之和（滑动平均）  [K, D]
    n = self.N.sum()
    N = (self.N + eps) / (n + self.K * eps) * n              # Laplace 平滑，避免除以 0
    self.codebook.weight.copy_(self.m / N.unsqueeze(1))      # e_j = m_j / N_j
```

**重置死码字。** 定期找出最近很少被选中的码字，把它们替换成当前 batch 中随机挑选的 $$z_e$$，可以显著提高码本利用率。

**改进的距离和码本维度。** 后续工作（如 ViT-VQGAN）发现，先把 $$z_e$$ 和码字都做 L2 归一化（即用余弦相似度找最近邻），并在低维空间（例如 32 维甚至更低）里做查找，码本利用率会高很多。

## 7. 后续工作

### 7.1 VQ-VAE 的后续：VQ-VAE-2、VQGAN、FSQ

| 方法 | 相对 VQ-VAE 的改动 | 作用 |
|---|---|---|
| VQ-VAE-2 [[18]](#ref-18) | 两层（top / bottom）latent 层级，先验用带 self-attention 的 PixelCNN | 顶层管全局结构、底层管细节，$$256 \times 256$$ 的生成质量接近 BigGAN |
| VQGAN [[19]](#ref-19) | 重建 loss 加上感知损失（LPIPS）和 PatchGAN 判别器；先验换成 Transformer | 压缩率更高时细节仍然清晰；它的自编码器后来被 [Latent Diffusion](/deep-learning/latent-diffusion/) 沿用（VQ 正则的版本） |
| FSQ [[20]](#ref-20) | 不要码本：把 $$z_e$$ 的每一维有界化后直接四舍五入到少数几个整数（例如 $$\{-2,\dots,2\}$$），各维的组合就是隐式的码本 | 不需要码本 loss、commitment loss 和 EMA，几乎不存在死码字 |
| MAGVIT-v2 的 LFQ [[20]](#ref-20) | 码字维度降到 0：每一维只取 $$\pm 1$$，码本大小 $$2^{18}$$ | 支持很大的码本；配合视频 tokenizer，让语言模型式的生成在图像和视频上超过扩散模型 |
| VideoGPT [[20]](#ref-20) | 用 3D 卷积的 VQ-VAE 把视频压成时空 token，再用 GPT 做先验 | 把 VQ-VAE + 自回归先验的两阶段思路直接搬到视频上 |

MAGVIT-v2 [[20]](#ref-20) 中对 VQ-VAE 的概括可以作为这一节的总结：视觉 tokenizer 的作用是把像素映射到一个适合生成模型的离散表示。VQ-VAE 是图像 tokenization 的奠基工作，由 CNN Encoder、向量量化瓶颈和 CNN Decoder 组成；对视频 $$V \in \mathbb{R}^{T \times H \times W \times 3}$$，Encoder 输出 $$Z = E(V) \in \mathbb{R}^{T' \times H' \times W' \times d}$$，每个向量 $$z$$ 被量化为码本 $$C \in \mathbb{R}^{K \times d}$$ 中最近的码字 $$q(z) = c_i,\ i = \arg\min_j \lVert z - c_j \rVert_2$$；去掉嵌入维度，用索引 $$X \in \{1, \dots, K\}^{T' \times H' \times W'}$$ 表示视频，就得到了离散 token；Decoder 再由这些 token 的嵌入重建 $$\hat V = D(Z)$$。

### 7.2 VAE 本身的变体

| 方法 | 改动 | 要点 |
|---|---|---|
| β-VAE [[21]](#ref-21) | loss 改为 $$\text{重建} + \beta \cdot KL$$，$$\beta > 1$$ | 更强地把各维推向独立的先验，学到更"解耦"的 latent（每一维对应一个属性），代价是重建变模糊；3.3 节说明了 $$\beta$$ 和高斯似然中 $$\sigma^2$$ 的等价关系 |
| Conditional VAE (CVAE) | Encoder 和 Decoder 都额外输入条件 $$c$$（如类别）：$$q_\phi(z\vert x, c)$$、$$p_\theta(x\vert z, c)$$ | 生成时指定 $$c$$ 就能生成指定类别 |
| SD 中的 KL-VAE | latent 保留为 $$4 \times 64 \times 64$$ 的特征图，KL 权重很小（约 $$10^{-6}$$），重建 loss 加上感知损失和 GAN 损失 | 目的不是从先验采样，而是做一个几乎无损的压缩器，生成交给扩散模型；见 [Latent Diffusion 篇](/deep-learning/latent-diffusion/) |

### 7.3 要点回顾

- **VAE 在最大化什么？** 对数似然 $$\log p_\theta(x)$$ 的下界 ELBO $$= \mathbb{E}_{q_\phi(z\vert x)}[\log p_\theta(x\vert z)] - KL(q_\phi(z\vert x)\Vert p(z))$$。下界和真实值之间差 $$KL(q_\phi(z\vert x)\Vert p_\theta(z\vert x))$$。
- **loss 怎么写？** $$-\text{ELBO}$$ = 对像素求和的重建 loss + 对 latent 维度求和的 KL。重建期望用一个样本近似；KL 有闭式解 $$-\frac12\sum_j(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$。
- **为什么 MSE 的写法容易出问题？** 高斯似然的重建项是 $$\frac{1}{2\sigma^2}\lVert x - \hat x\rVert^2$$，系数决定了和 KL 的权衡（$$\beta = 2\sigma^2$$）；对像素取平均相当于 $$\sigma^2 = 392$$，会导致后验坍缩。
- **重参数化解决什么？** 采样不可导；写成 $$z = \mu + \sigma\odot\epsilon$$ 后，随机性来自外部输入 $$\epsilon$$，梯度可以传回 Encoder。
- **VQ-VAE 和 VAE 的区别？** latent 是码本中的离散索引；后验是确定的 one-hot，KL 是常数 $$\log K$$ 而被去掉；loss 是重建 + 码本 + commitment 三项；argmin 不可导，用梯度直通；先验在第二阶段用自回归模型单独学。
