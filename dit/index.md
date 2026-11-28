---
date: 2026-11-28 02:00:00 -0800
tags: [Diffusion, Transformer]
keywords: >-
  DiT, Diffusion Transformer, Scalable Diffusion Models with Transformers, ViT, patchify, unpatchify, 2D sin-cos position embedding, adaLN, adaLN-Zero, in-context conditioning, cross-attention, Gflops, scaling, latent diffusion, learned sigma, SiT, PixArt-alpha, MM-DiT, SD3, FLUX, Sora, PyTorch.
summary: >-
  DiT（Diffusion Transformer）把扩散模型的 U-Net 主干换成标准 Transformer：图像（latent）切成 patch 变成 token，经过若干个用 adaLN-Zero 注入时间和类别条件的 Transformer block，再拼回图像。本文介绍它要解决的问题、ViT 背景、patchify 与位置编码、四种条件注入方式和 adaLN-Zero 的零初始化、模型配置与计算量的缩放规律，给出从零实现的完整 PyTorch 代码（参数量和计算量与论文 12 个配置逐一吻合），在 FashionMNIST 上用 Flow Matching 训练小 DiT、和 U-Net 及四种条件注入方式对比，最后梳理 SiT、PixArt-α、SD3 的 MM-DiT、FLUX 和视频 DiT。
layout: note
title: "DiT：用 Transformer 做扩散模型的主干网络"
description: "Diffusion Transformers from scratch: why replace the U-Net, patchify, adaLN-Zero conditioning, model configs and scaling, a verified PyTorch implementation, FashionMNIST experiments, and the path to SD3, FLUX and video DiTs."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig svg .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;fill:#475467}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
</style>

# DiT：用 Transformer 做扩散模型的主干网络

前面几篇讲的都是扩散模型"怎么训练、怎么采样"：[DDPM](/deep-learning/ddpm-from-scratch/) 预测噪声，[DDIM](/deep-learning/ddim/) 解 ODE，[Flow Matching](/deep-learning/flow-matching/) 预测速度。这些方法对网络只有一个要求：**输入带噪的图像 $$z_t$$ 和条件（时间 $$t$$、类别或文本），输出一个同样大小的张量**。从 DDPM 到 Stable Diffusion 1.5/2/XL，这个网络一直是 U-Net。

DiT（Diffusion Transformer，Peebles 和 Xie，2023）[[1]](#ref-1) 把 U-Net 换成了一个几乎不加修改的标准 Transformer。它证明了 U-Net 的卷积结构不是扩散模型必需的，并且 Transformer 主干的生成质量随计算量稳定提升。此后的 Sora、Stable Diffusion 3、FLUX、Movie Gen、Wan 等主流图像和视频生成模型，主干都是 DiT 的变体。

本文的记号和前面各篇一致：扩散在 latent 空间进行时，输入记作 $$z_t$$（和 [Stable Diffusion](/deep-learning/stable-diffusion/)、[LDM](/deep-learning/latent-diffusion/) 篇相同）；DiT 论文里记作 $$x_t$$。像素空间的实验中 $$z_t$$ 就是图像本身。

## 参考资料

本文主要参考 DiT 原论文 [1] 和官方代码 [2]，ViT 的背景来自 [3]；第 8 节的后续工作参考 SiT [4]、PixArt-α [5] 和 SD3 [6]。正文中用 [n] 标注具体出处。

- <a id="ref-1"></a>**[1] 主论文**：Peebles, Xie. *Scalable Diffusion Models with Transformers*. ICCV 2023. [arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)。模型结构、四种条件注入方式的对比、配置表和缩放实验都来自这篇。
- <a id="ref-2"></a>**[2] 官方代码**：[facebookresearch/DiT](https://github.com/facebookresearch/DiT)。只有几百行，`models.py` 是整个模型。本文的实现按它逐模块对应，初始化方式也相同。
- <a id="ref-3"></a>**[3] ViT**：Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)。把图像切成 patch 当作 token 的做法出自这篇。
- <a id="ref-4"></a>**[4] SiT**：Ma et al. *SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers*. ECCV 2024. [arxiv.org/abs/2401.08740](https://arxiv.org/abs/2401.08740)。网络和 DiT 完全相同，只把训练目标换成 Flow Matching 一类的插值，在相同结构下效果更好。
- <a id="ref-5"></a>**[5] PixArt-α**：Chen et al. *PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis*. ICLR 2024. [arxiv.org/abs/2310.00426](https://arxiv.org/abs/2310.00426)。把 DiT 用到文生图：cross-attention 接文本，adaLN-single 减少参数。
- <a id="ref-6"></a>**[6] SD3**：Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. ICML 2024. [arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)。MM-DiT：文本 token 和图像 token 各用一套权重，在同一个注意力里交互。

## 背景

本节介绍后文用到的基础知识和基本概念。熟悉 ViT 和 Stable Diffusion U-Net 结构的读者可以直接阅读第 0 节。

### 前置知识

- **Transformer**：self-attention、多头注意力、LayerNorm、残差和前馈网络。见 [Transformer 学习](/deep-learning/transformer-study/)。DiT 只用到 encoder 部分（没有 causal mask，也没有 decoder）。
- **U-Net 主干**：时间嵌入、ResBlock、下采样/上采样、skip connection、cross-attention。见 [Stable Diffusion 拆解](/deep-learning/stable-diffusion/) 第 5、6 节和 [DDPM](/deep-learning/ddpm-from-scratch/) 第 5 节。
- **latent 空间**：自编码器把 $$256 \times 256 \times 3$$ 的图像压成 $$32 \times 32 \times 4$$ 的 latent，扩散在 latent 里进行。见 [LDM](/deep-learning/latent-diffusion/)。
- **训练目标**：DiT 论文用 DDPM（预测噪声 $$\epsilon$$，外加学习方差），本文的实验用 [Flow Matching](/deep-learning/flow-matching/)（预测速度 $$\epsilon - z_0$$）。主干网络和训练目标是两件独立的事，第 6 节会说明换目标时网络只需改输出通道数。

### 为什么要用 DiT

#### 扩散模型主干网络的演化

从 2020 年到 2023 年，扩散模型的主干一直是 U-Net，并且越做越大、越改越复杂；2023 年之后，主流模型几乎全部换成了 Transformer：

| 模型 | 年份 | 主干 | 参数量 | 关键变化 |
|---|---|---|---|---|
| DDPM | 2020 | U-Net | 35.7M（CIFAR-10） | 来自 PixelCNN++ 的 U-Net，16×16 分辨率加自注意力 |
| ADM | 2021 | U-Net | 554M（ImageNet 256） | 大量结构搜索：更深、更多注意力头、在 32/16/8 三个分辨率加注意力、BigGAN 式残差块、AdaGN 注入条件 |
| LDM / SD 1.5 | 2022 | U-Net | 860M | 改在 latent 里做扩散；加 cross-attention 读文本 |
| SDXL | 2023 | U-Net | 2.6B | 重新分配各层的 Transformer 块（最高分辨率不放、最低分辨率放 10 个），去掉最深的一级下采样 |
| **DiT-XL/2** | 2023 | **Transformer** | 675M | 去掉卷积、下采样和跳连，标准 Transformer 处理 latent patch |
| SD3 | 2024 | MM-DiT | 2B–8B | 文本和图像 token 联合注意力，Rectified Flow |
| FLUX.1 | 2024 | MM-DiT + 单流 DiT | 12B | 前半部分双流、后半部分文本和图像共用权重；二维 RoPE |
| Sora、Movie Gen 等 | 2024 | 视频 DiT | Movie Gen 为 30B | 时空 patch |

这张表反映了一个趋势：U-Net 每做大一次，都要重新设计结构（ADM 的结构搜索、SDXL 的注意力块重新分配）；而 Transformer 的放大方式是固定的，只要增加层数、宽度或 token 数。

#### U-Net 的长处和局限

U-Net 并不是一个"差"的结构，它适合图像有明确的原因：

- **卷积的归纳偏置。** 卷积核只看邻近的像素（局部性），并且同一个卷积核在图像所有位置共享（平移等变性）。"相邻像素通常相似""同一个纹理出现在哪里都一样"这些关于图像的先验知识被直接写进了结构，网络不需要从数据里学。
- **多尺度。** 下采样到低分辨率后，每个位置代表原图的一大块区域，适合处理整体结构；跳连把高分辨率的细节直接送到输出端。去噪恰好需要这两种信息：高噪声时要决定整体布局，低噪声时要修正细节。
- **任意分辨率。** 卷积不依赖输入大小，换分辨率不需要改结构。

但随着模型和数据变大，它的局限越来越明显：

1. **设计空间大，全靠经验。** 每一级的通道数、每级几个残差块、哪几级加注意力、注意力头数、条件怎样注入，都是需要调的超参数。ADM 论文的大部分篇幅就是在做这些消融；SDXL 又重新调了一遍。模型规模变化后，原来的最优配置不一定还是最优。
2. **没有清楚的缩放规律。** 想用 10 倍的算力训练一个更好的模型时，U-Net 没有一个"往哪个方向加"的明确答案：加通道、加层数、还是在更高分辨率加注意力，效果都不一样，也很难预测。
3. **全局交互有限。** 卷积的感受野是局部的；自注意力的代价随位置数平方增长，所以 U-Net 只在低分辨率层加注意力，高分辨率层的像素之间不能直接交换信息，全局一致性要靠一次次下采样间接获得。
4. **难以和其他模态统一。** 文本、音频、视频帧都是序列。U-Net 只能通过 cross-attention 单向地"读"文本；做视频时要在空间卷积之外另加时间卷积和时间注意力层，结构越来越复杂。
5. **工程生态是分开的。** 2020 年以后，大规模训练的基础设施（FlashAttention、张量并行和序列并行、混合精度、各种推理优化）几乎都是围绕 Transformer 发展的。U-Net 的卷积和多分辨率结构很难直接用上这些工具。

#### Transformer 带来了什么

在 DiT 之前，Transformer 已经在其他领域显示出两个重要性质：

- **可预测的缩放规律。** 语言模型的研究（Kaplan 等，2020）发现，测试损失随参数量、数据量和计算量按幂律平稳下降，可以用小模型的结果预测大模型的效果。这让"花更多算力换更好模型"变成一件可规划的事。
- **数据足够多时，可以不要归纳偏置。** ViT [[3]](#ref-3) 的实验很有代表性：只在 ImageNet（128 万张图）上训练时，ViT 不如同等规模的 ResNet；在 ImageNet-21k（1400 万张）上训练时两者相当；在 JFT-300M（3 亿张）上训练时，ViT 超过 ResNet。论文的解释是：卷积的归纳偏置在数据少时很有用，数据多时，Transformer 能直接从数据中学到这些规律，并且学到的比人为写进去的更灵活。

此外，Transformer 的结构是**同一个 block 重复 $$L$$ 次**，所有层处理同样长度的 token 序列。放大模型只有三个旋钮：层数 $$L$$、维度 $$D$$、token 数 $$N$$（由 patch 大小决定），这让系统地研究缩放规律成为可能。

#### DiT 回答的两个问题

DiT 之前已经有工作尝试把 ViT 直接用作扩散模型的主干（如 GenViT），但生成质量明显不如 U-Net，同期的 U-ViT 则保留了 U-Net 式的长跳连。DiT [[1]](#ref-1) 系统地研究了 Transformer 主干的设计选择和缩放行为，回答了两个问题：

- **U-Net 的卷积归纳偏置对扩散模型是不是必需的？** 结论：不是。在 latent 上工作的标准 Transformer，在足够的训练规模下效果超过了当时所有 U-Net 扩散模型。
- **扩散模型能不能像语言模型一样，靠加大 Transformer 稳定变好？** 结论：能。生成质量（FID）和网络的计算量（Gflops）高度相关，无论计算量来自更深更宽的网络，还是来自更多的 token（更小的 patch）。

注意第一个结论里的"足够的训练规模"：DiT 的实验在 ImageNet 上训练 40 万到 700 万步。和 ViT 一样，Transformer 主干的优势是在数据和算力足够时才体现出来的；数据少、训练短时，带着图像先验的 U-Net 反而更容易训好。第 7 节的小规模实验正好可以看到这一点。

#### U-Net 和 DiT 的取舍

| | U-Net | DiT |
|---|---|---|
| 图像先验 | 局部性、平移等变、多尺度都写在结构里 | 几乎没有，只有 patch 和位置编码 |
| 小数据、短训练 | 收敛快、效果好 | 收敛慢，需要从数据里学局部规律 |
| 大数据、大算力 | 加大规模的方式不明确，收益难预测 | 效果随 Gflops 稳定提升 |
| 全局信息 | 只在低分辨率层有注意力 | 每一层所有 token 互相可见 |
| 放大方式 | 需要重新设计各级通道、注意力位置 | 只调 $$L$$、$$D$$、patch 大小 |
| 和文本、视频统一 | 需要额外的 cross-attention、时间层 | 都是 token，可以拼进同一个序列 |
| 工程生态 | 卷积专用优化 | 直接复用大语言模型的训练和推理基础设施 |
| 计算量 | 随像素数线性增长 | 注意力随 token 数平方增长，必须在 latent 上并配合 patchify |

#### DiT 适合什么场景、在什么条件下优于 U-Net

综合上面的取舍，DiT 的优势在下面这些条件下最明显：

- **数据量大。** 百万到数十亿张图像（ImageNet 是 128 万张，文生图模型通常是数亿到数十亿对图文）。数据越多，卷积先验带来的好处越小，Transformer 从数据中学习的灵活性越重要。
- **训练算力大。** DiT 的论文结论来自 40 万到 700 万步的训练；FID 随训练计算量持续下降，没有出现饱和，而且同样的总计算量下大模型更好。
- **需要继续放大模型。** 有缩放规律可循：可以先训练小模型，再按趋势估计大模型的效果，决定投入多少算力。
- **输入是长序列或多模态。** 文生图时文本 token 和图像 token 可以放在同一个注意力里（SD3、FLUX）；视频生成时，所有帧的时空 patch 组成一个序列，每个 token 能直接看到其他帧的任意位置，不需要单独设计时间层。

因此 DiT 的典型应用是**大规模文生图**（PixArt-α、SD3、FLUX）和**视频生成**（Sora、Movie Gen、Wan、HunyuanVideo 等，见第 8 节）。视频是 Transformer 主干收益最大的场景之一：数据量极大，token 数多，时间和空间的长距离依赖都很重要，而序列并行等大模型工程手段可以直接用来处理很长的 token 序列。

反过来，**数据少、算力有限、图像小**（例如几万张 $$28 \times 28$$ 的图、训练几千步）时，U-Net 通常更容易训好：卷积先验相当于"免费"给了模型一部分知识。第 7 节在 FashionMNIST 上的小实验就是这种情况，它可以用来验证实现的正确性、观察 adaLN-Zero 的作用，但**不能用来证明 DiT 比 U-Net 更好**；DiT 的优势需要在大数据、长训练的规模上才能体现。

#### 用 Transformer 做生成要解决的三个具体问题

把 Transformer 用在扩散上，需要解决三个具体问题，也就是 DiT 的三个设计：


| 问题 | 原因 | DiT 的做法 |
|---|---|---|
| token 太多 | 256×256 的像素图有 65536 个像素，注意力的代价和 token 数的平方成正比 | 在 LDM 的 $$32\times32\times4$$ latent 上工作，再把 $$p \times p$$ 的小块合成一个 token（patchify），第 3 节 |
| 怎样注入时间和类别 | U-Net 把时间嵌入加到每个 ResBlock 的特征上；Transformer 没有现成的位置 | 比较了四种方式，选 adaLN-Zero：用条件生成每层 LayerNorm 的缩放、平移和残差门控，门控初始化为 0，第 4 节 |
| 输出要是一张图 | Transformer 输出的是 token 序列 | 每个 token 线性映射回 $$p \times p$$ 个位置的预测值，再按位置拼回（unpatchify），第 4.6 节 |

### ViT：把图像当成 token 序列

DiT 的输入部分直接来自 ViT [[3]](#ref-3)。ViT 做图像分类的流程是：

1. 把 $$224 \times 224$$ 的图像切成 $$16 \times 16$$ 的小块，共 $$14 \times 14 = 196$$ 块；
2. 每块拉平成 $$16 \times 16 \times 3 = 768$$ 维的向量，用一个线性层映射到 $$D$$ 维，得到 196 个 token；
3. 加上位置编码（否则 Transformer 不知道每块原来在哪里），再在最前面加一个可学习的 `[CLS]` token；
4. 经过若干层标准的 Transformer encoder，取 `[CLS]` 位置的输出做分类。

DiT 沿用了 1–2 步和位置编码，但不需要 `[CLS]`：扩散要对**每个位置**都输出预测值，所以每个 token 的输出都要用到。下一篇 ViT 与 CLIP 会更详细地介绍 ViT 本身。

<figure class="ddpm-fig">
<svg viewBox="0 0 760 300" role="img" aria-label="DiT 的整体数据流">
  <defs><marker id="da" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker></defs>
  <rect x="14" y="70" width="92" height="92" rx="6" fill="#eef4ff" stroke="#2a6fdb"/>
  <g stroke="#2a6fdb" stroke-width="0.8" opacity="0.7">
    <line x1="37" y1="70" x2="37" y2="162"/><line x1="60" y1="70" x2="60" y2="162"/><line x1="83" y1="70" x2="83" y2="162"/>
    <line x1="14" y1="93" x2="106" y2="93"/><line x1="14" y1="116" x2="106" y2="116"/><line x1="14" y1="139" x2="106" y2="139"/></g>
  <text x="60" y="182" text-anchor="middle">带噪 latent z<tspan baseline-shift="sub" font-size="9">t</tspan></text>
  <text x="60" y="198" text-anchor="middle" class="mono">[B, C, I, I]</text>
  <line x1="108" y1="116" x2="140" y2="116" stroke="#475467" marker-end="url(#da)"/>
  <rect x="142" y="86" width="96" height="60" rx="8" fill="#fff" stroke="#475467"/>
  <text x="190" y="110" text-anchor="middle">Patchify</text><text x="190" y="128" text-anchor="middle" class="sm">+ 位置编码</text>
  <line x1="240" y1="116" x2="272" y2="116" stroke="#475467" marker-end="url(#da)"/>
  <g fill="#fff4ec" stroke="#e0662b"><rect x="274" y="70" width="14" height="92" rx="3"/><rect x="291" y="70" width="14" height="92" rx="3"/><rect x="308" y="70" width="14" height="92" rx="3"/><rect x="325" y="70" width="14" height="92" rx="3"/></g>
  <text x="307" y="182" text-anchor="middle">N 个 token</text><text x="307" y="198" text-anchor="middle" class="mono">[B, N, D]</text>
  <line x1="342" y1="116" x2="372" y2="116" stroke="#475467" marker-end="url(#da)"/>
  <rect x="374" y="60" width="130" height="112" rx="10" fill="#f8f9fa" stroke="#475467"/>
  <text x="439" y="96" text-anchor="middle">DiT Block × L</text>
  <text x="439" y="116" text-anchor="middle" class="sm">self-attention + MLP</text>
  <text x="439" y="134" text-anchor="middle" class="sm">adaLN-Zero 注入条件</text>
  <rect x="374" y="226" width="130" height="50" rx="8" fill="#ecfdf3" stroke="#12b76a"/>
  <text x="439" y="246" text-anchor="middle">条件向量 c</text><text x="439" y="264" text-anchor="middle" class="sm">时间 t 的嵌入 + 类别 y 的嵌入</text>
  <line x1="439" y1="224" x2="439" y2="174" stroke="#12b76a" marker-end="url(#da)"/>
  <line x1="506" y1="116" x2="536" y2="116" stroke="#475467" marker-end="url(#da)"/>
  <rect x="538" y="86" width="96" height="60" rx="8" fill="#fff" stroke="#475467"/>
  <text x="586" y="110" text-anchor="middle">Final Layer</text><text x="586" y="128" text-anchor="middle" class="sm">+ Unpatchify</text>
  <line x1="636" y1="116" x2="664" y2="116" stroke="#475467" marker-end="url(#da)"/>
  <rect x="666" y="70" width="80" height="92" rx="6" fill="#fff" stroke="#475467"/>
  <text x="706" y="112" text-anchor="middle">预测</text><text x="706" y="130" text-anchor="middle" class="sm">ε（和 Σ）</text>
  <text x="706" y="182" text-anchor="middle" class="mono">[B, C_out, I, I]</text>
  <line x1="504" y1="250" x2="586" y2="250" stroke="#12b76a"/><line x1="586" y1="250" x2="586" y2="148" stroke="#12b76a" marker-end="url(#da)"/>
  <text x="380" y="30" text-anchor="middle" class="sm">切块 → token 序列 → 标准 Transformer（条件通过 adaLN-Zero 进入每一层） → 拼回图像</text>
</svg>
<figcaption>DiT 的整体结构（对应论文图 3 左）。I 是 latent 的边长，p 是 patch 边长，N = (I/p)²，D 是 token 维度。</figcaption>
</figure>

### 数值示例：一张 256×256 图像经过 DiT-XL/2

1. **编码到 latent。** SD 的自编码器下采样 8 倍：$$256 \times 256 \times 3 \to 32 \times 32 \times 4$$，即 $$I = 32$$、$$C = 4$$。
2. **Patchify。** patch 边长 $$p = 2$$：每个 $$2 \times 2 \times 4 = 16$$ 个数合成一个 token，共 $$N = (32/2)^2 = 256$$ 个 token。一个线性层把 16 维映射到 $$D = 1152$$ 维。
3. **28 个 DiT block。** 每层对 256 个 token 做 self-attention 和 MLP，同时用条件向量 $$c$$ 调制。整个网络约 6.75 亿参数，一次前向约 118.6 GFLOPs（第 5 节用代码验证这两个数）。
4. **输出。** 最后一层把每个 token 从 1152 维映射到 $$p \times p \times 2C = 2 \times 2 \times 8 = 32$$ 维，拼回 $$32 \times 32 \times 8$$：前 4 个通道是预测的噪声 $$\epsilon$$，后 4 个通道用于方差 $$\Sigma$$。

如果改用 $$p = 4$$，token 数变成 64，计算量降到约 1/4（29.1 GFLOPs），参数量几乎不变：patch 大小只影响输入、输出两个线性层。

### 一个可以运行的最小例子：patchify 和 unpatchify

Patchify 有两种等价写法：显式切块后做线性层，或者用一个 kernel 和 stride 都等于 $$p$$ 的卷积。下面验证两者相同，以及 unpatchify 能把 token 精确拼回原图：

```python
import torch

B, C, I, p, D = 2, 4, 32, 2, 8
z = torch.randn(B, C, I, I)
conv = torch.nn.Conv2d(C, D, kernel_size=p, stride=p)        # DiT 用的写法

# 写法 1：卷积。[B, D, I/p, I/p] -> [B, N, D]
tok_conv = conv(z).flatten(2).transpose(1, 2)

# 写法 2：先切块拉平，再乘同一组权重
h = I // p
# [B, C, I, I] -> [B, N, C·p·p]
patches = z.reshape(B, C, h, p, h, p).permute(0, 2, 4, 1, 3, 5).reshape(B, h * h, C * p * p)
tok_lin = patches @ conv.weight.reshape(D, -1).T + conv.bias
print(torch.allclose(tok_conv, tok_lin, atol=1e-5))         # True：stride=p 的卷积就是逐块的线性层

# unpatchify：每个 token 带 p·p·C 个数，按 (行块, 列块, 块内行, 块内列, 通道) 放回
x = z.reshape(B, C, h, p, h, p).permute(0, 2, 4, 3, 5, 1).reshape(B, h * h, p * p * C)   # 模拟网络输出
x = x.reshape(B, h, h, p, p, C)
img = torch.einsum("nhwpqc->nchpwq", x).reshape(B, C, h * p, h * p)
print(torch.equal(img, z))                                   # True
```

注意两段代码里 token 内部的数值顺序不同：patchify 时卷积权重按 (通道, 块内行, 块内列) 排列，网络输出时按 (块内行, 块内列, 通道) 排列。只要 unpatchify 和最后一层的输出约定一致即可，网络会自己学会这个顺序。

## 0. 全文路线

| 节 | 内容 | 关键结论 |
|---|---|---|
| 1 | 符号约定 | |
| 2 | 整体结构和前向过程 | 五步：patchify、加位置编码、L 个 block、final layer、unpatchify |
| 3 | Patchify 与位置编码 | token 数 $$N = (I/p)^2$$；固定的二维 sin-cos 编码 |
| 4 | 条件注入 | 四种方式，adaLN-Zero 最好；零初始化让每个 block 起初是恒等映射 |
| 5 | 配置、计算量和缩放 | 代码算出的参数量和 Gflops 与论文一致；FID 跟着 Gflops 走 |
| 6 | 训练和采样 | 论文的 DDPM 设定；换成 Flow Matching 只改输出通道 |
| 7 | FashionMNIST 实验 | DiT 和 U-Net 对比；四种条件注入方式对比 |
| 8 | 后续发展 | SiT、PixArt-α、MM-DiT（SD3）、FLUX、视频 DiT |
| 9–10 | 自检、要点回顾 | |

## 1. 符号约定

| 公式符号 | 含义 | 代码变量 | shape |
|---|---|---|---|
| $$z_t$$ | 带噪的 latent（像素空间实验中就是图像） | `z_t` | `[B, C, I, I]` |
| $$C$$ | 输入通道数（SD latent 为 4，灰度图为 1） | `in_channels` | |
| $$I$$ | latent 的边长（256 图像 → 32） | `input_size` | |
| $$p$$ | patch 边长 | `patch_size` | |
| $$N = (I/p)^2$$ | token 数 | `num_patches` | |
| $$D$$ | token 维度（hidden size） | `hidden_size` | |
| $$L$$ | block 数（depth） | `depth` | |
| $$t$$ | 时间步（DDPM 为 $$0 \dots 999$$；Flow Matching 为 $$[0,1]$$，输入前乘 1000） | `t` | `[B]` |
| $$y$$ | 类别标签，$$y = K$$ 表示空类别（CFG 用） | `y` | `[B]` |
| $$c$$ | 条件向量：时间嵌入 + 类别嵌入 | `c` | `[B, D]` |
| $$\gamma, \beta$$ | adaLN 的缩放和平移，由 $$c$$ 回归得到 | `scale`, `shift` | `[B, D]` |
| $$\alpha$$ | adaLN-Zero 的残差门控，初始化为 0 | `gate` | `[B, D]` |
| $$C_{\text{out}}$$ | 输出通道：学习方差时 $$2C$$，否则 $$C$$ | `out_channels` | |

论文里 adaLN 的公式写作 $$\gamma \cdot \mathrm{LN}(x) + \beta$$，官方代码写作 $$\mathrm{LN}(x)\cdot(1 + \gamma) + \beta$$。两者只差参数化方式：代码的写法下 $$\gamma = \beta = 0$$ 正好是恒等变换，和零初始化配合。本文的公式按代码的写法。

## 2. 整体结构和前向过程

把上面的流程图写成公式：

$$
\begin{aligned}
h^{(0)} &= \mathrm{Patchify}(z_t) + E_{\text{pos}} &&\in \mathbb{R}^{N \times D} \\
c &= \mathrm{Emb}_t(t) + \mathrm{Emb}_y(y) &&\in \mathbb{R}^{D} \\
h^{(\ell)} &= \mathrm{Block}_\ell\big(h^{(\ell-1)},\ c\big), \quad \ell = 1, \dots, L \\
\text{out} &= \mathrm{Unpatchify}\big(\mathrm{FinalLayer}(h^{(L)}, c)\big) &&\in \mathbb{R}^{C_{\text{out}} \times I \times I}
\end{aligned}
\tag{1}
$$

对应的代码（完整模型见第 4.7 节）：

```python
def forward(self, z_t, t, y):
    x = self.x_embedder(z_t).flatten(2).transpose(1, 2) + self.pos_embed   # [B, N, D]，式 (1) 第一行
    c = self.t_embedder(t) + self.y_embedder(y, self.training)             # [B, D]，第二行
    for blk in self.blocks:                                                 # 第三行，L 次
        x = blk(x, c)
    x = self.final_layer(x, c)                                              # [B, N, p*p*C_out]
    return self.unpatchify(x)                                               # [B, C_out, I, I]，第四行
```

和 U-Net 相比，这里没有下采样、上采样和 skip connection：所有 block 都在同一个分辨率（$$N$$ 个 token）上工作，每个 token 从第一层起就能通过注意力看到全图。

## 3. Patchify 与位置编码

### 3.1 Patchify

如背景中的例子，patchify 就是一个 kernel 和 stride 都为 $$p$$ 的卷积，输出 $$[B, D, I/p, I/p]$$，拉平成 $$N$$ 个 token。$$p$$ 是 DiT 最重要的超参数之一 [[1]](#ref-1)：

- **$$p$$ 减半，token 数变成 4 倍**，注意力的计算量约变成 16 倍（$$N^2$$），MLP 约变成 4 倍，总 Gflops 约变成 4 倍；
- **参数量几乎不变**，只有 patchify 和输出层的权重和 $$p^2$$ 成正比；
- 论文的结论是：**参数量固定时，更小的 patch（更多计算）FID 明显更好**。DiT-XL/2（$$p=2$$）是最终模型。

### 3.2 二维 sin-cos 位置编码

Transformer 的注意力对 token 顺序不敏感，必须加位置信息。DiT 用固定（不训练）的二维正弦编码：$$D$$ 维向量的前一半编码 token 所在的列，后一半编码所在的行，每一半都是和 Transformer 原论文相同的一维 sin-cos 编码：

$$
\mathrm{PE}_{1D}(m)_{2i} = \sin\!\big(m \cdot \omega_i\big),\quad
\mathrm{PE}_{1D}(m)_{2i+1} \to \cos\!\big(m \cdot \omega_i\big),\qquad \omega_i = 10000^{-i/(D/4)}
$$

（实现上 sin 和 cos 分别拼成前后两段，而不是交错，效果相同。）

```python
import numpy as np

def get_2d_sincos_pos_embed(D, grid_size):
    """固定的二维 sin-cos 位置编码：[grid*grid, D]。前 D/2 维编码列号，后 D/2 维编码行号。"""
    def emb_1d(d, pos):                                            # pos: [M] -> [M, d]
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))   # ω_i，[d/2]
        out = np.einsum("m,d->md", pos.reshape(-1), omega)                         # m·ω_i，[M, d/2]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)                  # [M, d]
    ar = np.arange(grid_size, dtype=np.float64)
    g0, g1 = np.meshgrid(ar, ar, indexing="xy")             # g0 = 列号，g1 = 行号
    return np.concatenate([emb_1d(D // 2, g0), emb_1d(D // 2, g1)], axis=1)       # [N, D]
```

在模型里用 `register_buffer` 存成 `[1, N, D]`，前向时直接加到 token 上，不参与训练。

> **易错点**
>
> 1. **编码和 token 的顺序要一致。** 卷积输出 `flatten(2)` 后 token 按行优先排列（先第 0 行的所有列），`meshgrid(..., indexing="xy")` 得到的网格也是行优先，第 $$k$$ 个 token 位于第 $$\lfloor k / (I/p) \rfloor$$ 行、第 $$k \bmod (I/p)$$ 列。换成 `indexing="ij"` 会把行列对调。
> 2. **固定编码不能换分辨率直接用可学习编码的权重。** DiT 用固定编码的好处之一是换分辨率时只需重新计算这张表；后来的 SD3、FLUX 改用 RoPE 或可插值的编码来支持多种分辨率和长宽比。

## 4. 条件注入

### 4.1 时间和类别的嵌入

时间 $$t$$ 先做 256 维的正弦编码（和 DDPM 篇第 5.2 节相同），再经过一个两层 MLP 映射到 $$D$$ 维；类别 $$y$$ 查一张 $$(K + 1) \times D$$ 的嵌入表，多出来的一行是空类别。训练时以 10% 的概率把标签换成空类别，用于 classifier-free guidance（和 DDPM 篇第 9 节相同）。

```python
class TimestepEmbedder(nn.Module):
    """t [B] -> 正弦编码 [B, 256] -> MLP -> [B, D]。"""
    def __init__(self, D, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, D), nn.SiLU(), nn.Linear(D, D))

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        # [half]
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]                     # [B, half]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


class LabelEmbedder(nn.Module):
    """y [B] -> [B, D]。最后一行（id = K）是空类别。"""
    def __init__(self, num_classes, D, dropout_prob=0.1):
        super().__init__()
        self.table = nn.Embedding(num_classes + 1, D)
        self.num_classes, self.dropout_prob = num_classes, dropout_prob

    def forward(self, y, train):
        if train and self.dropout_prob > 0:                       # 训练时随机丢标签，供 CFG 使用
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        return self.table(y)
```

两者相加得到 $$c = \mathrm{Emb}_t(t) + \mathrm{Emb}_y(y)$$。文本条件（第 8 节）不能这样压成一个向量，需要别的办法。

### 4.2 四种注入方式

怎样把 $$c$$ 送进 Transformer？论文比较了四种 block 设计 [[1]](#ref-1)：

| 方式 | 做法 | 额外计算 |
|---|---|---|
| **In-context** | 把 $$\mathrm{Emb}_t$$、$$\mathrm{Emb}_y$$ 当作两个额外的 token 拼在序列前面（类似 ViT 的 `[CLS]`），block 本身不变；最后一层前去掉这两个 token | 几乎没有 |
| **Cross-attention** | 两个嵌入组成长度为 2 的序列，在 self-attention 后加一层 cross-attention 去读它（类似 SD 的 U-Net 读文本） | 约 +15% Gflops |
| **adaLN** | 把 LayerNorm 的缩放和平移换成由 $$c$$ 回归出来的 $$\gamma, \beta$$ | 很少 |
| **adaLN-Zero** | 在 adaLN 的基础上，每个残差分支再乘一个由 $$c$$ 回归的门控 $$\alpha$$，并把回归层初始化为 0 | 很少 |

论文在 DiT-XL/2 上训练 40 万步的对比中，**adaLN-Zero 的 FID 明显最低**，in-context 最差，cross-attention 和 adaLN 居中；adaLN-Zero 的计算量又和最便宜的方式相当。所以最终模型全部用 adaLN-Zero。第 7.2 节在小规模上重复了这个对比。

<figure class="ddpm-fig">
<svg viewBox="0 0 760 360" role="img" aria-label="adaLN-Zero DiT block 的结构">
  <defs><marker id="db" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker>
  <marker id="dg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#12b76a"/></marker></defs>
  <text x="40" y="186" text-anchor="middle">输入 token</text><text x="40" y="202" text-anchor="middle" class="mono">[B,N,D]</text>
  <line x1="78" y1="190" x2="98" y2="190" stroke="#475467" marker-end="url(#db)"/>
  <rect x="100" y="170" width="74" height="40" rx="6" fill="#fff" stroke="#475467"/><text x="137" y="195" text-anchor="middle">LayerNorm</text>
  <line x1="174" y1="190" x2="192" y2="190" stroke="#475467" marker-end="url(#db)"/>
  <rect x="194" y="170" width="84" height="40" rx="6" fill="#ecfdf3" stroke="#12b76a"/><text x="236" y="188" text-anchor="middle">缩放、平移</text><text x="236" y="203" text-anchor="middle" class="sm">γ₁, β₁</text>
  <line x1="278" y1="190" x2="296" y2="190" stroke="#475467" marker-end="url(#db)"/>
  <rect x="298" y="170" width="84" height="40" rx="6" fill="#fff" stroke="#475467"/><text x="340" y="188" text-anchor="middle">多头</text><text x="340" y="203" text-anchor="middle" class="sm">self-attention</text>
  <line x1="382" y1="190" x2="398" y2="190" stroke="#475467" marker-end="url(#db)"/>
  <rect x="400" y="170" width="56" height="40" rx="6" fill="#ecfdf3" stroke="#12b76a"/><text x="428" y="188" text-anchor="middle">× α₁</text><text x="428" y="203" text-anchor="middle" class="sm">门控</text>
  <line x1="456" y1="190" x2="472" y2="190" stroke="#475467" marker-end="url(#db)"/>
  <circle cx="484" cy="190" r="11" fill="#fff" stroke="#475467"/><text x="484" y="195" text-anchor="middle">+</text>
  <path d="M88,190 L88,140 L484,140 L484,177" fill="none" stroke="#475467" stroke-dasharray="4 3" marker-end="url(#db)"/>
  <text x="290" y="132" text-anchor="middle" class="sm">残差</text>
  <line x1="495" y1="190" x2="520" y2="190" stroke="#475467" marker-end="url(#db)"/>
  <text x="640" y="160" text-anchor="middle">同样的结构再来一次：</text>
  <text x="640" y="182" text-anchor="middle" class="sm">LayerNorm → γ₂, β₂ → MLP</text>
  <text x="640" y="200" text-anchor="middle" class="sm">→ × α₂ → 加残差</text>
  <rect x="524" y="140" width="232" height="72" rx="8" fill="none" stroke="#98a2b3" stroke-dasharray="4 3"/>
  <rect x="160" y="282" width="310" height="50" rx="8" fill="#ecfdf3" stroke="#12b76a"/>
  <text x="315" y="302" text-anchor="middle">条件 c → SiLU → Linear(D, 6D)</text>
  <text x="315" y="320" text-anchor="middle" class="sm">切成 β₁, γ₁, α₁, β₂, γ₂, α₂（权重初始化为 0）</text>
  <line x1="236" y1="280" x2="236" y2="212" stroke="#12b76a" marker-end="url(#dg)"/>
  <line x1="428" y1="280" x2="428" y2="212" stroke="#12b76a" marker-end="url(#dg)"/>
  <path d="M470,307 L640,307 L640,214" fill="none" stroke="#12b76a" marker-end="url(#dg)"/>
  <text x="380" y="40" text-anchor="middle" class="sm">一个 adaLN-Zero DiT block（对应论文图 3 右）。绿色部分由条件 c 计算，对所有 token 相同。</text>
  <text x="380" y="62" text-anchor="middle" class="sm">初始化时 α₁ = α₂ = 0，两个残差分支都被关掉，整个 block 等于恒等映射。</text>
</svg>
<figcaption>adaLN-Zero block。LayerNorm 本身不带可学习参数，缩放和平移全部来自条件 c。</figcaption>
</figure>

### 4.3 adaLN：让条件控制每一层的归一化

普通 LayerNorm 对每个 token 的 $$D$$ 维向量做标准化，再乘一个可学习的缩放、加一个可学习的平移，这两个参数对所有输入都一样。**adaLN**（adaptive LayerNorm）把它们换成由条件回归出来的值：

$$
\mathrm{adaLN}(h, c) = \mathrm{LN}(h)\odot\big(1 + \gamma(c)\big) + \beta(c),\qquad \big[\beta(c), \gamma(c)\big] = W\,\mathrm{SiLU}(c) + b
\tag{2}
$$

其中 $$\mathrm{LN}$$ 不带可学习参数（`elementwise_affine=False`），$$\gamma(c), \beta(c) \in \mathbb{R}^D$$ 对同一张图的所有 token 相同。这个思路来自 GAN 和风格迁移里的 AdaIN、FiLM，也和 U-Net 里"把时间嵌入投影后加到特征上"（DDPM 篇第 5.3 节）是同一类做法：**条件通过逐通道的缩放和平移改变整层的特征分布**。

它比 cross-attention 更适合 $$t$$ 和 $$y$$：条件只是一个向量，不需要逐 token 地"查询"它；而它又比 in-context 更直接：条件作用在每一层，而不是只靠注意力从两个额外 token 里慢慢读出来。

### 4.4 adaLN-Zero：每个 block 从恒等映射开始

adaLN-Zero 在每个残差分支的输出上再乘一个门控 $$\alpha(c)$$，一个 block 的完整计算是：

$$
\begin{aligned}
h &\leftarrow h + \alpha_1(c) \odot \mathrm{Attn}\big(\mathrm{LN}(h)\odot(1 + \gamma_1(c)) + \beta_1(c)\big) \\
h &\leftarrow h + \alpha_2(c) \odot \mathrm{MLP}\big(\mathrm{LN}(h)\odot(1 + \gamma_2(c)) + \beta_2(c)\big)
\end{aligned}
\tag{3}
$$

六个向量 $$\beta_1, \gamma_1, \alpha_1, \beta_2, \gamma_2, \alpha_2$$ 由同一个线性层一次算出（$$D \to 6D$$），**这个线性层的权重和偏置全部初始化为 0**。于是在训练开始时：

$$
\alpha_1 = \alpha_2 = 0 \ \Rightarrow\ h \leftarrow h + 0,\qquad \text{每个 block 都是恒等映射}
$$

整个网络在初始时等价于"patchify → 直接 final layer"，而 final layer 的输出线性层也初始化为 0，所以**初始输出恰好为 0**。

为什么这样好？

- **深层网络更容易训练。** 残差网络在初始时如果每个分支都输出 0，信号和梯度都能无损地穿过所有层，训练初期不会因为几十层随机变换的叠加而不稳定。这和 ResNet 训练中把每个残差块最后一个 BN 的缩放初始化为 0（zero-γ）、以及 [ControlNet](/deep-learning/controlnet/) 的零卷积是同一个思路。
- **门控本身也受条件控制。** $$\alpha(c)$$ 随时间和类别变化，网络可以学到"在某些时间步，某一层的注意力分支更重要"。
- **零初始化不会让训练卡住。** 门控线性层的权重为 0 时，它对自身权重的梯度是 $$\partial \mathcal{L} / \partial \alpha \cdot \mathrm{SiLU}(c)^\top$$，只要传到 $$\alpha$$ 的梯度不为 0，它就不为 0。第一步时最后的输出层也是 0，梯度还传不进各个 block；第一步只更新输出层，从第二步起门控就开始离开 0，各分支逐渐起作用（第 9 节自检第 5 项验证了这一点）。第 7.3 节测量了训练后门控的大小。

### 4.5 代码：DiT block

```python
def modulate(x, shift, scale):
    """adaLN：x ⊙ (1 + scale) + shift。x: [B, N, D]；shift、scale: [B, D]，对所有 token 相同。"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """adaLN-Zero block，式 (3)。x: [B, N, D]，c: [B, D]。"""
    def __init__(self, D, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)   # 不带 γ、β，由 adaLN 提供
        self.attn = Attention(D, heads)                                      # 标准多头 self-attention
        self.norm2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(D, mlp_ratio)                                          # D -> 4D -> D
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))  # c -> 6 个 [B, D]

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=1)
        # 式 (3) 第一行、第二行
        x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x
```

`Attention` 和 `Mlp` 就是 Transformer 篇里的标准模块（这里用 `F.scaled_dot_product_attention` 实现注意力，MLP 的激活用 tanh 近似的 GELU，和官方代码一致）：

```python
class Attention(nn.Module):
    """多头 self-attention。x: [B, N, D] -> [B, N, D]。"""
    def __init__(self, D, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)

    def forward(self, x):
        B, N, D = x.shape
        # 各 [B, h, N, D/h]
        q, k, v = self.qkv(x).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)                   # softmax(QKᵀ/√d)V
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class Mlp(nn.Module):
    def __init__(self, D, ratio=4.0):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(D, int(D * ratio)), nn.Linear(int(D * ratio), D)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))
```

### 4.6 Final layer 和 unpatchify

最后一层同样用 adaLN（只有平移和缩放，没有门控），再用一个线性层把每个 token 映射到 $$p \cdot p \cdot C_{\text{out}}$$ 维：

```python
class FinalLayer(nn.Module):
    """[B, N, D] -> [B, N, p*p*C_out]。"""
    def __init__(self, D, p, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(D, p * p * out_ch)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


def unpatchify(self, x):
    """[B, N, p*p*C_out] -> [B, C_out, I, I]。"""
    B, p, c, h = x.shape[0], self.p, self.out_channels, self.grid       # h = I / p
    # 每个 token 的 p*p*C 个数按 (块内行, 块内列, 通道) 排列
    x = x.reshape(B, h, h, p, p, c)
    x = torch.einsum("nhwpqc->nchpwq", x)                                 # [B, C, h, p, w, p]
    return x.reshape(B, c, h * p, h * p)
```

### 4.7 完整模型和初始化

```python
class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28,
                 num_heads=16, num_classes=1000, class_dropout_prob=0.1, learn_sigma=True):
        super().__init__()
        self.p, self.C = patch_size, in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels      # ε（和 Σ）
        D = hidden_size
        # patchify：kernel = stride = p 的卷积
        self.x_embedder = nn.Conv2d(in_channels, D, kernel_size=patch_size, stride=patch_size)
        self.t_embedder = TimestepEmbedder(D)
        self.y_embedder = LabelEmbedder(num_classes, D, class_dropout_prob)
        self.grid = input_size // patch_size                                      # I / p
        self.register_buffer("pos_embed", torch.from_numpy(
            get_2d_sincos_pos_embed(D, self.grid)).float().unsqueeze(0))         # [1, N, D]，固定
        self.blocks = nn.ModuleList([DiTBlock(D, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(D, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():                               # 所有线性层：Xavier，偏置 0
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        w = self.x_embedder.weight.data                        # patchify 当作线性层初始化
        nn.init.xavier_uniform_(w.view(w.shape[0], -1)); nn.init.zeros_(self.x_embedder.bias)
        nn.init.normal_(self.y_embedder.table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for blk in self.blocks:                                    # adaLN-Zero 的 "Zero"
            nn.init.zeros_(blk.adaLN_modulation[-1].weight)
            nn.init.zeros_(blk.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)                            # 初始输出为 0
        nn.init.zeros_(self.final_layer.linear.bias)

    # forward 见第 2 节，unpatchify 见第 4.6 节
```

实验代码里的版本多一个 `cond_mode` 参数，用来切换第 4.2 节的四种 block；默认的 `adaln_zero` 和上面完全相同。

> **易错点**
>
> 1. **LayerNorm 不能带可学习的仿射参数。** adaLN 已经提供了缩放和平移；如果 `elementwise_affine=True`，会多出一组和条件无关的参数，和官方实现不一致，加载官方权重时也会报错。
> 2. **`1 + scale`，不是 `scale`。** 零初始化时 `scale = 0`，必须是 `1 + scale` 才是恒等变换；写成 `x * scale + shift` 会让初始输出全为 0，网络从一个"把所有特征乘 0"的状态开始，训练很慢。
> 3. **三处零初始化缺一不可。** 每个 block 的 adaLN 线性层、final layer 的 adaLN 线性层、final layer 的输出线性层。`self.apply(...)` 式的通用初始化要放在最前面，否则会把零初始化覆盖掉。
> 4. **chunk 的顺序。** 官方代码的顺序是 `shift, scale, gate`（先平移后缩放），和论文公式里 $$\gamma, \beta$$ 的书写顺序相反。自己实现时顺序随意，但加载官方权重时必须一致。
> 5. **条件对所有 token 相同。** `shift`、`scale`、`gate` 的 shape 是 `[B, D]`，用之前要 `unsqueeze(1)` 变成 `[B, 1, D]` 再广播到 $$N$$ 个 token。

### 4.8 总结：DiT 一次前向的完整流程

| 步骤 | 计算 | shape（DiT-XL/2，256 图像） |
|---|---|---|
| 1 | 输入带噪 latent $$z_t$$ | `[B, 4, 32, 32]` |
| 2 | Patchify：卷积 $$k = s = p$$，拉平 | `[B, 256, 1152]` |
| 3 | 加固定的二维 sin-cos 位置编码 | `[B, 256, 1152]` |
| 4 | $$c = \mathrm{Emb}_t(t) + \mathrm{Emb}_y(y)$$ | `[B, 1152]` |
| 5 | 28 个 adaLN-Zero block，式 (3) | `[B, 256, 1152]` |
| 6 | Final layer：adaLN + Linear | `[B, 256, 32]` |
| 7 | Unpatchify | `[B, 8, 32, 32]`：$$\epsilon$$ 和方差各 4 个通道 |

## 5. 配置、计算量和缩放规律

### 5.1 四种规模

论文沿用 ViT 的命名，定义了四种规模，每种再配三种 patch 大小（DiT-XL/2 表示 XL 规模、$$p = 2$$）[[1]](#ref-1)：

| 模型 | 层数 $$L$$ | 维度 $$D$$ | 头数 | 参数量 | GFLOPs（$$p$$ = 8 / 4 / 2） |
|---|---|---|---|---|---|
| DiT-S | 12 | 384 | 6 | 33M | 0.36 / 1.41 / 6.06 |
| DiT-B | 12 | 768 | 12 | 130M | 1.42 / 5.56 / 23.01 |
| DiT-L | 24 | 1024 | 16 | 458M | 5.01 / 19.70 / 80.70 |
| DiT-XL | 28 | 1152 | 16 | 675M | 7.39 / 29.04 / 118.62 |

表中的参数量和 GFLOPs 是用本文的实现**实际计算**的：构造每个配置，输入一张 $$32 \times 32 \times 4$$ 的 latent，用 PyTorch 的 `FlopCounterMode` 统计一次前向的乘加次数（论文的 "Gflops" 指乘加次数）。12 个配置全部和论文给出的数值一致（例如 DiT-XL/2：674.8M 参数，118.62 GFLOPs；论文为 675M、118.6），说明实现的结构和尺寸是对的。

```python
from torch.utils.flop_counter import FlopCounterMode

m = DiT(input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16).eval()
print(sum(p.numel() for p in m.parameters()) / 1e6)              # 674.8（M）
with FlopCounterMode(display=False) as fc, torch.no_grad():
    m(torch.randn(1, 4, 32, 32), torch.tensor([500]), torch.tensor([1]))
print(fc.get_total_flops() / 2 / 1e9)                            # 118.6（G 乘加）
```

### 5.2 缩放规律

论文在 ImageNet 256×256 上训练了全部 12 个模型，得到三个结论 [[1]](#ref-1)：

1. **加大模型、减小 patch 都让 FID 稳定下降**，在整个训练过程中都是如此，没有出现饱和。
2. **决定效果的是计算量，而不是参数量。** 12 个模型的 FID 和 Gflops 强相关；DiT-XL/8 参数最多，但 Gflops 只有 7.4，效果反而不如参数量小得多的 DiT-B/2。
3. **小模型训练更久也追不上大模型。** 按训练总计算量（Gflops × 步数 × batch）画图，同样的总计算量下大模型更好。

最终的 DiT-XL/2 训练 700 万步，在 ImageNet 256×256 类别条件生成上 FID 达到 2.27（CFG 尺度 1.5），超过了当时所有基于 U-Net 的扩散模型；并且它每次前向只需 118.6 GFLOPs，比像素空间的 ADM（约 1120 GFLOPs）少一个数量级，和 LDM-4（103.6 GFLOPs）相当。

## 6. 训练和采样

### 6.1 论文的设定

DiT 本身只是一个网络结构，训练方式完全沿用已有的做法 [[1]](#ref-1)：

- **latent**：用 Stable Diffusion 的自编码器，256 图像编码成 $$32 \times 32 \times 4$$，乘以缩放因子 0.18215（见 SD 篇第 3.2 节）。
- **扩散过程**：DDPM，$$T = 1000$$，线性 $$\beta$$ 调度（$$10^{-4}$$ 到 $$0.02$$）；网络预测噪声 $$\epsilon$$，同时按 Improved DDPM 的做法**学习方差**：输出的后 $$C$$ 个通道 $$v$$ 在两个可选方差之间插值，$$\Sigma_\theta = \exp\big(v \log\beta_t + (1-v)\log\tilde\beta_t\big)$$，用变分下界的那一项训练。所以输出通道是 $$2C = 8$$。
- **优化**：AdamW，学习率 $$10^{-4}$$ 恒定，无 weight decay，无 warmup，batch 256，EMA 衰减 0.9999；数据增强只有水平翻转。论文特别指出：不需要学习率 warmup 和额外的正则化，训练就很稳定，这在很大程度上归功于 adaLN-Zero 的初始化。
- **采样**：250 步 DDPM，CFG。

### 6.2 换成 Flow Matching

训练目标和网络是解耦的：[Flow Matching](/deep-learning/flow-matching/) 篇第 6 节的训练和采样代码，只要把 Tiny U-Net 换成 DiT 就能直接用。需要改的只有两处：

1. **输出通道**：只预测速度，不学方差，`learn_sigma=False`，$$C_{\text{out}} = C$$；
2. **时间输入**：$$t \in [0, 1]$$，乘以 1000 后再进正弦编码（和 Flow Matching 篇对 U-Net 的处理相同）。

```python
def fm_loss(model, z0, y):
    """Flow Matching 损失，网络是 DiT。z0: [B,C,I,I]，y: [B]（丢标签在 LabelEmbedder 里做）。"""
    B = z0.shape[0]
    eps = torch.randn_like(z0)                                     # ε ~ N(0, I)
    t = torch.rand(B)                                               # t ~ U(0, 1)
    z_t = (1 - t.view(B, 1, 1, 1)) * z0 + t.view(B, 1, 1, 1) * eps  # z_t = (1−t) z0 + t ε
    return F.mse_loss(model(z_t, t * 1000.0, y), eps - z0)         # 目标速度 ε − z0
```

这正是 SiT [[4]](#ref-4) 做的事：网络和 DiT 完全相同，只把训练目标换成插值/速度预测，ImageNet 256 上的 FID 从 2.27 降到 2.06。

### 6.3 CFG 的一个实现细节

官方代码的 `forward_with_cfg` 只对输出的**前 3 个通道**做 guidance，代码注释说明这是为了复现论文数值，标准做法是对全部 $$C$$ 个噪声通道做 guidance（方差通道不做）。自己实现或者用于其他通道数时，应对全部噪声通道做 guidance。

> **易错点**
>
> 1. **学习方差时，前 $$C$$ 个通道才是 $$\epsilon$$。** 用 MSE 训练时只对前 $$C$$ 个通道算损失，后 $$C$$ 个通道用变分下界训练；采样时也只把前 $$C$$ 个通道当作噪声预测。
> 2. **latent 的缩放因子。** 输入 DiT 前要乘 0.18215，解码前要除回去，否则 latent 的方差和标准高斯不匹配（LDM 篇已详细讨论）。
> 3. **训练和推理的 dropout 开关。** 标签丢弃只在 `model.train()` 时生效；采样前要 `model.eval()`，并且显式传空类别 id 来得到无条件分支，不能依赖随机丢弃。

## 7. 实验：FashionMNIST

用第 6.2 节的 Flow Matching 损失，在 FashionMNIST（$$28 \times 28 \times 1$$，像素空间）上训练一个小的类别条件 DiT：$$p = 4$$（49 个 token）、$$D = 192$$、$$L = 6$$、3 个头，约 4.2M 参数。优化器 AdamW，学习率 $$3 \times 10^{-4}$$（100 步 warmup 后恒定），batch 128，EMA 0.999，训练 2500 步，4 核 CPU 上约 23 分钟。另外训练了一个 $$p = 2$$ 的版本（196 个 token，$$D = 128$$、$$L = 6$$、4 个头，约 1.9M 参数，2500 步约 60 分钟），并把 $$p = 4$$ 的模型从 2500 步继续训练到 7500 步（再用约 41 分钟，优化器状态重新初始化）。为了做相同步数的对比，U-Net 也用完全相同的方式从 2500 步继续训练到 7500 步（约 39 分钟）。

对照组是 [Flow Matching](/deep-learning/flow-matching/) 篇第 7 节训练的 Tiny U-Net：损失、batch、步数、采样器都相同。U-Net 和 $$p = 4$$ 的 DiT 每张图一次前向的计算量几乎相同（0.131 和 0.137 G 乘加），每步训练耗时也相近（约 0.5 秒），所以 2500 步时是一个**相同计算量**下的对比；$$p = 2$$ 的 DiT 参数更少，但 token 多 4 倍，计算量约为 0.29 G 乘加。

为了定量比较，另外训练了一个小的 FashionMNIST 分类器（测试准确率 89.3%），对每个模型各生成 1000 张图（每类 100 张，50 步 Euler），计算：

- **类别准确率**：分类器认为生成的图属于指定类别的比例；
- **特征 Fréchet 距离（FD）**：在分类器倒数第二层的 128 维特征上，生成图和 1 万张测试图的高斯分布距离（和 FID 的计算方式相同，只是特征提取器换成了这个小分类器）。越小越好。

### 7.1 DiT 和 U-Net

![DiT 和 U-Net 的生成结果](/deep-learning/assets/dit/dit_vs_unet.png)

每行一个类别，三个模型用同样的初始噪声，CFG $$w = 3$$，50 步 Euler。定量结果（1000 张图）：

| 模型 | 参数量 | 每张图前向（G 乘加） | 训练步数 | 类别准确率 $$w=1$$ / $$w=3$$ | 特征 FD $$w=1$$ / $$w=3$$ |
|---|---|---|---|---|---|
| U-Net | 0.46M | 0.131 | 2500 | 48.7% / 87.8% | 107.5 / 33.5 |
| DiT $$p=4$$ | 4.17M | 0.137 | 2500 | 23.0% / 52.0% | 231.8 / 116.5 |
| DiT $$p=2$$ | 1.87M | 0.291 | 2500 | 21.6% / 45.9% | 268.4 / 147.1 |
| U-Net | 0.46M | 0.131 | 7500 | 78.4% / 97.4% | 26.1 / 26.9 |
| DiT $$p=4$$ | 4.17M | 0.137 | 7500 | 83.9% / 99.3% | 9.7 / 25.3 |

**训练 2500 步时，DiT 明显不如 U-Net。** 图像轮廓和类别大体正确，但表面布满斑点状的噪声，$$p=4$$ 时还能看到 $$4 \times 4$$ 的块状边界。这不是实现错误：第 5 节的参数量和计算量与论文完全一致，训练损失也正常下降。按时间步拆开测量测试集上的速度误差，可以看到 DiT 落后的位置：

| $$t$$ | 0.02 | 0.05 | 0.1 | 0.3 | 0.5 | 0.8 | 0.95 |
|---|---|---|---|---|---|---|---|
| U-Net | 0.840 | 0.570 | 0.401 | 0.238 | 0.217 | 0.252 | 0.256 |
| DiT $$p=4$$，2500 步 | 0.810 | 0.656 | 0.517 | 0.317 | 0.275 | 0.278 | 0.291 |
| DiT $$p=2$$，2500 步 | 0.657 | 0.548 | 0.451 | 0.319 | 0.286 | 0.270 | 0.277 |
| U-Net，7500 步 | **0.469** | **0.363** | **0.285** | **0.190** | 0.178 | 0.217 | 0.234 |
| DiT $$p=4$$，7500 步 | 0.515 | 0.399 | 0.311 | 0.198 | **0.170** | **0.185** | **0.217** |

2500 步时，DiT 的误差主要大在 $$t = 0.05 \sim 0.5$$ 这一段，也就是图像的局部结构和纹理成形的阶段。U-Net 的卷积天生只看邻近像素、在所有位置共享权重，"相邻像素应该相似"这类局部规律不需要学；DiT 每个 token 从一开始就能看到全图，但这些局部规律要从数据里学出来，在数据和训练量都很小的时候就吃亏了。这就是常说的**归纳偏置**：卷积把关于图像的先验写进了结构，Transformer 没有，它要靠更多的数据和计算来弥补。

**继续训练后，DiT 也能生成清晰的图。** 同一个 $$p = 4$$ 的 DiT 训练到 7500 步后，所有时间步的误差都明显下降，生成的图清晰干净，$$w = 3$$ 时特征 FD 从 116.5 降到 25.3，类别准确率 99.3%：

![DiT 训练 2500 步和 7500 步](/deep-learning/assets/dit/dit_longer.png)

**相同步数的对比：两者都训练到 7500 步。** 把 U-Net 也用同样的方式（同样的优化器、学习率、batch、EMA，优化器状态同样重新初始化）从 2500 步继续训练到 7500 步，两个模型每张图的计算量几乎相同（0.131 和 0.137 G 乘加），总训练量也就相同。结果：

![U-Net 和 DiT 都训练 7500 步](/deep-learning/assets/dit/dit_vs_unet_7500.png)

![第 2500–7500 步的训练损失](/deep-learning/assets/dit/dit_curves_7500.png)

- **生成指标：DiT 略好。** $$w = 3$$ 时 DiT 的类别准确率 99.3%、FD 25.3，U-Net 为 97.4%、26.9；$$w = 1$$（不用 CFG）时差距更明显，DiT 的 FD 为 9.7，U-Net 为 26.1，准确率 83.9% 对 78.4%。
- **训练损失：几乎相同。** 最后 500 步的平均损失 DiT 为 0.234，U-Net 为 0.237。从损失曲线看，U-Net 在第 2500–4000 步还领先，之后两条曲线基本重合。
- **按时间步看：各有所长。** U-Net 在低噪声端（$$t \le 0.3$$，局部细节）误差仍然更小，DiT 在高噪声端（$$t \ge 0.5$$，整体结构）误差更小。这和归纳偏置的解释一致：卷积擅长局部规律，注意力让每个 token 直接看到全图，有利于决定整体布局。
- **从样本看**，两者质量接近，肉眼很难分出好坏；DiT 生成的连衣裙和衬衫花纹更多样，凉鞋（第 6 行）两者都不太好。

**这个对比的结论要打折扣。** 它是一个"相同训练步数、相同每步计算量"的对比，比 2500 步时公平得多，但仍有几个限制：

1. **参数量不同。** DiT 有 4.17M 参数，是 U-Net（0.46M）的 9 倍。两者是按每张图的计算量对齐的（这是 DiT 论文的比较方式），不是按参数量对齐的。
2. **U-Net 的前 2500 步用的是 Flow Matching 篇的学习率调度**（OneCycle），后 5000 步才和 DiT 相同；DiT 全程是恒定学习率。
3. **只有一个随机种子**，生成指标用 1000 张图、一个小分类器的特征计算。准确率差 2 个百分点、$$w = 3$$ 时 FD 差 1.6，都在可能的随机波动范围内；$$w = 1$$ 时 FD 的差距（9.7 对 26.1）比较大，相对可信一些。
4. **两者都还没有训练到收敛**，再训练下去结果可能还会变化。

所以能得出的结论是：**在这个小数据集上，训练量足够后，DiT 至少能追上同计算量的 U-Net，并在不用 CFG 的生成指标和高噪声端的误差上略好；训练初期 U-Net 明显更好。** 这和背景里的说法一致：卷积先验让 U-Net 起步更快，训练量增加后这个优势逐渐消失。但这个结果不能说明 DiT 在大规模上更好，也不能说明它在这个数据集上一定更好。

**这个实验能说明什么、不能说明什么。** FashionMNIST 只有 6 万张 $$28 \times 28$$ 的灰度图，训练几千步，正是背景中"数据少、训练短"的情况，U-Net 的卷积先验让它在训练初期明显占优，训练量增加后 DiT 追了上来，这和背景中的分析一致。DiT 相对 U-Net 的优势（效果随计算量稳定提升、大规模时超过 U-Net）是在 ImageNet、数十万到数百万步的规模上得出的，这里的小实验无法复现，也不应该用它来判断哪种结构更好。它能验证的是：实现正确、adaLN-Zero 收敛更快（第 7.2 节）、门控随时间步变化（第 7.3 节）。

**$$p = 2$$ 和 $$p = 4$$。** $$p = 2$$ 的模型参数少一半多，但 token 多 4 倍、计算量约 2 倍，训练损失更低（第 7.2 节的图），在最低噪声端（$$t \le 0.05$$）误差明显更小，块状边界也消失了；但在 2500 步时，中间时间段的误差和 $$p = 4$$ 相当，生成指标反而略差。这个规模下"更小的 patch 更好"只在训练损失和细节上体现出来，要看到论文那样明确的结论，需要更长的训练。

### 7.2 四种条件注入方式

把 block 换成第 4.2 节的另外三种，其余设置完全相同，各训练 400 步（adaLN-Zero 用主实验的前 400 步），比较训练损失（右图；左图是第 7.1 节三个模型 2500 步的训练损失）：

![训练损失曲线](/deep-learning/assets/dit/dit_curves.png)

第 300–400 步的平均训练损失：

| adaLN-Zero | adaLN | In-context | Cross-attention |
|---|---|---|---|
| 0.394 | 0.407 | 0.417 | 0.434 |

- **adaLN-Zero 收敛最快**，在前 200 步领先最多（图中平滑后的曲线在第 200 步约为 0.51，其他三种为 0.60–0.71）；400 步时仍然最低。这和论文图 5 的结论一致。
- **adaLN 和 adaLN-Zero 的唯一区别是门控和零初始化**，前期差距明显，说明"每个 block 从恒等映射开始"确实让训练更顺利。
- 这个规模下四种方式最终差距不大，论文里的差距是在 DiT-XL/2、40 万步上测出来的。这里只能作为趋势参考。

### 7.3 门控学到了什么

对训练 7500 步的 $$p = 4$$ 模型，在三个时间步上计算每个 block 的门控 $$\lvert\alpha_1\rvert$$（注意力分支）和 $$\lvert\alpha_2\rvert$$（MLP 分支）的平均值（对 10 个类别和 192 维取平均）。初始时它们全为 0：

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $$t = 0.05$$：注意力 / MLP | 0.08 / 0.24 | 0.08 / 0.26 | 0.11 / 0.31 | 0.10 / 0.34 | 0.14 / 0.46 | 0.22 / 0.55 |
| $$t = 0.5$$：注意力 / MLP | 0.10 / 0.23 | 0.14 / 0.23 | 0.19 / 0.27 | 0.17 / 0.31 | 0.24 / 0.40 | 0.30 / 0.42 |
| $$t = 0.95$$：注意力 / MLP | 0.12 / 0.26 | 0.18 / 0.23 | 0.22 / 0.26 | 0.20 / 0.30 | 0.26 / 0.32 | 0.30 / 0.31 |

- 所有门控都离开了 0，整体上越深的 block 门控越大。
- **门控随时间步变化。** 注意力分支的门控在高噪声端（$$t = 0.95$$）更大，MLP 分支的门控在低噪声端的深层最大（0.55）。一种可能的解释是：输入几乎全是噪声时，整体结构要靠 token 之间交换的全局信息决定；最后去除少量噪声时，主要是逐 token 的局部修正。这正是"门控由条件 $$c$$ 回归"的作用：同一个网络在不同时间步自动调整各分支的权重。

## 8. 后续发展

DiT 之后，几乎所有大规模图像和视频生成模型都改用 Transformer 主干，主要的变化集中在两处：**训练目标**（从 DDPM 换成 Flow Matching）和**文本条件的注入方式**（类别是一个向量，文本是一串 token）。

| 模型 | 年份 | 相对 DiT 的主要改动 |
|---|---|---|
| U-ViT | 2023 | 与 DiT 同期。时间、条件、图像全部作为 token（in-context），并在浅层和深层之间加 U-Net 式的长跳连 |
| SiT [[4]](#ref-4) | 2024 | 网络不变，训练目标换成 Flow Matching 一类的插值，同结构下 FID 2.27 → 2.06 |
| PixArt-α [[5]](#ref-5) | 2023 | 文生图：每个 block 加 cross-attention 读 T5 文本 token；adaLN-single 让所有 block 共享一个从 $$t$$ 算调制参数的 MLP，每层只加一个可学习的偏移，参数量减少约 27% |
| SD3（MM-DiT）[[6]](#ref-6) | 2024 | 文本 token 和图像 token 各用一套权重（两条流），在注意力里拼在一起做联合注意力；$$t$$ 和池化后的文本向量通过 adaLN 注入；Rectified Flow 训练 |
| FLUX.1 | 2024 | 前若干层是 MM-DiT 式的双流 block，后面是文本和图像共用权重的单流 block；位置编码换成二维 RoPE；约 12B 参数 |
| 视频 DiT | 2024– | 视频 latent 按时间和空间一起切块（spacetime patch），token 数大幅增加；Sora、Movie Gen、Wan、HunyuanVideo 等都是"时空 VAE + DiT + Flow Matching"的结构，见 [实时视频扩散加速](/deep-learning/fast-video-diffusion/) |

**为什么文本条件不再用 adaLN。** 类别只是一个向量，adaLN 把它压成每层的缩放、平移很合适；文本是一串（几十到几百个）token，压成一个向量会丢掉大部分信息。所以文生图的 DiT 通常保留 adaLN 只注入时间（和一个池化后的全局文本向量），而用 cross-attention（PixArt-α）或联合注意力（MM-DiT）读取完整的文本 token 序列。

**MM-DiT 的联合注意力**（SD3）可以写成：文本 token $$h_{\text{txt}}$$ 和图像 token $$h_{\text{img}}$$ 各自用自己的权重算 $$Q, K, V$$，再沿序列维拼接后做一次注意力，结果切开后各自经过自己的 MLP：

$$
[Q; K; V] = \big[\,W^{\text{txt}}_{qkv} h_{\text{txt}}\ \big\|\ W^{\text{img}}_{qkv} h_{\text{img}}\,\big],\qquad
\mathrm{Attn} = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d}}\Big)V
$$

这样文本和图像可以双向交流（cross-attention 只能图像读文本），同时两种模态的表示又不必共用一套权重。

## 9. 自检

下面的检查不需要训练，用于确认实现的结构没有写错：

```python
import torch

m = DiT(input_size=28, patch_size=4, in_channels=1, hidden_size=192, depth=6, num_heads=3,
        num_classes=10, learn_sigma=False)
z, t, y = torch.randn(4, 1, 28, 28), torch.rand(4) * 1000, torch.randint(0, 10, (4,))

# 1. 输出和输入同形状；学习方差时通道数翻倍
assert m(z, t, y).shape == (4, 1, 28, 28)
assert DiT(input_size=28, patch_size=4, in_channels=1, hidden_size=192, depth=2, num_heads=3,
           num_classes=10, learn_sigma=True)(z, t, y).shape == (4, 2, 28, 28)

# 2. 零初始化：初始输出全为 0，每个 block 是恒等映射
assert m(z, t, y).abs().max() == 0
x = torch.randn(4, 49, 192); c = torch.randn(4, 192)
assert torch.equal(m.blocks[0](x, c), x)

# 3. 位置编码：49 个 token 互不相同，且不参与训练
pe = m.pos_embed[0]
assert pe.shape == (49, 192) and torch.cdist(pe, pe).add(torch.eye(49)).min() > 0
assert "pos_embed" not in dict(m.named_parameters())

# 4. patchify 后再 unpatchify 能还原（用恒等的 token 内容模拟）
h, p = 7, 4
tok = z.reshape(4, 1, h, p, h, p).permute(0, 2, 4, 3, 5, 1).reshape(4, h * h, p * p)
m.out_channels = 1
assert torch.equal(m.unpatchify(tok), z)

# 5. 一步更新后门控离开 0（梯度不为 0）
m2 = DiT(input_size=28, patch_size=4, in_channels=1, hidden_size=64, depth=2, num_heads=2,
         num_classes=10, learn_sigma=False)
opt = torch.optim.SGD(m2.parameters(), lr=0.1)
for _ in range(2):                         # 第一步只更新最后的输出层，第二步门控才收到梯度
    loss = (m2(z, t, y) - z).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
assert m2.blocks[0].adaLN_modulation[-1].weight.abs().sum() > 0
print("all checks passed")
```

第 5 项值得说明：初始时最后的输出线性层为 0，所以第一步反向传播时，梯度传不到更早的层（输出层权重为 0，传回 token 的梯度为 0）；第一步更新后输出层不再为 0，第二步各 block 的门控才收到非零梯度。这是零初始化网络的正常行为，不影响训练。

## 10. 要点回顾

- **DiT 解决的问题。** 证明扩散模型不需要 U-Net，标准 Transformer 就够；并且 Transformer 主干的生成质量随计算量稳定提升，可以沿用其他领域的缩放经验。
- **结构只有五步。** Patchify（$$p \times p$$ 小块变 token）→ 加固定的二维 sin-cos 位置编码 → $$L$$ 个 Transformer block → final layer → unpatchify。在 LDM 的 latent 上工作，token 数才可控。
- **条件用 adaLN-Zero 注入。** 时间和类别嵌入相加成 $$c$$，回归出每个 block 的 LayerNorm 缩放、平移和残差门控；门控初始化为 0，每个 block 起初是恒等映射，初始输出为 0，训练稳定，不需要 warmup。
- **效果跟着计算量走。** patch 越小、模型越大，Gflops 越高，FID 越低；参数量相同时，更小的 patch 更好。
- **网络和训练目标解耦。** 论文用 DDPM（预测 $$\epsilon$$ + 学习方差）；换成 Flow Matching 只需改输出通道和时间输入，这就是 SiT 和 SD3 的做法。
- **适用场景。** DiT 的优势在大数据、大算力、需要继续放大的场景，典型应用是大规模文生图和视频生成；数据少、训练短时，带卷积先验的 U-Net 更容易训好。FashionMNIST 的小实验中，相同计算量训练 2500 步时 DiT 明显不如 U-Net；两者都训练到 7500 步后，训练损失几乎相同，DiT 的生成指标略好（不用 CFG 时更明显），U-Net 在低噪声端误差仍更小。这个规模只能说明 DiT 训练足够后能追上 U-Net，不能用来判断哪种结构更好。四种条件注入方式中 adaLN-Zero 收敛最快。
- **后续的主线。** 文本条件改用 cross-attention 或 MM-DiT 的联合注意力，训练目标换成 Rectified Flow，位置编码换成 RoPE，再扩展到视频的时空 patch。
