---
date: 2026-11-28 03:30:00 -0800
tags: [Transformer, Multimodal]
keywords: >-
  ViT, Vision Transformer, patch embedding, CLIP, contrastive learning, InfoNCE, zero-shot classification, SigLIP, PyTorch.
summary: >-
  ViT 把图像切成 patch 当作 token，用标准 Transformer 做图像分类；CLIP 用一个 ViT 图像编码器和一个 Transformer 文本编码器，在 4 亿对网络图文上做对比学习，得到图文对齐的向量空间，可以直接用文字做零样本分类。本文介绍两者要解决的问题和历史背景、ViT 的 patch embedding / [CLS] / 位置编码 / 数据量与归纳偏置、CLIP 的双塔结构 / 对称对比损失 / 可学习温度 / 零样本分类和提示集成，给出从零实现的 PyTorch 代码（ViT 与 torchvision 输出完全一致，CLIP 参数量与官方模型一致），在 FashionMNIST 上做 ViT 和 CNN 的数据量对比、位置编码可视化和一个迷你 CLIP，最后梳理 CLIP 在 Stable Diffusion、IP-Adapter、LLaVA 中的用法和 OpenCLIP、SigLIP 等后续工作。
layout: note
title: "ViT 与 CLIP：把图像变成 token，再用文字监督视觉"
description: "Vision Transformer and CLIP from scratch: why they exist, patch embedding, CLS token and position embeddings, data size versus inductive bias, the two-tower contrastive model with a learned temperature, zero-shot classification, verified PyTorch implementations, FashionMNIST experiments, and how CLIP is used in Stable Diffusion, IP-Adapter and LLaVA."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig svg .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;fill:#475467}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
</style>

# ViT 与 CLIP：把图像变成 token，再用文字监督视觉

[DiT](/deep-learning/dit/) 篇用到了 ViT 的做法：把图像切成 patch，每个 patch 当作一个 token 送进 Transformer。[Stable Diffusion](/deep-learning/stable-diffusion/) 篇用到了 CLIP 的文本编码器，[IP-Adapter](/deep-learning/ip-adapter/) 篇用到了 CLIP 的图像编码器。这两个模型是近几年视觉和多模态模型的基础，本文把它们完整地讲一遍。

- **ViT**（Vision Transformer，Dosovitskiy 等，2021）[[1]](#ref-1)：几乎不加修改地把 NLP 的 Transformer encoder 用在图像分类上。它说明了卷积不是视觉模型必需的，数据足够多时，Transformer 的效果超过 CNN。
- **CLIP**（Contrastive Language-Image Pre-training，Radford 等，2021）[[2]](#ref-2)：一个图像编码器（ViT 或 ResNet）加一个文本编码器（Transformer），在 4 亿对网络图文上用对比学习训练，让配对的图和文的向量靠近。训练好之后，不需要任何标注样本，只要写出类别名就能分类（零样本分类）；它的两个编码器也被大量生成模型和多模态模型直接拿来用。

本文的记号和 [DiT](/deep-learning/dit/) 篇一致：patch 大小 $$p$$，token 数 $$N$$，隐藏维度 $$D$$，层数 $$L$$。batch 大小记作 $$B$$；注意 [Stable Diffusion](/deep-learning/stable-diffusion/) 篇介绍 CLIP 的代码片段里 batch 记作 $$N$$，本文统一用 $$B$$，$$N$$ 只表示 token 数。

## 参考资料

本文主要参考 ViT 原论文 [1] 和 CLIP 原论文 [2]，CLIP 的实现细节对照官方代码 [3]；DeiT [4] 用于说明 ViT 在小数据上的训练方法，OpenCLIP 的缩放研究 [5] 和 SigLIP [6] 用于第 5 节的后续工作。正文中用 [n] 标注具体出处。

- <a id="ref-1"></a>**[1] ViT**：Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)。patch embedding、[CLS]、位置编码、模型配置和数据量实验都来自这篇。
- <a id="ref-2"></a>**[2] CLIP**：Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)。双塔结构、对比损失、零样本分类、提示工程和鲁棒性实验都来自这篇。
- <a id="ref-3"></a>**[3] CLIP 官方代码**：[openai/CLIP](https://github.com/openai/CLIP)。`clip/model.py` 是整个模型，本文的实现按它逐模块对应。
- <a id="ref-4"></a>**[4] DeiT**：Touvron et al. *Training data-efficient image transformers & distillation through attention*. ICML 2021. [arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877)。只用 ImageNet-1k、靠强数据增强和蒸馏把 ViT 训好。
- <a id="ref-5"></a>**[5] OpenCLIP 缩放**：Cherti et al. *Reproducible scaling laws for contrastive language-image learning*. CVPR 2023. [arxiv.org/abs/2212.07143](https://arxiv.org/abs/2212.07143)。在公开的 LAION 数据上复现并放大 CLIP，代码为 [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)。
- <a id="ref-6"></a>**[6] SigLIP**：Zhai et al. *Sigmoid Loss for Language Image Pre-Training*. ICCV 2023. [arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343)。把 softmax 对比损失换成逐对的 sigmoid 损失。

## 背景

本节介绍后文用到的基础知识，以及 ViT 和 CLIP 各自要解决的问题。熟悉这两个模型的读者可以直接阅读第 0 节。

### 前置知识

- **Transformer encoder**：self-attention、多头注意力、LayerNorm、残差和前馈网络。见 [Transformer 学习](/deep-learning/transformer-study/)。ViT 只用 encoder；CLIP 的文本编码器也是 encoder 结构，但加了 causal mask（每个 token 只看自己和前面的 token）。
- **Patchify**：把 $$p \times p$$ 的小块展平再线性映射成一个 token，等价于核大小和步长都为 $$p$$ 的卷积。见 [DiT](/deep-learning/dit/) 第 3.1 节。
- **CLIP 在生成模型里的位置**：SD 1.x 用 CLIP ViT-L/14 的文本编码器把 prompt 变成 $$77 \times 768$$ 的向量，见 [Stable Diffusion 拆解](/deep-learning/stable-diffusion/) 第 2 节；IP-Adapter 用 CLIP 图像编码器提取参考图特征，见 [IP-Adapter](/deep-learning/ip-adapter/) 第 2 节。本文讲这两个编码器是怎样训练出来的、为什么它们的向量"懂画面"。
- **交叉熵和 softmax**：CLIP 的损失就是把相似度矩阵的每一行（和每一列）当作一个分类问题做交叉熵。

### 为什么要有 ViT

#### 2020 年之前的视觉模型

从 2012 年的 AlexNet 到 2020 年，图像分类、检测、分割的主干几乎都是 CNN（VGG、ResNet、EfficientNet 等）。同一时期，NLP 在 2017 年之后全面换成了 Transformer，并且出现了 BERT、GPT 这种"在海量数据上预训练、再迁移到下游任务"的范式，模型越大效果越好。

视觉领域也尝试过把注意力引入 CNN，大致有两类：

- **在 CNN 里加注意力模块**：如 Non-local Networks、SENet、把 ResNet 的部分卷积换成自注意力。主干仍然是 CNN。
- **直接在像素上做自注意力**：每个像素当作一个 token。一张 $$224 \times 224$$ 的图有 50176 个像素，全局注意力矩阵有约 25 亿个元素，算不动；于是只能做局部窗口注意力或稀疏注意力（如 Stand-Alone Self-Attention、Axial Attention），需要专门实现，在硬件上跑不快，规模也做不大。

#### ViT 的做法

ViT [[1]](#ref-1) 的出发点是**尽量不改 Transformer**，直接复用 NLP 的模型和训练基础设施：

1. 把 $$224 \times 224$$ 的图切成 $$16 \times 16$$ 的小块，得到 $$14 \times 14 = 196$$ 个 patch。序列长度从 50176 降到 196，全局注意力就算得动了。
2. 每个 patch 展平成 $$16 \times 16 \times 3 = 768$$ 维，线性映射成一个 token，相当于 NLP 里的一个词。论文标题 *An Image is Worth 16x16 Words* 说的就是这件事。
3. 在序列前面加一个可学习的 [CLS] token（和 BERT 一样），加上可学习的位置编码，送进标准 Transformer encoder，最后用 [CLS] 的输出做分类。

除了切 patch 这一步，模型里没有任何针对图像的设计。

#### 归纳偏置和数据量

CNN 的结构里写了两个关于图像的先验（**归纳偏置**）：

- **局部性**：卷积核只看邻近的像素，"相邻像素相关"不需要学；
- **平移等变性**：同一个卷积核在所有位置共享，"一个边缘出现在哪里都是边缘"不需要学。

ViT 里几乎没有这些先验：只有切 patch 时用到了二维结构，位置编码是从随机初始化开始学的，patch 之间的空间关系全部要从数据中学到。

<figure class="ddpm-fig">
<svg viewBox="0 0 760 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="卷积只看局部邻域，自注意力每个 token 从第一层就能看到全图">
  <defs><marker id="ah1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2b6cb0"/></marker></defs>
  <text x="190" y="22" text-anchor="middle">卷积（CNN）</text>
  <g transform="translate(80,40)">
    <g stroke="#d0d5dd" fill="#f8fafc">
      <rect x="0" y="0" width="28" height="28"/><rect x="28" y="0" width="28" height="28"/><rect x="56" y="0" width="28" height="28"/><rect x="84" y="0" width="28" height="28"/><rect x="112" y="0" width="28" height="28"/><rect x="140" y="0" width="28" height="28"/><rect x="168" y="0" width="28" height="28"/>
      <rect x="0" y="28" width="28" height="28"/><rect x="28" y="28" width="28" height="28"/><rect x="56" y="28" width="28" height="28"/><rect x="84" y="28" width="28" height="28"/><rect x="112" y="28" width="28" height="28"/><rect x="140" y="28" width="28" height="28"/><rect x="168" y="28" width="28" height="28"/>
      <rect x="0" y="56" width="28" height="28"/><rect x="28" y="56" width="28" height="28"/><rect x="56" y="56" width="28" height="28"/><rect x="84" y="56" width="28" height="28"/><rect x="112" y="56" width="28" height="28"/><rect x="140" y="56" width="28" height="28"/><rect x="168" y="56" width="28" height="28"/>
      <rect x="0" y="84" width="28" height="28"/><rect x="28" y="84" width="28" height="28"/><rect x="56" y="84" width="28" height="28"/><rect x="84" y="84" width="28" height="28"/><rect x="112" y="84" width="28" height="28"/><rect x="140" y="84" width="28" height="28"/><rect x="168" y="84" width="28" height="28"/>
      <rect x="0" y="112" width="28" height="28"/><rect x="28" y="112" width="28" height="28"/><rect x="56" y="112" width="28" height="28"/><rect x="84" y="112" width="28" height="28"/><rect x="112" y="112" width="28" height="28"/><rect x="140" y="112" width="28" height="28"/><rect x="168" y="112" width="28" height="28"/>
      <rect x="0" y="140" width="28" height="28"/><rect x="28" y="140" width="28" height="28"/><rect x="56" y="140" width="28" height="28"/><rect x="84" y="140" width="28" height="28"/><rect x="112" y="140" width="28" height="28"/><rect x="140" y="140" width="28" height="28"/><rect x="168" y="140" width="28" height="28"/>
    </g>
    <rect x="28" y="28" width="84" height="84" fill="#2b6cb0" fill-opacity="0.18" stroke="#2b6cb0" stroke-width="2"/>
    <rect x="56" y="56" width="28" height="28" fill="#2b6cb0"/>
    <rect x="112" y="84" width="84" height="84" fill="#d9822b" fill-opacity="0.15" stroke="#d9822b" stroke-width="2" stroke-dasharray="5 3"/>
    <rect x="140" y="112" width="28" height="28" fill="#d9822b"/>
  </g>
  <text x="190" y="228" text-anchor="middle" class="sm">每个输出只看 3×3 邻域；两个位置用同一个卷积核</text>
  <text x="190" y="244" text-anchor="middle" class="sm">远处的信息要靠一层层堆叠才能传过来</text>
  <text x="570" y="22" text-anchor="middle">自注意力（ViT）</text>
  <g transform="translate(470,40)">
    <g stroke="#d0d5dd" fill="#f8fafc">
      <rect x="0" y="0" width="48" height="40"/><rect x="50" y="0" width="48" height="40"/><rect x="100" y="0" width="48" height="40"/><rect x="150" y="0" width="48" height="40"/>
      <rect x="0" y="42" width="48" height="40"/><rect x="50" y="42" width="48" height="40"/><rect x="100" y="42" width="48" height="40"/><rect x="150" y="42" width="48" height="40"/>
      <rect x="0" y="84" width="48" height="40"/><rect x="50" y="84" width="48" height="40"/><rect x="100" y="84" width="48" height="40"/><rect x="150" y="84" width="48" height="40"/>
      <rect x="0" y="126" width="48" height="40"/><rect x="50" y="126" width="48" height="40"/><rect x="100" y="126" width="48" height="40"/><rect x="150" y="126" width="48" height="40"/>
    </g>
    <rect x="50" y="42" width="48" height="40" fill="#2b6cb0"/>
    <g stroke="#2b6cb0" stroke-width="1.3" opacity="0.75">
      <line x1="74" y1="62" x2="24" y2="20"/><line x1="74" y1="62" x2="174" y2="20"/><line x1="74" y1="62" x2="124" y2="62"/><line x1="74" y1="62" x2="174" y2="104"/><line x1="74" y1="62" x2="24" y2="146"/><line x1="74" y1="62" x2="174" y2="146"/><line x1="74" y1="62" x2="124" y2="146"/><line x1="74" y1="62" x2="24" y2="104"/>
    </g>
  </g>
  <text x="570" y="228" text-anchor="middle" class="sm">每个 patch token 在第一层就和所有 token 做注意力</text>
  <text x="570" y="244" text-anchor="middle" class="sm">谁和谁相关、相邻 patch 是否相似，全部从数据中学</text>
</svg>
<figcaption>卷积把"局部性"和"平移等变"写进了结构；ViT 没有这些先验，要靠数据学出来。</figcaption>
</figure>

这带来了 ViT 论文最重要的实验结论 [[1]](#ref-1)：

- **只在 ImageNet-1k（128 万张）上预训练**：ViT 不如同等规模的 ResNet（BiT），而且更大的 ViT-L 反而不如 ViT-B，说明数据不够时大模型会过拟合。
- **在 ImageNet-21k（1400 万张）上预训练**：两者相当。
- **在 JFT-300M（3 亿张）上预训练**：ViT 超过 ResNet。ViT-H/14 在 ImageNet 上达到 88.55% 的 top-1，高于当时最好的 BiT-L（87.54%），而预训练计算量约为后者的四分之一（2.5k 对 9.9k TPUv3 核·天）。

论文的结论是：**数据少时，卷积的归纳偏置很有用；数据足够多时，从数据中学习这些规律就够了，而且学到的更灵活。** 第 4.1 节在 FashionMNIST 上用 1k、6k、60k 张图做了一个小实验，看看在这种"数据少"的规模下两种结构的差别。

#### 之后怎样让 ViT 在小数据上也能用

ViT 的结论一度被理解为"ViT 必须用 3 亿张图预训练"。后续工作说明，问题主要出在训练方法上：

- **DeiT** [[4]](#ref-4)：只用 ImageNet-1k，加上强数据增强（RandAugment、Mixup、CutMix、随机擦除）、更强的正则化（stochastic depth）和更长的训练，ViT-B 达到 81.8%；再加上从 CNN 教师蒸馏，达到 83.4%，和同等规模的 CNN 相当。
- **分层 ViT**：Swin Transformer 等把局部窗口注意力和逐级下采样加回来，相当于重新引入一部分卷积的先验，更适合检测、分割这类需要多尺度特征的任务。
- **自监督预训练**：MAE（遮住 75% 的 patch 让模型重建）、DINO / DINOv2 让 ViT 不需要标注就能从大量图像中学到好的特征。

反过来，ConvNeXt（2022）把 CNN 按 ViT 的训练方法和一些结构细节"现代化"之后，在 ImageNet 规模上和 Swin 打成平手。所以在 ImageNet 这个规模上，两种结构的差距很大一部分来自训练方法；ViT 真正明显的优势在更大的规模和多模态上。

#### ViT 适合什么场景

| 场景 | 更适合 | 原因 |
|---|---|---|
| 数据少（几万张以内）、算力少、从头训练 | CNN | 卷积先验相当于"免费"的知识，收敛快 |
| 大规模预训练（千万到数十亿张） | ViT | 先验不再是优势，Transformer 随规模提升更稳定 |
| 多模态（图文、视频、语言模型接图像） | ViT | 输出就是 token 序列，可以直接和文本 token 拼在一起 |
| 检测、分割等密集预测 | 分层 ViT 或 CNN | 需要多尺度特征；普通 ViT 只有一种分辨率 |
| 手机等端侧设备、实时推理 | CNN 或小 ViT | 卷积在端侧硬件上优化成熟，注意力随 token 数平方增长 |
| 扩散模型主干（大规模） | DiT（ViT 结构） | 见 [DiT](/deep-learning/dit/) 背景一节 |

今天，ViT 是大多数视觉基础模型的主干：CLIP、DINOv2、SAM 的图像编码器（MAE 预训练的 ViT-H）、LLaVA 等多模态大模型的视觉部分，以及 DiT 系列生成模型。

### 为什么要有 CLIP

#### 有监督分类的局限

2021 年之前，视觉模型的预训练主要靠有标注的分类数据集：ImageNet-1k 有 1000 个类别、128 万张图，JFT-300M 有约 1.8 万个类别。这种方式有三个局限：

1. **类别是固定的。** 模型只认识训练时的那 1000 个类别。想识别一个新类别（比如"电动滑板车"），需要重新收集标注、重新训练分类头。
2. **标注很贵。** ImageNet 用众包标注了数年。把数据规模再扩大 100 倍，靠人工标注几乎不可能。
3. **类别标签的信息量很少。** 一张图只对应一个类别编号，图里的其他物体、颜色、动作、风格、文字全部被丢掉了。一句描述"一只坐在红色沙发上的黑猫"包含的信息比标签"猫"多得多。

#### 用自然语言做监督

互联网上有大量现成的"图片 + 配文"（网页的 alt 文本、标题、说明）。如果能直接用这些文字监督视觉模型，就同时解决了上面三个问题：不需要人工标注，规模可以很大；文字可以描述任意概念，不限于固定类别；一句话包含的信息也远多于一个类别编号。

CLIP 之前已经有工作这样尝试（VirTex、ICMLM 用描述生成训练图像特征，ConVIRT 在医学图像上用对比学习），但数据规模都只有十几万到几十万对。Visual N-Grams（2017）做过零样本迁移，在 ImageNet 上只有 11.5% 的准确率。CLIP 的贡献是把规模做大，并找到了效率高的训练目标：

- **数据**：从互联网收集了 4 亿对图文（论文称为 WIT，WebImageText），用约 50 万个查询词保证覆盖面。
- **训练目标：对比学习，而不是生成描述。** 论文比较了三种目标 [[2]](#ref-2)：用 Transformer 语言模型逐词生成描述、预测描述中的词袋（bag of words）、以及对比学习（判断哪张图和哪段文字配对）。在相同计算量下，预测词袋的零样本 ImageNet 准确率提升速度是生成描述的 3 倍，对比学习又是预测词袋的 4 倍。原因是：同一张图可以有无数种合理的描述，逐词生成要求模型猜中具体用词，任务太难；对比学习只要求在一个 batch 里认出"哪段文字是配给这张图的"，难度合适，信号也更直接。

#### 零样本分类：把分类器换成文字

训练好之后，CLIP 有一个图像编码器和一个文本编码器，输出在同一个向量空间里。分类时不需要训练分类头：

1. 把每个类别名写成一句话，比如 "a photo of a dog."、"a photo of a cat."，用文本编码器得到每个类别的向量；
2. 用图像编码器得到图像向量；
3. 和哪个类别的向量最相似，就分到哪一类。

文本向量起到了"分类器权重"的作用，而且可以随时用文字定义新的类别。CLIP 的最大模型（ViT-L/14@336px）用这种方式在 ImageNet 上达到 76.2% 的 top-1，和完全在 ImageNet 128 万张有标注图像上训练的 ResNet-50 相当，但一张 ImageNet 的训练图都没有用 [[2]](#ref-2)。更重要的是鲁棒性：在 ImageNet 的分布偏移版本（素描、卡通、对抗筛选的 ImageNet-R、ImageNet-Sketch、ImageNet-A 等）上，零样本 CLIP 的准确率下降远小于在 ImageNet 上有监督训练的模型。

#### CLIP 的局限

CLIP 论文本身也列出了不少局限 [[2]](#ref-2)，理解它们有助于理解后续工作：

- **细粒度和抽象任务差**：区分车型、花的品种，数物体的个数，判断距离远近，效果都不好。零样本在 MNIST 手写数字上只有 88%，不如在像素上直接训练的逻辑回归。
- **组合关系弱**：后续研究发现 CLIP 的文本向量很多时候接近"词袋"，"马骑着人"和"人骑着马"的向量可能很接近。这也是早期文生图模型经常搞错属性绑定（"红色的方块和蓝色的球"画反颜色）的原因之一。
- **数据有偏**：网络图文带有社会偏见，配文和图像也常常不完全对应。
- **需要非常大的 batch 和算力**：原始 CLIP 用 32768 的 batch，最大的 ViT-L/14 在 256 张 V100 上训练了 12 天。

#### CLIP 为什么对生成模型重要

CLIP 的价值很大一部分体现在"被别的模型拿来用"：

| 用法 | 用的是哪部分 | 例子 |
|---|---|---|
| 文生图的文本条件 | 文本编码器（逐 token 输出） | SD 1.x（ViT-L/14）、SD 2.x（OpenCLIP ViT-H/14）、SDXL（两个 CLIP 拼接）、SD3 和 FLUX（CLIP + T5） |
| 图像提示 | 图像编码器 | IP-Adapter、DALL·E 2 的 unCLIP 解码器 |
| 多模态大模型的视觉部分 | 图像编码器（逐 patch 输出） | LLaVA（CLIP ViT-L/14@336） |
| 评价指标 | 两个编码器的相似度 | CLIP score（衡量生成图和 prompt 的匹配程度） |
| 开放词表的检测、分割 | 两个编码器 | OWL-ViT、各种 open-vocabulary 分割方法 |

它的文本向量之所以适合指导图像生成，是因为它是和图像一起训练出来的：两段文字的向量接近，意味着它们描述的画面接近，而不只是用词接近。

### 数值示例

**ViT-B/16 处理一张 224×224 的图：**

| 步骤 | 张量形状 | 说明 |
|---|---|---|
| 输入 | $$B \times 3 \times 224 \times 224$$ | |
| 切 patch 并线性映射 | $$B \times 196 \times 768$$ | $$14 \times 14$$ 个 patch，每个 $$16 \cdot 16 \cdot 3 = 768$$ 维，映射到 $$D = 768$$ |
| 加 [CLS] 和位置编码 | $$B \times 197 \times 768$$ | 位置编码也是 $$197 \times 768$$ 个可学习参数 |
| 12 个 Transformer block | $$B \times 197 \times 768$$ | 每个头的注意力矩阵是 $$197 \times 197$$ |
| 取 [CLS]，过分类头 | $$B \times 1000$$ | 约 8660 万参数，每张图约 175 亿次乘加 |

**CLIP 的相似度矩阵：** 假设一个 batch 有 $$B = 3$$ 对图文，归一化后的图像向量和文本向量两两做点积（余弦相似度）为

$$
\begin{bmatrix} 0.30 & 0.10 & 0.05 \\ 0.12 & 0.25 & 0.08 \\ 0.02 & 0.06 & 0.28 \end{bmatrix}
$$

第 $$i$$ 行第 $$j$$ 列是第 $$i$$ 张图和第 $$j$$ 段文字的相似度，对角线是配对的。余弦相似度只在 $$[-1, 1]$$ 之间，差别很小，直接做 softmax 几乎是均匀分布；CLIP 把它乘以 $$1/\tau$$（初始 $$\tau = 0.07$$，即乘以约 14.3）再做 softmax。第一行变成 $$[4.29, 1.43, 0.71]$$，softmax 后对角线的概率是 0.92，这一行的交叉熵损失是 $$-\log 0.92 = 0.08$$。如果不乘 $$1/\tau$$，概率只有 0.39，损失为 0.95。温度决定了损失对相似度差别有多敏感，第 3.4 节会详细讨论。

### 一个可以运行的最小例子

下面的代码把一张图切成 patch、加上 [CLS] 和位置编码，然后用上面的相似度矩阵算 CLIP 的对称对比损失：

```python
import torch, torch.nn as nn, torch.nn.functional as F

B, C, H, p, D = 2, 3, 224, 16, 768
x = torch.randn(B, C, H, H)
patch_embed = nn.Conv2d(C, D, kernel_size=p, stride=p)   # 等价于展平 p*p*C 后做 Linear
tok = patch_embed(x).flatten(2).transpose(1, 2)          # (B, 196, 768)
cls = torch.zeros(1, 1, D).expand(B, -1, -1)             # 可学习参数，这里用 0 代替
pos = torch.randn(1, 1 + tok.shape[1], D) * 0.02         # 可学习的位置编码
z0 = torch.cat([cls, tok], dim=1) + pos
print(z0.shape)                                          # torch.Size([2, 197, 768])

sim = torch.tensor([[0.30, 0.10, 0.05],
                    [0.12, 0.25, 0.08],
                    [0.02, 0.06, 0.28]])                 # 归一化向量的点积
logits = sim / 0.07                                      # 乘以 1/tau
y = torch.arange(3)                                      # 第 i 张图配第 i 段文字
loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y)) / 2
print(logits.softmax(1)[0].round(decimals=2), round(loss.item(), 3))
# tensor([0.9200, 0.0500, 0.0300]) 0.119
```

## 0. 全文路线

| 节 | 内容 |
|---|---|
| 1 | 符号约定 |
| 2 | ViT：patch embedding、[CLS]、位置编码、Transformer block、完整代码和验证、模型配置 |
| 3 | CLIP：双塔结构、文本编码器、对称对比损失、可学习温度、零样本分类和提示集成、代码和验证 |
| 4 | 实验（FashionMNIST）：ViT 和 CNN 的数据量对比、位置编码学到了什么、一个迷你 CLIP |
| 5 | 后续发展：OpenCLIP、SigLIP、EVA-CLIP，以及 CLIP 在生成模型里的具体用法 |
| 6–8 | 总结、自检、要点回顾 |

## 1. 符号约定

| 符号 | 含义 | ViT-B/16 的取值 |
|---|---|---|
| $$B$$ | batch 大小（CLIP 里也是一个 batch 的图文对数） | CLIP 训练时 32768 |
| $$C, H, W$$ | 输入图像的通道数和高宽 | $$3, 224, 224$$ |
| $$p$$ | patch 边长 | 16 |
| $$N = HW/p^2$$ | patch token 数（不含 [CLS]） | 196 |
| $$D$$ | Transformer 的隐藏维度 | 768 |
| $$L$$ | block 数 | 12 |
| $$h$$ | 注意力头数 | 12 |
| $$\mathbf{z}_\ell$$ | 第 $$\ell$$ 个 block 的输出，形状 $$(1+N) \times D$$ | |
| $$E$$ | CLIP 共享向量空间的维度 | ViT-B/32 为 512，ViT-L/14 为 768 |
| $$\mathbf{u}_i, \mathbf{v}_j$$ | 第 $$i$$ 张图、第 $$j$$ 段文字的归一化向量，$$\lVert\mathbf{u}_i\rVert = \lVert\mathbf{v}_j\rVert = 1$$ | |
| $$\tau$$，$$s = 1/\tau$$ | 温度和它的倒数（logit 缩放） | 初始 $$\tau = 0.07$$，$$s \le 100$$ |

和其他几篇的对应关系：$$p$$、$$N$$、$$D$$、$$L$$ 与 [DiT](/deep-learning/dit/) 篇完全相同；[Stable Diffusion](/deep-learning/stable-diffusion/) 篇的 CLIP 代码片段里 batch 记作 $$N$$，本文记作 $$B$$。ViT 论文用 $$P$$ 表示 patch 边长、$$N$$ 表示 patch 数，和本文只有大小写的差别。

## 2. ViT

### 2.1 整体结构

<figure class="ddpm-fig">
<svg viewBox="0 0 760 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="ViT 的结构：切 patch、线性映射、加 CLS 和位置编码、L 个 Transformer block、取 CLS 输出分类">
  <defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#667085"/></marker></defs>
  <g transform="translate(14,92)">
    <rect x="0" y="0" width="96" height="96" fill="#e6eef8" stroke="#2b6cb0"/>
    <g stroke="#2b6cb0" stroke-width="1"><line x1="24" y1="0" x2="24" y2="96"/><line x1="48" y1="0" x2="48" y2="96"/><line x1="72" y1="0" x2="72" y2="96"/><line x1="0" y1="24" x2="96" y2="24"/><line x1="0" y1="48" x2="96" y2="48"/><line x1="0" y1="72" x2="96" y2="72"/></g>
  </g>
  <text x="62" y="80" text-anchor="middle" class="sm">图像 C×H×W</text>
  <text x="62" y="206" text-anchor="middle" class="sm">切成 N 个 p×p 块</text>
  <line x1="116" y1="140" x2="150" y2="140" stroke="#667085" marker-end="url(#ah2)"/>
  <rect x="154" y="112" width="96" height="56" rx="8" fill="#f2f4f7" stroke="#98a2b3"/>
  <text x="202" y="135" text-anchor="middle">线性映射</text>
  <text x="202" y="153" text-anchor="middle" class="mono">Conv k=s=p</text>
  <line x1="250" y1="140" x2="280" y2="140" stroke="#667085" marker-end="url(#ah2)"/>
  <g transform="translate(286,60)">
    <rect x="0" y="0" width="22" height="22" fill="#fde7d0" stroke="#d9822b"/>
    <rect x="0" y="26" width="22" height="22" fill="#e6eef8" stroke="#2b6cb0"/>
    <rect x="0" y="52" width="22" height="22" fill="#e6eef8" stroke="#2b6cb0"/>
    <rect x="0" y="78" width="22" height="22" fill="#e6eef8" stroke="#2b6cb0"/>
    <text x="11" y="122" text-anchor="middle" class="sm">⋮</text>
    <rect x="0" y="136" width="22" height="22" fill="#e6eef8" stroke="#2b6cb0"/>
    <text x="34" y="16" class="sm">[CLS]</text>
    <text x="34" y="66" class="sm">patch 1…N</text>
    <text x="-4" y="184" class="sm">+ 位置编码</text>
    <text x="-4" y="200" class="mono">(1+N)×D</text>
  </g>
  <line x1="372" y1="140" x2="402" y2="140" stroke="#667085" marker-end="url(#ah2)"/>
  <rect x="406" y="44" width="160" height="196" rx="10" fill="#ffffff" stroke="#98a2b3" stroke-dasharray="5 3"/>
  <text x="486" y="36" text-anchor="middle" class="sm">重复 L 次（Pre-LN）</text>
  <rect x="426" y="60" width="120" height="30" rx="6" fill="#f2f4f7" stroke="#98a2b3"/><text x="486" y="80" text-anchor="middle">LayerNorm</text>
  <rect x="426" y="98" width="120" height="30" rx="6" fill="#e6eef8" stroke="#2b6cb0"/><text x="486" y="118" text-anchor="middle">多头自注意力</text>
  <text x="486" y="144" text-anchor="middle" class="mono">+ 残差</text>
  <rect x="426" y="152" width="120" height="30" rx="6" fill="#f2f4f7" stroke="#98a2b3"/><text x="486" y="172" text-anchor="middle">LayerNorm</text>
  <rect x="416" y="190" width="140" height="30" rx="6" fill="#e6eef8" stroke="#2b6cb0"/><text x="486" y="210" text-anchor="middle">MLP（4D，GELU）</text>
  <text x="486" y="234" text-anchor="middle" class="mono">+ 残差</text>
  <line x1="566" y1="140" x2="596" y2="140" stroke="#667085" marker-end="url(#ah2)"/>
  <rect x="600" y="112" width="70" height="56" rx="8" fill="#f2f4f7" stroke="#98a2b3"/>
  <text x="635" y="135" text-anchor="middle">LN</text>
  <text x="635" y="153" text-anchor="middle" class="sm">取 [CLS]</text>
  <line x1="670" y1="140" x2="694" y2="140" stroke="#667085" marker-end="url(#ah2)"/>
  <rect x="698" y="112" width="54" height="56" rx="8" fill="#fde7d0" stroke="#d9822b"/>
  <text x="725" y="135" text-anchor="middle">Linear</text>
  <text x="725" y="153" text-anchor="middle" class="sm">类别</text>
  <text x="380" y="280" text-anchor="middle" class="sm">除了第一步切 patch，其余部分和 BERT 的 encoder 完全一样；CLIP 的图像编码器把最后的 Linear 换成到共享空间的投影</text>
</svg>
<figcaption>ViT 的结构。</figcaption>
</figure>

用公式写，ViT 的前向过程是 [[1]](#ref-1)：

$$
\begin{aligned}
\mathbf{z}_0 &= [\,\mathbf{x}_{\text{cls}};\ \mathbf{x}_p^1 \mathbf{E};\ \mathbf{x}_p^2 \mathbf{E};\ \cdots;\ \mathbf{x}_p^N \mathbf{E}\,] + \mathbf{E}_{\text{pos}}, & \mathbf{E} &\in \mathbb{R}^{(p^2 C) \times D},\ \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(1+N) \times D} \\
\mathbf{z}'_\ell &= \mathrm{MSA}(\mathrm{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, & \ell &= 1, \dots, L \\
\mathbf{z}_\ell &= \mathrm{MLP}(\mathrm{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell, & \ell &= 1, \dots, L \\
\mathbf{y} &= \mathrm{LN}(\mathbf{z}_L^0)
\end{aligned}
$$

其中 $$\mathbf{x}_p^i \in \mathbb{R}^{p^2 C}$$ 是第 $$i$$ 个 patch 展平后的向量，$$\mathbf{z}_L^0$$ 是最后一层 [CLS] 位置的输出，分类头是作用在 $$\mathbf{y}$$ 上的线性层（预训练时论文用一个隐藏层的 MLP，微调时换成单个线性层）。

### 2.2 Patch embedding

"展平 $$p \times p \times C$$ 的块再乘矩阵 $$\mathbf{E}$$"和"核大小、步长都为 $$p$$ 的卷积"是同一个运算：卷积核在每个不重叠的块上做一次点积，输出通道数就是 $$D$$。所有实现（包括 torchvision 和 OpenAI CLIP）都用卷积来写，因为它不需要手动 reshape。

```python
patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
x = patch_embed(img)                 # (B, D, H/p, W/p)
x = x.flatten(2).transpose(1, 2)     # (B, N, D)，按行优先顺序排列 patch
```

> **易错点：patch 的排列顺序。** `flatten(2)` 按行优先展开，第 $$i$$ 个 token 对应第 $$i // (W/p)$$ 行、第 $$i \bmod (W/p)$$ 列的 patch。位置编码、可视化和插值都依赖这个顺序，自己 reshape 时顺序搞错，模型照样能训练（位置编码会学成另一种排列），但加载别人的预训练权重时就对不上了。

$$p$$ 的选择决定了 token 数：$$p$$ 减半，$$N$$ 变成 4 倍，注意力的计算量约变成 16 倍。ViT-B/16 和 ViT-B/32 参数量几乎相同（8660 万和 8820 万），但 B/16 的计算量约是 B/32 的 4.4 倍，准确率也更高。这和 DiT 里"patch 越小越好、代价是计算量"的结论一样。

### 2.3 [CLS] token

[CLS] 是一个可学习的 $$D$$ 维向量，拼在 patch token 前面。它本身不对应图像的任何位置，经过 $$L$$ 层注意力后，它的输出汇总了所有 patch 的信息，用来做分类。这个设计直接来自 BERT。

另一种做法是对所有 patch token 的输出取平均（global average pooling，GAP），这是 CNN 常用的做法。ViT 论文附录比较过，两者效果相当，但需要不同的学习率 [[1]](#ref-1)。后来的不少模型（如 MAE 微调、DINOv2 的部分用法、SigLIP 的注意力池化）用平均或注意力池化代替 [CLS]。CLIP 的 ViT 图像编码器用 [CLS]。

### 2.4 位置编码

自注意力对输入的顺序是置换等变的：把 token 打乱，输出也只是同样地被打乱。不加位置编码，模型就不知道哪个 patch 在哪里，图像就变成了一个"patch 的集合"。

ViT 用的是**可学习的一维位置编码**：一个 $$(1+N) \times D$$ 的参数矩阵，随机初始化，和模型一起训练，直接加到 token 上。论文附录比较了四种方式：不加位置编码、一维可学习、二维可学习（行和列各一套，拼起来）、相对位置编码。不加位置编码明显更差（ImageNet 少样本评估约 61.4% 对 64.2%），而三种位置编码之间差别很小 [[1]](#ref-1)。论文的解释是：ViT 在 patch 的粒度上工作，$$14 \times 14$$ 的网格很小，模型从一维编码里也能学到二维的空间关系。第 4.2 节会在小模型上看到这一点：学到的位置编码里，同一行、同一列的位置相似度更高。

> **对比 DiT：** DiT 用的是**固定的**二维 sin-cos 位置编码（[DiT](/deep-learning/dit/) 第 3.2 节），不训练。生成模型经常要在不同分辨率之间切换，固定编码可以直接按新网格计算；现代 DiT（FLUX 等）进一步换成了二维 RoPE。分类模型的 ViT 和 CLIP 用可学习编码。

**换分辨率时的位置编码插值。** 可学习的位置编码和网格大小绑定：在 224 上训练的 ViT-B/16 有 $$14 \times 14$$ 个位置；在 384 上微调时网格变成 $$24 \times 24$$，原来的编码不够用。ViT 的做法是把 $$14 \times 14$$ 的编码当作一张 $$D$$ 通道的图，用二维插值放大到 $$24 \times 24$$（[CLS] 的编码单独保留）。这是 ViT 里除了切 patch 之外唯一用到二维结构的地方。CLIP ViT-L/14@336 也是这样从 224 的模型继续训练得到的。

```python
def interpolate_pos(pos, new_grid):            # pos: (1, 1+g*g, D)
    cls, grid = pos[:, :1], pos[:, 1:]
    g = int(grid.shape[1] ** 0.5)
    grid = grid.reshape(1, g, g, -1).permute(0, 3, 1, 2)          # 当作 D 通道的图
    grid = F.interpolate(grid, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    return torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)], 1)
```

### 2.5 Transformer block

ViT 的 block 是 **Pre-LN** 结构：LayerNorm 放在注意力和 MLP 之前，残差直接从输入连到输出。原始 Transformer（[Transformer 学习](/deep-learning/transformer-study/) 第 3.6 节）是 Post-LN，LayerNorm 在残差相加之后。Pre-LN 的残差路径上没有归一化，梯度可以直接传到浅层，深层模型更容易训练，不需要很长的学习率 warmup。之后的 GPT-2、CLIP、DiT 都用 Pre-LN。

MLP 是两层线性层，中间维度为 $$4D$$，激活函数为 GELU。注意力是标准的多头缩放点积注意力：

$$
\mathrm{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D/h}}\right)\mathbf{V}
$$

```python
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)          # Q、K、V 的投影合成一个矩阵
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):                # x: (B, N, D)
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """Pre-LN：x + MSA(LN(x))，然后 x + MLP(LN(x))"""
    def __init__(self, dim, heads, mlp_ratio=4.0, act=nn.GELU, eps=1e-6):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads)
        self.ln2 = nn.LayerNorm(dim, eps=eps)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), act(), nn.Linear(hid, dim))

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))
```

`mask` 参数 ViT 用不到，留给第 3 节 CLIP 的文本编码器（causal mask）。

| 公式 | 代码 |
|---|---|
| $$\mathbf{Q}, \mathbf{K}, \mathbf{V} = \mathrm{LN}(\mathbf{z})\,[\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V]$$ | `self.qkv(x)` 后 reshape 成 3 份、$$h$$ 个头 |
| $$\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{D/h})\mathbf{V}$$ | `F.scaled_dot_product_attention(q, k, v)` |
| $$\mathbf{z}' = \mathrm{MSA}(\mathrm{LN}(\mathbf{z})) + \mathbf{z}$$ | `x = x + self.attn(self.ln1(x))` |
| $$\mathbf{z}'' = \mathrm{MLP}(\mathrm{LN}(\mathbf{z}')) + \mathbf{z}'$$ | `x + self.mlp(self.ln2(x))` |

> **易错点：LayerNorm 的 eps。** torchvision 和原始 JAX 实现的 ViT 用 `eps=1e-6`，OpenAI CLIP 用 PyTorch 默认的 `1e-5`。加载预训练权重时 eps 不对，输出会有小的偏差，在深层逐渐放大。本文的 `Block` 把 eps 作为参数，ViT 用 1e-6，CLIP 用 1e-5。

### 2.6 完整模型和验证

```python
class ViT(nn.Module):
    def __init__(self, img=224, patch=16, in_ch=3, dim=768, depth=12, heads=12,
                 mlp_ratio=4.0, n_classes=1000):
        super().__init__()
        self.n = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, self.n + 1, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio) for _ in range(depth))
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)               # (B, N, D)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x)[:, 0])                                # 用 [CLS] 分类
```

完整代码在 [`code/vitclip/vit.py`](https://github.com/chaowang15/deep-learning/tree/main/code/vitclip)。为了确认实现正确，`check_vit.py` 把 torchvision 的 `vit_b_16`（随机初始化）的权重逐层复制到上面的模型里，在同一个随机输入上比较输出：

```
max |ours - torchvision| = 0.0
```

两者完全一致。权重名称的对应关系：`conv_proj` → `patch_embed`，`class_token` → `cls`，`encoder.pos_embedding` → `pos`，`encoder.layers.encoder_layer_i.self_attention.in_proj_weight` → `blocks.i.attn.qkv.weight`，`mlp.0 / mlp.3` → `mlp.0 / mlp.2`，`heads.head` → `head`。

### 2.7 模型配置

| 模型 | $$L$$ | $$D$$ | MLP 维度 | 头数 | 论文参数量 | 本文实现（224 输入，1000 类） |
|---|---|---|---|---|---|---|
| ViT-Ti/16（DeiT） | 12 | 192 | 768 | 3 | — | 5.7M |
| ViT-S/16（DeiT） | 12 | 384 | 1536 | 6 | — | 22.1M |
| ViT-B/16 | 12 | 768 | 3072 | 12 | 86M | 86.6M |
| ViT-L/16 | 24 | 1024 | 4096 | 16 | 307M | 304.3M |
| ViT-H/14 | 32 | 1280 | 5120 | 16 | 632M | 632.0M |

论文表 1 只给了 Base、Large、Huge 三种 [[1]](#ref-1)，Tiny 和 Small 来自 DeiT [[4]](#ref-4)。参数量和论文基本一致；论文没有说明计数时用的分类头和输入分辨率，L 的小差异应来自这里。模型名里的 "/16" 是 patch 大小，同一个模型可以配不同的 patch。

## 3. CLIP

### 3.1 整体结构

<figure class="ddpm-fig">
<svg viewBox="0 0 760 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="CLIP 的结构：图像编码器和文本编码器分别输出向量，归一化后两两点积得到 B×B 相似度矩阵，对角线为正样本">
  <defs><marker id="ah3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#667085"/></marker></defs>
  <text x="20" y="46" class="sm">B 张图</text>
  <g transform="translate(20,56)"><rect x="0" y="0" width="34" height="34" fill="#e6eef8" stroke="#2b6cb0"/><rect x="38" y="0" width="34" height="34" fill="#e6eef8" stroke="#2b6cb0"/><rect x="76" y="0" width="34" height="34" fill="#e6eef8" stroke="#2b6cb0"/></g>
  <line x1="134" y1="73" x2="164" y2="73" stroke="#667085" marker-end="url(#ah3)"/>
  <rect x="168" y="48" width="130" height="50" rx="8" fill="#e6eef8" stroke="#2b6cb0"/>
  <text x="233" y="70" text-anchor="middle">图像编码器</text><text x="233" y="88" text-anchor="middle" class="sm">ViT，取 [CLS] → 投影</text>
  <line x1="298" y1="73" x2="328" y2="73" stroke="#667085" marker-end="url(#ah3)"/>
  <text x="334" y="68" class="mono">u = norm(·)</text><text x="334" y="84" class="mono">B×E</text>
  <text x="20" y="196" class="sm">B 段文字</text>
  <g transform="translate(20,206)"><rect x="0" y="0" width="110" height="20" rx="4" fill="#fde7d0" stroke="#d9822b"/><text x="55" y="14" text-anchor="middle" class="sm">"a dog on grass"</text><rect x="0" y="24" width="110" height="20" rx="4" fill="#fde7d0" stroke="#d9822b"/><text x="55" y="38" text-anchor="middle" class="sm">"a red car"</text></g>
  <line x1="134" y1="226" x2="164" y2="226" stroke="#667085" marker-end="url(#ah3)"/>
  <rect x="168" y="201" width="130" height="50" rx="8" fill="#fde7d0" stroke="#d9822b"/>
  <text x="233" y="223" text-anchor="middle">文本编码器</text><text x="233" y="241" text-anchor="middle" class="sm">causal mask，取 [EOT]</text>
  <line x1="298" y1="226" x2="328" y2="226" stroke="#667085" marker-end="url(#ah3)"/>
  <text x="334" y="221" class="mono">v = norm(·)</text><text x="334" y="237" class="mono">B×E</text>
  <g transform="translate(470,60)">
    <text x="90" y="-14" text-anchor="middle" class="sm">logits = s · u vᵀ（B×B）</text>
    <g stroke="#d0d5dd" fill="#f8fafc">
      <rect x="0" y="0" width="44" height="44"/><rect x="46" y="0" width="44" height="44"/><rect x="92" y="0" width="44" height="44"/><rect x="138" y="0" width="44" height="44"/>
      <rect x="0" y="46" width="44" height="44"/><rect x="46" y="46" width="44" height="44"/><rect x="92" y="46" width="44" height="44"/><rect x="138" y="46" width="44" height="44"/>
      <rect x="0" y="92" width="44" height="44"/><rect x="46" y="92" width="44" height="44"/><rect x="92" y="92" width="44" height="44"/><rect x="138" y="92" width="44" height="44"/>
      <rect x="0" y="138" width="44" height="44"/><rect x="46" y="138" width="44" height="44"/><rect x="92" y="138" width="44" height="44"/><rect x="138" y="138" width="44" height="44"/>
    </g>
    <g fill="#2b6cb0"><rect x="0" y="0" width="44" height="44"/><rect x="46" y="46" width="44" height="44"/><rect x="92" y="92" width="44" height="44"/><rect x="138" y="138" width="44" height="44"/></g>
    <g class="mono"><text x="22" y="27" text-anchor="middle" style="fill:#fff">u₁·v₁</text><text x="68" y="27" text-anchor="middle">u₁·v₂</text><text x="68" y="73" text-anchor="middle" style="fill:#fff">u₂·v₂</text><text x="114" y="119" text-anchor="middle" style="fill:#fff">u₃·v₃</text><text x="160" y="165" text-anchor="middle" style="fill:#fff">u₄·v₄</text></g>
    <text x="192" y="26" class="sm">← 每行：图 i 在</text><text x="192" y="42" class="sm">B 段文字里选 i</text>
    <text x="0" y="206" class="sm">↑ 每列：文字 j 在 B 张图里选出 j</text>
    <text x="0" y="224" class="sm">对角线是正样本，其余 B²−B 个都是负样本</text>
  </g>
  <line x1="420" y1="78" x2="462" y2="110" stroke="#667085" marker-end="url(#ah3)"/>
  <line x1="420" y1="226" x2="462" y2="190" stroke="#667085" marker-end="url(#ah3)"/>
  <text x="380" y="314" text-anchor="middle" class="sm">s = exp(logit_scale) = 1/τ 是可学习的标量；损失 = (行方向交叉熵 + 列方向交叉熵) / 2</text>
</svg>
<figcaption>CLIP 的训练：两个编码器各自把图和文映射到同一个 E 维空间，配对的图文（对角线）相似度要高于同一个 batch 里所有不配对的组合。</figcaption>
</figure>

CLIP 是一个**双塔**（dual encoder）模型：图像和文字各有一个编码器，互不交互，只在最后通过点积比较。这和后来的多模态大模型（图像 token 和文本 token 在同一个 Transformer 里交互）不同。双塔的好处是可以分别预先算好所有图像和所有文字的向量，检索时只需要做一次矩阵乘法。

### 3.2 图像编码器

CLIP 训练了 5 个 ResNet 和 3 个 ViT 版本 [[2]](#ref-2)，效果最好、后来被广泛使用的是 ViT 版本。它和第 2 节的 ViT 有四处小差别（对照官方代码 [[3]](#ref-3)）：

1. patch embedding 的卷积**没有 bias**；
2. 加位置编码之后、进入 Transformer 之前，多一个 LayerNorm（`ln_pre`）；
3. 激活函数用 **QuickGELU**：$$x \cdot \sigma(1.702x)$$，是 GELU 的近似；
4. 最后取 [CLS] 的输出，过 `ln_post`，再乘一个 $$D \times E$$ 的投影矩阵（没有 bias）映射到共享空间，代替分类头。

```python
class ImageEncoder(nn.Module):
    def __init__(self, img=224, patch=32, in_ch=3, dim=768, depth=12, heads=12, embed=512):
        super().__init__()
        n = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, stride=patch, bias=False)
        self.cls = nn.Parameter(torch.randn(dim) * dim ** -0.5)
        self.pos = nn.Parameter(torch.randn(n + 1, dim) * dim ** -0.5)
        self.ln_pre = nn.LayerNorm(dim)
        self.tower = Tower(dim, depth, heads)            # L 个 Block，QuickGELU，eps=1e-5
        self.ln_post = nn.LayerNorm(dim)
        self.proj = nn.Parameter(torch.randn(dim, embed) * dim ** -0.5)

    def forward(self, x):                                # (B, C, H, W) -> (B, E)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), 1, -1), x], 1) + self.pos
        x = self.tower(self.ln_pre(x))
        return self.ln_post(x[:, 0]) @ self.proj
```

> **易错点：IP-Adapter 和 LLaVA 用的不是同一个输出。** 上面的 `forward` 返回的是投影后的全局向量（$$E$$ 维），这是 CLIP 训练时用来做对比的。IP-Adapter（基础版）用的就是这个全局向量；IP-Adapter Plus 和 LLaVA 用的是倒数第二层**所有 patch token** 的输出（$$N \times D$$，没有经过 `ln_post` 和投影），因为它们需要空间细节。

### 3.3 文本编码器

文本编码器是一个 12 层、宽度 512（ViT-L/14 版本为 768）、8 个头的 Transformer，结构和 GPT-2 类似 [[2]](#ref-2)：

- **分词**：小写化的字节级 BPE，词表 49408 个（包括起始符 `[SOT]` 和结束符 `[EOT]`），序列最长 77 个 token。超出截断，不足补 0。`[EOT]` 是词表中下标最大的 token（49407）。
- **causal mask**：每个 token 只能看到自己和前面的 token。论文说保留它是为了以后可以加语言建模的目标；它带来的结果是，只有最后一个有效 token 看到了整句话。
- **取 [EOT] 位置的输出**：经过最后的 LayerNorm（`ln_final`）后，取 `[EOT]` 位置的向量（因为 causal mask，只有它看到了全句），乘投影矩阵得到 $$E$$ 维的文本向量。

```python
class TextEncoder(nn.Module):
    def __init__(self, vocab=49408, ctx=77, dim=512, depth=12, heads=8, embed=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(ctx, dim) * 0.01)
        self.tower = Tower(dim, depth, heads)
        self.ln_final = nn.LayerNorm(dim)
        self.proj = nn.Parameter(torch.randn(dim, embed) * dim ** -0.5)
        self.register_buffer("mask", torch.ones(ctx, ctx, dtype=torch.bool).tril(), persistent=False)

    def forward(self, ids):                              # (B, L) int -> (B, E)
        L = ids.shape[1]
        x = self.tok(ids) + self.pos[:L]
        x = self.ln_final(self.tower(x, self.mask[:L, :L]))
        eot = ids.argmax(-1)                             # [EOT] 的下标最大
        return x[torch.arange(len(x)), eot] @ self.proj
```

> **易错点：`ids.argmax(-1)` 找 [EOT] 的前提。** 官方代码用 `argmax` 找 `[EOT]` 的位置，依赖"`[EOT]` 是词表中下标最大的 token"。自己换分词器（比如第 4.3 节的词级分词器）时，必须把 `[EOT]` 放在词表最后，否则取到的是别的 token 的输出，模型照样能训练，但效果变差，而且很难发现。

> **易错点：Stable Diffusion 用的是逐 token 输出，不是这里的 $$E$$ 维向量。** SD 1.x 取的是 `ln_final` 之后全部 77 个位置的输出（$$77 \times 768$$），送给 U-Net 的 cross-attention（[Stable Diffusion 拆解](/deep-learning/stable-diffusion/) 第 2.3 节）；SD 2.x 用的是 OpenCLIP ViT-H 倒数第二层的输出。上面 `forward` 的最后一行（取 `[EOT]`、乘投影）只在对比训练和 SDXL 的 "pooled" 文本向量里用到。

### 3.4 对称对比损失和可学习温度

一个 batch 有 $$B$$ 对图文。把两个编码器的输出归一化为单位向量 $$\mathbf{u}_i, \mathbf{v}_j$$，相似度矩阵为

$$
\ell_{ij} = s \cdot \mathbf{u}_i^\top \mathbf{v}_j, \qquad s = 1/\tau
$$

损失把每一行和每一列各当作一个 $$B$$ 分类问题 [[2]](#ref-2)：

$$
\mathcal{L} = \frac{1}{2}\left[\frac{1}{B}\sum_{i=1}^{B} -\log \frac{e^{\ell_{ii}}}{\sum_{j} e^{\ell_{ij}}} \;+\; \frac{1}{B}\sum_{j=1}^{B} -\log \frac{e^{\ell_{jj}}}{\sum_{i} e^{\ell_{ij}}}\right]
$$

第一项是"图找文"：第 $$i$$ 张图要在 $$B$$ 段文字中认出第 $$i$$ 段；第二项是"文找图"。这种损失叫 InfoNCE（或 multi-class N-pair loss）。

```python
class CLIP(nn.Module):
    def __init__(self, image, text, init_temp=0.07):
        super().__init__()
        self.image, self.text = image, text
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temp)))

    def forward(self, imgs, ids):
        u = F.normalize(self.image(imgs), dim=-1)            # (B, E)
        v = F.normalize(self.text(ids), dim=-1)              # (B, E)
        s = self.logit_scale.exp().clamp(max=100)
        return s * u @ v.t()                                 # (B, B)


def clip_loss(logits):
    y = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y)) / 2
```

| 公式 | 代码 |
|---|---|
| $$\mathbf{u}_i = f_{\text{img}}(x_i) / \lVert f_{\text{img}}(x_i) \rVert$$ | `F.normalize(self.image(imgs), dim=-1)` |
| $$s = 1/\tau = e^{\text{logit\_scale}}$$，上限 100 | `self.logit_scale.exp().clamp(max=100)` |
| $$\ell_{ij} = s\, \mathbf{u}_i^\top \mathbf{v}_j$$ | `s * u @ v.t()` |
| 行方向交叉熵（图找文） | `F.cross_entropy(logits, y)` |
| 列方向交叉熵（文找图） | `F.cross_entropy(logits.t(), y)` |

**温度为什么要学、为什么要设上限。** 归一化之后点积只在 $$[-1, 1]$$ 之间。如果 $$s = 1$$，logit 的差别最多是 2，softmax 几乎是均匀分布，损失下不去（背景一节的数值例子里损失是 0.95 而不是 0.08）。$$s$$ 越大，softmax 越"尖"，模型越关注最难区分的负样本。CLIP 把 $$\log s$$ 作为可学习参数，初始 $$\tau = 0.07$$（$$s \approx 14.3$$），训练中 $$s$$ 会一路增大；为了防止训练不稳定，$$s$$ 被限制在 100 以内 [[2]](#ref-2)。官方的 CLIP 模型训练结束时 $$s$$ 都到了 100。参数化成 $$\log s$$ 而不是 $$s$$，保证了 $$s$$ 始终为正，并且乘性的变化在对数空间里是加性的，更容易优化。

**batch 大小为什么重要。** 每张图的负样本数是 $$B-1$$。负样本越多，任务越难，学到的表示越细。CLIP 用了 $$B = 32768$$：相似度矩阵有 10 亿个元素，需要把各个 GPU 上的向量收集到一起（all-gather）再算。这是 CLIP 训练代价高的主要原因之一，也是 SigLIP 要解决的问题（第 5 节）。

> **易错点：batch 内的"假负样本"。** 损失假设一个 batch 里只有对角线是配对的。但如果 batch 里有两张都是"一只狗"的图，配文也差不多，它们之间也会被当作负样本推开。网络数据规模大、内容杂，这种情况比例不高；在类别很少的数据上（比如第 4.3 节的 FashionMNIST 只有 10 类），一个 batch 里大量样本是同类的，损失有一个明显的下限，第 4.3 节会算出这个下限。

### 3.5 参数量验证

| 模型 | 图像编码器 | 文本编码器 | $$E$$ | 本文实现：总计 / 图像 / 文本 |
|---|---|---|---|---|
| ViT-B/32 | $$L=12, D=768, p=32$$ | $$L=12, D=512$$，8 头 | 512 | 151.3M / 87.8M / 63.4M |
| ViT-B/16 | $$L=12, D=768, p=16$$ | 同上 | 512 | 149.6M / 86.2M / 63.4M |
| ViT-L/14 | $$L=24, D=1024, p=14$$ | $$L=12, D=768$$，12 头 | 768 | 427.6M / 304.0M / 123.7M |
| ViT-L/14@336 | 同上，输入 336 | 同上 | 768 | 427.9M / 304.3M / 123.7M |

这与官方 `openai/CLIP` 发布的模型参数量一致（ViT-B/32 约 151.3M，ViT-L/14 约 427.6M）。其中 ViT-L/14 的文本编码器约 1.2 亿参数，就是 SD 1.x 里冻结使用的那个文本编码器。另外一个检查：随机初始化时所有 logit 几乎相等，损失应该约等于 $$\log B$$；$$B = 256$$ 时实测 5.545，$$\log 256 = 5.545$$。

### 3.6 零样本分类和提示工程

零样本分类就是把类别名变成文本向量，用它们当分类器的权重：

```python
@torch.no_grad()
def zero_shot_classifier(model, classnames, templates):
    W = []
    for c in classnames:
        texts = [t.format(c) for t in templates]           # 例如 "a photo of a {}."
        v = F.normalize(model.text(tokenize(texts)), dim=-1)
        W.append(F.normalize(v.mean(0), dim=-1))           # 多个提示的平均（提示集成）
    return torch.stack(W)                                  # (类别数, E)

# 预测：logits = F.normalize(model.image(x), dim=-1) @ W.t()，取 argmax
```

CLIP 论文发现提示的写法影响很大 [[2]](#ref-2)：

- **只用类别名效果差**，因为训练时的文字很少是单个词，而且一词多义（"crane" 可以是鹤也可以是起重机）。用 "A photo of a {label}." 这样的模板，ImageNet 准确率就提高 1.3%。
- **针对数据集加上下文**：例如宠物数据集用 "A photo of a {label}, a type of pet."。
- **提示集成**：对同一个类别用多个模板（ImageNet 用了 80 个，如 "a bad photo of a {}."、"a sculpture of a {}."），把文本向量平均后再归一化。在 ImageNet 上比单个默认提示再提高 3.5%。提示工程和集成合起来提高近 5%。

集成在向量空间里做（平均文本向量），而不是在概率上做，所以推理时的代价和单个提示一样：每个类别只有一个向量。

**零样本和线性探针。** 另一种评估方式是冻结图像编码器，用标注数据在它的输出上训练一个线性分类器（linear probe）。CLIP 论文发现，零样本 CLIP 的效果大约相当于每类用 4 个样本训练的线性探针；有足够多的标注数据时，线性探针更好。第 4.3 节的迷你 CLIP 同时报告这两个数。

## 4. 实验：FashionMNIST

三个小实验，都在 4 核 CPU 上运行，代码在 [`code/vitclip/`](https://github.com/chaowang15/deep-learning/tree/main/code/vitclip)。

### 4.1 ViT 和 CNN：数据量的影响

同样的训练设置下，比较一个小 CNN 和一个小 ViT 在不同训练数据量下的测试准确率（`exp_data_size.py`）：

- **CNN**：两个 $$3 \times 3$$ 卷积层（32、64 通道，各接 2 倍池化）加两层全连接，42 万参数，每张图约 400 万次乘加。
- **ViT**：$$p = 4$$（$$7 \times 7 = 49$$ 个 patch token 加 [CLS]），$$D = 64$$，$$L = 6$$，4 个头，31 万参数，每张图约 1500 万次乘加。
- **训练**：从 6 万张训练图中随机取 1k、6k、60k 张，两个模型都训练 2000 步，batch 128，AdamW（学习率 $$10^{-3}$$，权重衰减 0.05，100 步 warmup 后余弦衰减），数据增强为随机平移 2 像素和水平翻转。在完整的 1 万张测试集上评估。

![数据量和准确率](/deep-learning/assets/vit-clip/data_size.png)

| 训练图像数 | CNN 训练 / 测试准确率 | ViT 训练 / 测试准确率 | 差距（测试） |
|---|---|---|---|
| 1k | 99.2% / 83.9% | 90.7% / 79.1% | 4.8 |
| 6k | 92.2% / 87.9% | 84.2% / 82.7% | 5.2 |
| 60k | 89.8% / 89.1% | 83.6% / 83.5% | 5.5 |

（训练准确率在训练子集的前 5000 张上测，不加数据增强。）

**结果：在所有数据量下 CNN 都更好，差距也没有随数据量缩小（约 5 个百分点）。** 这个实验没有复现 ViT 论文里"数据多了 ViT 追上并超过 CNN"的趋势，原因也很清楚：

- **60k 张图离 ViT 的优势区间还差几个数量级。** ViT 论文里两者相当是在 ImageNet-21k（1400 万张）上，ViT 超过 ResNet 是在 JFT-300M（3 亿张）上。1k 到 60k 全部属于"数据少"的区间，在这个区间里卷积先验一直有用。
- **ViT 不仅泛化差，而且没有训练充分。** 1k 张时，CNN 的训练准确率已经到 99.2%（几乎背下了训练集），测试仍有 83.9%；ViT 训练准确率只有 90.7%。6k、60k 时 ViT 的训练准确率（84%）也明显低于 CNN（90%–92%）。也就是说，同样 2000 步，ViT 连训练集都拟合得更慢：没有局部性先验，它需要先从数据中学会"相邻像素相关"，这本身就要花训练量。这和 [DiT](/deep-learning/dit/) 第 7.1 节看到的现象一样：Transformer 起步慢，需要更多的训练。
- **ViT 每步更贵。** 在 CPU 上 ViT 每次运行约 250 秒，CNN 约 50 秒。

所以这个实验能说明的是"数据少、训练短时 CNN 的先验是实打实的优势"，而不是 ViT 论文的主结论。要在 CPU 规模上看到 ViT 追上 CNN 是做不到的；论文的结论需要千万张以上的预训练数据。

### 4.2 位置编码学到了什么

取第 4.1 节在 60k 张图上训练的 ViT，计算 49 个 patch 位置编码两两之间的余弦相似度。下图第 $$r$$ 行第 $$c$$ 列的小图，画的是位置 $$(r, c)$$ 的编码和所有 49 个位置的相似度（亮表示相似）：

![位置编码的相似度](/deep-learning/assets/vit-clip/pos_embed_sim.png)

位置编码是随机初始化、按一维序列加上去的，模型并不知道"第 8 个 token 在第 1 个 token 的正下方"。但训练之后，按二维位置分组统计相似度的平均值：

| 两个位置的关系 | 平均余弦相似度 |
|---|---|
| 上下或左右相邻 | 0.51 |
| 同一行、左右镜像（第 $$c$$ 列和第 $$6-c$$ 列） | 0.59 |
| 同一行（其他） | 0.21 |
| 同一列（其他） | 0.25 |
| 不同行也不同列 | −0.08 |

- **学到了二维结构。** 相邻位置的编码最相似，同一行、同一列次之，其余位置接近正交。这和 ViT 论文附录的可视化一致 [[1]](#ref-1)：一维可学习编码自己学出了行列结构，这也是一维和二维编码效果差不多的原因。
- **学到了左右对称。** 镜像位置的相似度甚至高于相邻位置。这来自训练时的**水平翻转**数据增强：翻转后，第 $$c$$ 列的内容出现在第 $$6-c$$ 列，对分类来说这两个位置可以互换，模型就让它们的编码变得相似。位置编码学到的不只是几何位置，还包括数据（和数据增强）里的对称性。

### 4.3 一个迷你 CLIP

真正的 CLIP 需要上亿对图文，这里用 FashionMNIST 加**合成的描述**做一个能在 CPU 上几分钟训练完的版本（`exp_clip.py`），目的是把第 3 节的每个部件真正跑一遍。

**数据。** 每张训练图每次被采样时，随机生成一句描述：

- 从 6 个模板中随机选一个："a photo of a {}"、"a picture of a {}"、"an image of a {}"、"a {}"、"this is a {}"、"a product photo of a {}"；
- 类别名从同义词中随机选，比如 trousers / pants / jeans，sneaker / trainer / running shoe；
- 一半的描述前面加一个形容词 "dark" 或 "light"：衣物像素的平均亮度低于该类的中位数就是 dark，否则是 light。这让描述包含了类别之外的信息。

例如同一张裤子的图，这一次的描述是 "a picture of a dark jeans"，下一次可能是 "this is a pants"。**训练过程不使用类别标签**，只有图文配对。

**模型。** 图像塔是第 3.2 节的 `ImageEncoder`（$$p = 4$$，$$D = 64$$，$$L = 4$$，4 个头）；文本塔是第 3.3 节的 `TextEncoder`（词级分词器，44 个词，最长 12 个 token，$$D = 64$$，$$L = 2$$，带 causal mask，`[EOT]` 放在词表最后）；共享空间 $$E = 64$$，共 32 万参数。batch 256，训练 1500 步，AdamW（学习率 $$10^{-3}$$，余弦衰减），初始 $$\tau = 0.07$$。4 核 CPU 上约 4 分钟。

**训练曲线和温度：**

![训练损失和温度](/deep-learning/assets/vit-clip/clip_curves.png)

**零样本分类**（1 万张测试图，不使用任何标注训练分类器）：

| 方法 | 测试准确率 |
|---|---|
| 单个提示 "a photo of a {类别名}" | 88.7% |
| 提示集成（6 个模板 × 每类所有同义词，平均文本向量） | 88.6% |
| 训练中没出现过的模板 "a photo of the {类别名}" | 88.5% |
| 线性探针（冻结图像塔，用 1 万张有标注的训练图训练线性分类器） | 88.7% |

每类一张测试图，对 10 个类别提示的 softmax 概率：

![零样本分类的概率矩阵](/deep-learning/assets/vit-clip/clip_zeroshot_matrix.png)

**文本检索图像**（对 1 万张测试图按相似度排序，每行取前 8 张）：

![文本检索图像](/deep-learning/assets/vit-clip/clip_retrieval.png)

结果和分析：

- **只靠图文配对就学会了分类。** 零样本准确率 88.7%，和用标注训练的线性探针（88.7%）几乎一样，也接近第 4.1 节有监督训练的 CNN（60k 张图，89.1%）。CLIP 论文中零样本效果相当于每类 4 个样本的线性探针，这里两者持平，是因为这个任务里描述携带的类别信息和标签完全一样，而真实的网络图文噪声大得多。混淆主要出现在 shirt、t-shirt、pullover、coat 之间（shirt 只有 63% 被分对，14% 被分成 t-shirt），三种鞋之间也有少量混淆，这和有监督分类器在 FashionMNIST 上的常见错误一致。
- **学到了类别之外的属性。** 在每个类别内用 "a photo of a dark X" 和 "a photo of a light X" 二选一，判断图像属于较暗还是较亮的一半，平均准确率 94.4%（凉鞋和运动鞋最低，约 88%–90%，它们的像素少，亮度估计更不稳定）。检索结果中 "dark bag" 和 "light bag" 返回的图明显不同。这是用文字监督相对于类别标签的好处：描述里有什么，模型就学什么。
- **提示集成在这里没有提升。** 训练用的模板和同义词本身就很少，每个提示得到的类别向量已经几乎一样，平均之后没有新信息。真实 CLIP 的提示集成有效，是因为训练文本的写法千变万化，单个提示的向量带有偶然的偏差，平均能抵消掉。没见过的模板 "a photo of the ..." 准确率几乎不变，说明文本塔没有死记整句话，主要依赖类别词。
- **损失停在约 3.2，远高于 0。** 这就是第 3.4 节说的"假负样本"：10 个类别、batch 256，每张图在 batch 里平均有 $$k = 26.5$$ 张同类图（含自己）。同类图的描述只有模板、同义词和亮度词的区别，而模板和同义词不携带任何图像信息，模型不可能分辨"这句 a picture of a pants 配的是这 26 张裤子图里的哪一张"。假设模型能完美分辨类别和亮度、其余都无法分辨，可以估算损失下限：图找文时约有 $$0.75k$$ 段描述同样合适（一半不带形容词的描述，加上四分之一带对了形容词的），文找图时不带形容词的描述对应 $$k$$ 张图、带形容词的对应 $$k/2$$ 张，下限约为 $$\tfrac{1}{2}\log(0.75k) + \tfrac{1}{4}[\log k + \log(k/2)] \approx 2.96$$。实际的 3.2 离这个下限不远。
- **温度几乎没有变化**（$$1/\tau$$ 从 14.3 变到 14.9），而真实 CLIP 的 $$1/\tau$$ 会一路升到 100。原因同上：同类样本无法区分时，再把 softmax 变尖也降不了损失，反而会让同类之间的错误更"自信"，所以温度停在中间。真实数据里几乎每段描述都是唯一的，正确配对可以和所有负样本区分开，把 $$1/\tau$$ 调大才能继续降低损失。这个对比说明，温度学到多大，取决于数据中正负样本能被区分到什么程度。

> **易错点：不要对 logit_scale 做权重衰减。** 和 OpenCLIP 一样，本实验的 `logit_scale`、偏置和 LayerNorm 参数不加权重衰减。如果对所有参数加 0.05 的权重衰减，`logit_scale` 会被往 0 拉（$$1/\tau$$ 被往 1 拉），在这个实验里最终 $$1/\tau$$ 为 14.0（不加衰减时是 14.9），零样本准确率差别不大（88.6% 对 88.7%）。在真实规模的训练中，温度需要一路升到 100，对它做衰减的影响会大得多。

### 4.4 这组实验能说明什么、不能说明什么

| 实验 | 能说明 | 不能说明 |
|---|---|---|
| ViT 和 CNN 的数据量对比 | 数据少、训练短时，卷积先验有实打实的优势；ViT 连训练集都拟合得更慢 | ViT 在大数据上超过 CNN。这需要千万到上亿张图，CPU 规模做不到 |
| 位置编码可视化 | 一维可学习编码能自己学出二维结构，还会学到数据增强带来的对称性 | 一维和二维编码哪个更好（需要在更大的数据上做消融） |
| 迷你 CLIP | 对称对比损失、双塔、[EOT] 取特征、提示分类都能正常工作；文字监督能学到类别之外的属性；batch 内同类样本造成损失下限，并让温度停在中间 | 真实 CLIP 的零样本能力和鲁棒性。那来自 4 亿对多样的图文，这里的描述是用 10 个类别合成的，"零样本"的类别名在训练描述里出现过 |

最后一点值得强调：这里的"零样本"是指**没有训练分类器、没有使用标签**，而不是"没见过这个类别"。真实 CLIP 的零样本也是同样的意思：ImageNet 的类别名在网络文本里大多出现过，CLIP 的能力来自训练数据覆盖了足够多的概念。

## 5. 后续发展

| 模型 | 年份 | 相对 CLIP 的主要变化 |
|---|---|---|
| ALIGN | 2021 | 18 亿对噪声更大、几乎不过滤的图文对；EfficientNet + BERT。说明数据规模可以弥补噪声 |
| OpenCLIP [[5]](#ref-5) | 2022–2023 | 在公开的 LAION-400M / LAION-2B 上复现 CLIP 并放大。ViT-H/14 零样本 ImageNet 78.0%，ViT-bigG/14 80.1%；发现零样本效果随模型、数据、计算量按幂律提升 |
| CoCa | 2022 | 在对比损失之外加一个生成描述的解码器，两个目标一起训练 |
| EVA-CLIP | 2023 | 用 MIM 预训练的 ViT 初始化图像编码器，加上训练技巧，大幅降低训练代价 |
| SigLIP [[6]](#ref-6) | 2023 | 损失从 softmax 换成逐对的 sigmoid，见下文 |
| MetaCLIP、DFN | 2023 | 主要改进数据：按元数据平衡类别分布、用过滤网络筛选高质量图文对 |

**SigLIP 的 sigmoid 损失。** softmax 对比损失要对整行、整列做归一化，必须先把整个 batch 的相似度矩阵算出来。SigLIP 把每一对 $$(i, j)$$ 当作独立的二分类问题："这对图文是不是配对的" [[6]](#ref-6)：

$$
\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B} \log \sigma\big(z_{ij}\,(s\, \mathbf{u}_i^\top \mathbf{v}_j + b)\big), \qquad z_{ij} = \begin{cases} 1 & i = j \\ -1 & i \ne j \end{cases}
$$

$$b$$ 是一个可学习的偏置，初始化为 $$-10$$，因为负样本远多于正样本，开始时要让模型倾向于"不配对"。每一项只和一对样本有关，可以在设备之间分块计算，不需要 all-gather 整个矩阵；在小 batch 下效果明显好于 softmax 损失，batch 超过约 3.2 万后两者差别不大。SigLIP 的图像编码器后来被 PaliGemma 等多模态模型使用。

**CLIP 在生成模型里的具体用法**（补充背景一节的表格）：

- **SD 1.x**：OpenAI CLIP ViT-L/14 的文本编码器，取最后一层 $$77 \times 768$$ 的逐 token 输出。
- **SD 2.x**：换成 OpenCLIP ViT-H/14 的文本编码器，取倒数第二层输出（$$77 \times 1024$$）。
- **SDXL**：CLIP ViT-L 和 OpenCLIP ViT-bigG 两个文本编码器的倒数第二层输出在通道维拼接（$$768 + 1280 = 2048$$），bigG 的 pooled 向量额外加到时间嵌入上。
- **SD3、FLUX**：CLIP 的 pooled 向量作为全局条件（通过 adaLN，类似 [DiT](/deep-learning/dit/) 的类别嵌入），逐 token 的文本主要来自 T5-XXL 这样的纯语言模型，因为 CLIP 的文本编码器在长 prompt 和复杂组合关系上较弱。
- **DALL·E 2（unCLIP）**：先用一个先验模型从 CLIP 文本向量生成 CLIP 图像向量，再用扩散解码器从图像向量生成图像。
- **IP-Adapter**：CLIP 图像向量投影成几个 token，加一路 cross-attention（[IP-Adapter](/deep-learning/ip-adapter/)）。
- **LLaVA**：CLIP ViT-L/14@336 倒数第二层的 576 个 patch token，经 MLP 投影后作为语言模型的输入 token。

## 6. 总结

| | ViT | CLIP |
|---|---|---|
| 要解决的问题 | 视觉能不能像 NLP 一样用标准 Transformer，并随规模变好 | 摆脱固定类别和人工标注，用网络图文直接监督视觉 |
| 输入 | 图像 | 图文对 |
| 结构 | patch embedding + [CLS] + 可学习位置编码 + $$L$$ 个 Pre-LN block | 图像塔（ViT）+ 文本塔（causal Transformer），各自投影到 $$E$$ 维 |
| 训练目标 | 有监督分类（交叉熵） | 对称 InfoNCE，可学习温度 $$\tau$$，$$1/\tau \le 100$$ |
| 关键超参数 | $$p$$（决定 token 数和计算量）、$$L$$、$$D$$ | batch 大小（负样本数）、温度、数据规模 |
| 结论 | 数据少时不如 CNN，数据多时超过 CNN | 零样本 ImageNet 76.2%，分布偏移下更鲁棒 |
| 主要局限 | 需要大数据或强数据增强；单一分辨率 | 细粒度、计数、组合关系弱；batch 和算力要求高 |
| 在生成模型中 | DiT 的主干结构 | SD 的文本编码器、IP-Adapter 的图像编码器、CLIP score |

## 7. 自检

1. **ViT-B/16 和 ViT-B/32 的参数量几乎一样，为什么 B/16 计算量大得多？**
   <br>参数量主要在 Transformer block 里，和 token 数无关；patch 从 32 变成 16，token 数从 49 变成 196，每层线性层的计算量变成 4 倍，注意力变成约 16 倍。只有 patch embedding 和位置编码的参数量随 $$p$$ 变化。

2. **不加位置编码，ViT 还能工作吗？**
   <br>能训练，但图像变成了"patch 的集合"，模型不知道哪个 patch 在哪里。论文的消融里准确率明显下降；而一维、二维、相对位置编码之间差别很小。

3. **在 224 上训练的 ViT 要在 384 上微调，位置编码怎么办？**
   <br>把 patch 部分的 $$14 \times 14$$ 个位置编码当作一张 $$D$$ 通道的图，双三次插值到 $$24 \times 24$$，[CLS] 的编码保持不变。

4. **为什么 ViT 在 ImageNet-1k 上不如 ResNet，在 JFT-300M 上却超过它？**
   <br>CNN 的局部性和平移等变性是写在结构里的先验，数据少时很有用；ViT 要从数据中学到这些规律，需要更多数据。数据足够多时，先验的好处消失，而 Transformer 学到的模式更灵活。DeiT 说明强数据增强和蒸馏可以部分弥补数据量。

5. **CLIP 的损失为什么是"对称"的？只用一个方向可以吗？**
   <br>行方向让每张图在 $$B$$ 段文字中找到配对的那段（零样本分类就是这个方向），列方向让每段文字在 $$B$$ 张图中找到配对的那张（以文搜图是这个方向）。只用一个方向也能训练，但另一个方向的检索没有被直接优化；两个方向取平均，两种用法都被训练到。

6. **温度 $$\tau$$ 太大或太小会怎样？为什么要设 $$1/\tau \le 100$$？**
   <br>$$\tau$$ 太大（$$1/\tau$$ 小），余弦相似度的差别被压缩，softmax 接近均匀，损失下不去；$$\tau$$ 太小，softmax 只看最难的负样本，梯度很尖，训练不稳定。CLIP 让它可学习，并把 $$1/\tau$$ 限制在 100 以内防止不稳定。

7. **CLIP 的文本编码器为什么取 [EOT] 位置的输出？**
   <br>文本编码器有 causal mask，只有最后一个有效 token 看到了整句话。官方代码用 `ids.argmax(-1)` 找它，前提是 [EOT] 的下标在词表中最大。

8. **Stable Diffusion 用 CLIP 文本编码器的哪个输出？和对比训练时用的有什么不同？**
   <br>SD 1.x 用 `ln_final` 之后全部 77 个位置的逐 token 输出（$$77 \times 768$$），给 cross-attention 用；对比训练只用 [EOT] 位置的一个向量再乘投影矩阵。SDXL、SD3、FLUX 另外用到这个 pooled 向量作为全局条件。

9. **零样本分类时，提示集成是在哪里做平均的？代价是多少？**
   <br>对同一个类别的多个提示的文本向量取平均再归一化，每个类别仍然只有一个向量。推理时的代价和单个提示完全一样，只是预先多算几次文本编码。

10. **SigLIP 的 sigmoid 损失相对 softmax 损失的主要好处是什么？偏置 $$b$$ 为什么初始化成 $$-10$$？**
    <br>每对图文的损失独立，不需要整行整列归一化，可以分块计算，不必 all-gather 整个 $$B \times B$$ 矩阵，小 batch 下效果更好。每张图有 1 个正样本、$$B-1$$ 个负样本，$$b = -10$$ 让模型开始时预测"不配对"，避免初期被大量负样本的损失主导。

## 8. 要点回顾

- **ViT 把图像切成 $$p \times p$$ 的 patch 当作 token**，加 [CLS] 和可学习位置编码，用标准 Pre-LN Transformer encoder 分类；patch embedding 就是核大小、步长为 $$p$$ 的卷积。
- **ViT 几乎没有图像的归纳偏置**：数据少时不如 CNN，数据多（JFT-300M）时超过 CNN；DeiT 用强数据增强和蒸馏在 ImageNet-1k 上也把它训好了。 FashionMNIST 上从 1k 到 60k 张图，CNN 都比同规模的 ViT 高约 5 个百分点，ViT 连训练集都拟合得更慢。
- **位置编码是一维、可学习的**，但学出来的结构是二维的；换分辨率时对它做二维插值。
- **CLIP 是双塔模型**：ViT 图像编码器取 [CLS]，causal Transformer 文本编码器取 [EOT]，各自投影、归一化到同一个空间，用对称 InfoNCE 训练。
- **温度可学习**：$$\log(1/\tau)$$ 是参数，初始 $$\tau = 0.07$$，$$1/\tau$$ 上限 100；batch 越大负样本越多，CLIP 用了 32768。
- **零样本分类**就是用类别提示的文本向量当分类器权重；提示模板和提示集成能明显提升效果。 迷你 CLIP 不用标签、只靠合成的图文对，零样本准确率 88.7%，和线性探针持平，还学会了 dark / light 这样的属性。
- **CLIP 的局限**：细粒度、计数、组合关系弱；batch 内同类样本会被当作负样本。
- **在生成模型里**：SD 用 CLIP 文本编码器的逐 token 输出，IP-Adapter 用图像向量，LLaVA 用倒数第二层的 patch token；SD3、FLUX 把 CLIP 和 T5 一起用。
- 本文的实现：ViT 与 torchvision `vit_b_16` 输出完全一致；CLIP 各配置的参数量与官方模型一致。
