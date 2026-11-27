---
date: 2026-11-27 05:00:00 -0800
keywords: >-
  LoRA, low-rank adaptation, parameter-efficient fine-tuning, PEFT, intrinsic rank, merge weights, Stable Diffusion LoRA, DreamBooth, prior preservation, style LoRA, AnimateDiff, MotionLoRA, QLoRA, DoRA, LyCORIS, diffusers, PyTorch.
summary: >-
  LoRA 为什么只训练一对低秩矩阵就能微调大模型：公式、初始化、α/r 缩放、合并后零推理开销，以及论文对"更新矩阵本来就是低秩"的分析。从一个可运行的 LoRALinear / LoRAConv2d 实现出发，在 FashionMNIST 小扩散模型上实际训练一个"线稿风格" LoRA，对比全量微调和不同 rank；再讲 Stable Diffusion 上 LoRA 插在哪、有多大，两个经典应用（DreamBooth 式的人物/风格定制，AnimateDiff 的视频 MotionLoRA），多 LoRA 组合和常见变体。
layout: note
title: "LoRA：用低秩矩阵微调大模型"
description: "How LoRA fine-tunes large models by training a low-rank update: math, initialization, alpha/r scaling, merging, a runnable implementation, a style-LoRA experiment on a tiny diffusion model, LoRA in Stable Diffusion, DreamBooth, AnimateDiff MotionLoRA, multi-LoRA composition and variants."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig svg .lane{font-size:14px;font-weight:600}
.ddpm-fig svg .it{font-style:italic;font-family:"Times New Roman",serif;font-size:16px}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
</style>

# LoRA：用低秩矩阵微调大模型

想让一个预训练好的大模型学会新东西（一种画风、一个人物、一个领域的文本），最直接的办法是**全量微调**：所有参数都参与训练。问题是模型越大越负担不起：每个任务都要存一整份模型，训练时还要为每个参数存梯度和优化器状态。

LoRA（Low-Rank Adaptation，Hu 等人，ICLR 2022，论文 *LoRA: Low-Rank Adaptation of Large Language Models*）的做法是：**冻结原模型的全部权重，只给其中一些线性层旁边加一对很小的矩阵 $$B A$$，训练时只更新这对矩阵**。它最早用在 GPT-3 这样的大语言模型上，后来几乎成了 [Stable Diffusion](/deep-learning/stable-diffusion/) 社区里"定制风格、人物、概念"的标准方式：一个 LoRA 文件只有几 MB 到几百 MB，可以随时加载、调强度、和别的 LoRA 叠加。

## 参考资料

本文主要参考 LoRA 原论文 [1]；Stable Diffusion 上的用法参考 Diffusers 文档 [3]，两个应用分别来自 DreamBooth [4] 和 AnimateDiff [5]。正文中用 [n] 标注具体出处。

- <a id="ref-1"></a>**[1] 主论文**：Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)。公式、初始化、α/r 缩放、GPT-3 上的结果和第 2.5 节的低秩分析（论文第 7 节）。
- <a id="ref-2"></a>**[2] 官方代码**：microsoft/LoRA [github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)。`loralib`，第 3 节的实现可以和它对照。
- <a id="ref-3"></a>**[3] Diffusers LoRA 训练文档**：Hugging Face Diffusers: LoRA [huggingface.co/docs/diffusers/en/training/lora](https://huggingface.co/docs/diffusers/en/training/lora)。第 5 节 SD 上的训练脚本和加载方式。
- <a id="ref-4"></a>**[4] DreamBooth**：Ruiz et al. *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*. CVPR 2023. [arxiv.org/abs/2208.12242](https://arxiv.org/abs/2208.12242)。第 6 节的稀有 token 和先验保留损失。
- <a id="ref-5"></a>**[5] AnimateDiff**：Guo et al. *AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning*. ICLR 2024. [arxiv.org/abs/2307.04725](https://arxiv.org/abs/2307.04725)。第 7 节的运动模块和 MotionLoRA；代码在 [guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff)。
- <a id="ref-6"></a>**[6] QLoRA**：Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)。第 9 节变体表里的第一行。

## 0. 全文路线

| 节 | 内容 |
|---|---|
| 1 | 全量微调贵在哪，已有的省参数方法有什么问题 |
| 2 | LoRA 的公式：低秩分解、初始化、$$\alpha/r$$ 缩放、合并；为什么低秩就够 |
| 3 | 实现：`LoRALinear`、`LoRAConv2d`、注入、合并、只存 LoRA 权重 |
| 4 | 实验：给 FashionMNIST 小扩散模型训练一个"线稿风格" LoRA |
| 5 | Stable Diffusion 上的 LoRA：插在哪、有多大、怎么用 Diffusers 训练和推理 |
| 6 | 经典应用一：DreamBooth 式的人物/物体/风格定制 |
| 7 | 经典应用二：视频生成，AnimateDiff 与 MotionLoRA |
| 8 | 多个 LoRA 的组合、强度和合并；训练时的常见问题 |
| 9 | 常见变体（QLoRA、DoRA、LoHa/LoKr 等），要点回顾 |

**符号约定**：

| 符号 | 含义 | 代码 | shape |
|---|---|---|---|
| $$W_0$$ | 预训练权重，冻结 | `base.weight` | $$d \times k$$（输出 × 输入） |
| $$A$$ | 降维矩阵，随机初始化 | `lora_A` | $$r \times k$$ |
| $$B$$ | 升维矩阵，初始化为 0 | `lora_B` | $$d \times r$$ |
| $$\Delta W = \frac{\alpha}{r} B A$$ | 学到的权重更新，秩 $$\le r$$ | `delta_weight()` | $$d \times k$$ |
| $$r$$ | rank（秩），远小于 $$d$$ 和 $$k$$ | `r` | 常用 4～128 |
| $$\alpha$$ | 缩放常数 | `alpha` | 常取 $$r$$ 或 $$2r$$ |
| $$s$$ | 推理时的 LoRA 强度 | `scale` | 0～1 左右 |

## 1. 全量微调贵在哪

### 1.1 三份成本

以 Adam 优化器、fp32 训练为例，每个可训练参数要占：参数本身 4 字节、梯度 4 字节、Adam 的两个动量 8 字节，一共约 16 字节。一个 175B 参数的 GPT-3 全量微调光这些就要 2.8 TB 显存。即使是 SD 1.5 这样 8.6 亿参数的 U-Net，全量微调也要十几 GB，并且：

1. **训练显存**：梯度和优化器状态正比于可训练参数量；
2. **存储和分发**：每个微调任务都要存一整份模型（SD 1.5 的 U-Net 约 3.4 GB fp32 / 1.7 GB fp16）；
3. **切换成本**：服务多个任务时，要么同时放多份模型，要么频繁换整套权重。

### 1.2 已有的省参数方法

在 LoRA 之前，常见的"只训练一小部分参数"的方法有两类，都有明显的缺点：

- **Adapter 层**：在 Transformer 的每层里串联插入一个小的瓶颈 MLP。参数少，但它是**串联**的，推理时每层多了一次计算，对小 batch、低延迟的场景影响明显，而且无法消掉。
- **Prefix / Prompt tuning**：在输入前面拼一段可学习的 token。参数更少，但占用了序列长度，训练不稳定，效果随可训练参数数量的变化也不单调。

LoRA 的出发点是：能不能**并联**地加一个可训练分支，训完以后**合并回原权重**，推理时完全没有额外开销？

### 1.3 低秩假设

之前的研究发现，预训练模型的微调存在一个很低的"内在维度"：只在一个很小的随机子空间里优化，也能达到全量微调的大部分效果。LoRA 进一步假设：**微调带来的权重变化 $$\Delta W$$ 本身是低秩的**。如果这成立，就没必要学一个完整的 $$d \times k$$ 矩阵，学两个瘦长的矩阵相乘就够了。

## 2. LoRA 的公式

### 2.1 前向

对一个线性层 $$h = W_0 x$$（$$W_0 \in \mathbb{R}^{d \times k}$$），LoRA 把它变成 [[1]](#ref-1)

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A x,
\qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)
$$

<figure class="ddpm-fig">
<svg viewBox="0 0 760 330" role="img" aria-label="LoRA 结构示意，按论文图 1 重画">
  <defs><marker id="ra" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker></defs>
  <!-- input x -->
  <rect x="250" y="285" width="260" height="30" rx="6" fill="#fff" stroke="#475467"/>
  <text x="380" y="305" text-anchor="middle">输入 x（k 维）</text>
  <!-- output h -->
  <rect x="250" y="15" width="260" height="30" rx="6" fill="#ecfdf3" stroke="#12b76a"/>
  <text x="380" y="35" text-anchor="middle">输出 h（d 维）</text>
  <!-- sum -->
  <circle cx="380" cy="75" r="12" fill="#fff" stroke="#475467"/><text x="380" y="80" text-anchor="middle">+</text>
  <line x1="380" y1="63" x2="380" y2="47" stroke="#475467" marker-end="url(#ra)"/>
  <!-- W0 -->
  <rect x="150" y="115" width="160" height="140" rx="8" fill="#eef4ff" stroke="#2a6fdb"/>
  <text x="230" y="175" text-anchor="middle" class="lane">预训练权重 W₀</text>
  <text x="230" y="196" text-anchor="middle" class="sm">d × k，冻结</text>
  <path d="M300,285 L300,270 L230,270 L230,257" fill="none" stroke="#475467" marker-end="url(#ra)"/>
  <path d="M230,115 L230,75 L366,75" fill="none" stroke="#475467" marker-end="url(#ra)"/>
  <!-- A trapezoid (wide bottom k -> narrow top r) -->
  <polygon points="450,255 610,255 565,205 495,205" fill="#fff7ed" stroke="#f79009"/>
  <text x="530" y="236" text-anchor="middle">A：r × k</text>
  <text x="640" y="238" class="sm">随机初始化</text>
  <!-- bottleneck r -->
  <rect x="500" y="178" width="60" height="22" rx="4" fill="#fff" stroke="#98a2b3"/>
  <text x="530" y="194" text-anchor="middle" class="sm">r 维</text>
  <!-- B trapezoid (narrow bottom r -> wide top d) -->
  <polygon points="495,170 565,170 610,120 450,120" fill="#fff7ed" stroke="#f79009"/>
  <text x="530" y="150" text-anchor="middle">B：d × r</text>
  <text x="640" y="150" class="sm">初始化为 0</text>
  <path d="M460,285 L460,270 L530,270 L530,257" fill="none" stroke="#475467" marker-end="url(#ra)"/>
  <path d="M530,118 L530,75 L394,75" fill="none" stroke="#475467" marker-end="url(#ra)"/>
  <text x="540" y="100" class="sm">× α/r（推理时再乘强度 s）</text>
  <text x="380" y="110" text-anchor="middle" class="sm">两条路并联</text>
</svg>
<figcaption>按 LoRA 论文图 1 重画。蓝色的 W₀ 冻结不动，只训练橙色的 A、B；中间只有 r 维。B 初始化为 0，所以训练开始时输出和原模型完全一样。</figcaption>
</figure>

要点：

- **参数量**：一个 $$d \times k$$ 的层，全量微调要训练 $$dk$$ 个参数，LoRA 只训练 $$r(d + k)$$ 个。例如 $$d = k = 768$$、$$r = 4$$ 时是 6144 对 589,824，约 1%。
- **计算方式**：前向时先算 $$Ax$$（降到 $$r$$ 维）再乘 $$B$$，从不显式构造 $$d \times k$$ 的 $$BA$$，额外计算量也只有 $$r(d + k)$$ 量级。
- **梯度**：$$W_0$$ 不需要梯度，也不需要优化器状态；反向传播仍然要**穿过** $$W_0$$（把梯度传给前面的层和 $$A$$），这部分激活不能省。

### 2.2 初始化：为什么 B = 0

论文用随机高斯初始化 $$A$$，把 $$B$$ 初始化为 0（实现里 $$A$$ 常用 Kaiming 均匀初始化，效果一样）。于是训练开始时 $$\Delta W = BA = 0$$，**模型和预训练模型逐位相同**，微调从原模型出发，不会一上来就被随机分支打乱。这和 [ControlNet](/deep-learning/controlnet/) 的零卷积是同一个思路。

为什么不两个都设成 0？因为

$$
\frac{\partial L}{\partial B} = \frac{\alpha}{r} \cdot \frac{\partial L}{\partial h}\, (A x)^\top,\qquad
\frac{\partial L}{\partial A} = \frac{\alpha}{r} \cdot B^\top \frac{\partial L}{\partial h}\, x^\top
$$

$$A = 0$$ 时 $$B$$ 的梯度为 0，$$B = 0$$ 时 $$A$$ 的梯度为 0，两个都是 0 就永远学不动。只让 $$B = 0$$：第一步 $$B$$ 有梯度、$$A$$ 没有；$$B$$ 一旦离开 0，$$A$$ 也开始更新。

### 2.3 α/r 缩放

$$\Delta W x$$ 前面乘了一个常数 $$\alpha / r$$。原因：$$A$$ 的初始化尺度固定时，$$r$$ 越大，$$BAx$$ 的初始更新幅度越大，换一个 $$r$$ 就要重新调学习率。除以 $$r$$ 抵消了这一点；$$\alpha$$ 设成第一个试的 $$r$$ 值以后就不再调（论文的说法：调 $$\alpha$$ 和调学习率差不多）。实践中常见 $$\alpha = r$$（缩放为 1）或 $$\alpha = 2r$$。

> 注意不同工具对 $$\alpha$$ 的默认值不同：有的默认 $$\alpha = r$$，有的默认 $$\alpha = 1$$。同样标着 "rank 32" 的两个 LoRA，有效强度可能差 32 倍。

### 2.4 合并：推理时零额外开销

训练完以后，可以把 LoRA 直接加回原权重：

$$
W = W_0 + s \cdot \frac{\alpha}{r} B A
$$

合并后就是一个普通的线性层，**推理速度和原模型完全一样**，这是它相对 Adapter 层的最大优势。想换任务时减回去 $$W_0 = W - s \frac{\alpha}{r} BA$$ 就行。不合并时，强度 $$s$$ 可以在推理时随时调：$$s = 0$$ 精确回到原模型，$$s = 1$$ 是训练时的效果，$$0 < s < 1$$ 是两者之间。

### 2.5 低秩为什么够：论文的分析

论文在 GPT-3 175B 上只给自注意力的 $$W_q$$、$$W_v$$ 加 LoRA（MLP 和其余层都冻结），观察到 [[1]](#ref-1)：

- **可训练参数减少约 10,000 倍，训练显存减少约 3 倍**（1.2 TB → 350 GB），每个任务的权重文件从 350 GB 降到约 35 MB，训练吞吐还提高了约 25%（大部分参数不需要算梯度）；在多个基准上和全量微调持平或更好。
- **很小的 $$r$$ 就够了**：只适配 $$W_q$$、$$W_v$$ 时，$$r = 1$$ 或 2 的效果就已经很接近 $$r = 64$$。用不同 $$r$$ 学到的 $$\Delta W$$，最重要的几个奇异方向高度重合，再增加 $$r$$ 多出来的方向大多是噪声。
- **$$\Delta W$$ 放大的是原模型"有但没强调"的方向**：$$\Delta W$$ 和 $$W_0$$ 的主方向相关性不高，而是放大了 $$W_0$$ 里已经存在、但权重很小的一些特征方向，而且放大倍数很大（论文里 $$r = 4$$ 时约 20 倍）。也就是说，微调更像"把模型已经会的、和新任务相关的能力调出来"，而不是学全新的东西，这就是低秩能够用的原因。
- **适配多种矩阵比只适配一种、用较大的 $$r$$ 更好**：同样的参数预算下，$$W_q$$ 和 $$W_v$$ 各用 $$r = 4$$，好于只给 $$W_q$$ 用 $$r = 8$$。

> **易错点（原理）**
>
> 1. **LoRA 不省前向和反向的激活。** 省的是梯度、优化器状态和存储；反向传播仍然要穿过冻结的 $$W_0$$。所以 LoRA 微调的显存降幅主要来自优化器，激活要靠 gradient checkpointing、混合精度等手段另外省。
> 2. **$$A$$、$$B$$ 不能都初始化为 0。** 两个都为 0 时梯度永远为 0；只有 $$B = 0$$ 才能既保证初始输出不变，又能学。
> 3. **$$r$$ 不是越大越好。** $$r$$ 大了参数变多、更容易过拟合，在小数据（比如十几张人物照片）上尤其明显；先从小 $$r$$ 试。
> 4. **$$\alpha$$ 的约定要对上。** 换训练工具或者加载别人的 LoRA 时，先确认 $$\alpha$$ 和 $$r$$，否则强度会差很多。

## 3. 实现

### 3.1 线性层

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """y = W0 x + (α/r) · B A x；W0 冻结，只训练 A、B"""
    def __init__(self, base: nn.Linear, r=4, alpha=4.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))       # A: [r, k]
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))      # B: [d, r]，初始为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r
        self.scale = 1.0                                                   # 推理时可调的 LoRA 强度

    def forward(self, x):
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()                    # B A x，不显式算 d×k 的矩阵
        return self.base(x) + self.scale * self.scaling * delta

    def delta_weight(self):
        return self.scaling * self.lora_B @ self.lora_A                    # ΔW = (α/r) B A，[d, k]
```

### 3.2 卷积层

扩散模型的 U-Net 里大部分是卷积。把一个 $$C_{\text{out}} \times C_{\text{in}} \times k_h \times k_w$$ 的卷积看成"输入是 $$C_{\text{in}} k_h k_w$$ 维的 patch、输出 $$C_{\text{out}}$$ 维"的线性层，LoRA 就自然推广过来：$$A$$ 是一个 $$k_h \times k_w$$ 的卷积，把 $$C_{\text{in}}$$ 降到 $$r$$；$$B$$ 是 $$1 \times 1$$ 卷积，把 $$r$$ 升到 $$C_{\text{out}}$$。社区里这种做法叫 LoCon。 [[2]](#ref-2)

```python
class LoRAConv2d(nn.Module):
    """卷积版：A 是 r 个输出通道的 k×k 卷积，B 是 1×1 卷积；ΔW 的秩 ≤ r"""
    def __init__(self, base: nn.Conv2d, r=4, alpha=4.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Conv2d(base.in_channels, r, base.kernel_size, base.stride,
                                base.padding, bias=False)
        self.lora_B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)                                 # B = 0 → 初始 ΔW = 0
        self.scaling = alpha / r
        self.scale = 1.0

    def forward(self, x):
        return self.base(x) + self.scale * self.scaling * self.lora_B(self.lora_A(x))

    def delta_weight(self):
        # [out, r, 1, 1] 和 [r, in, kh, kw] 相乘 → [out, in, kh, kw]
        return self.scaling * torch.einsum("or,rikl->oikl", self.lora_B.weight[:, :, 0, 0],
                                           self.lora_A.weight)
```

### 3.3 注入、合并、保存

```python
def inject_lora(model, r=4, alpha=4.0, target=(nn.Linear, nn.Conv2d)):
    """把 model 里所有目标层原地替换成 LoRA 层；先冻结整个模型"""
    model.requires_grad_(False)
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and nn.Linear in target:
            setattr(model, name, LoRALinear(child, r, alpha))
        elif isinstance(child, nn.Conv2d) and nn.Conv2d in target:
            setattr(model, name, LoRAConv2d(child, r, alpha))
        else:
            inject_lora(child, r, alpha, target)                           # 递归进子模块
    return model


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, (LoRALinear, LoRAConv2d))]


def set_lora_scale(model, scale):
    for m in lora_modules(model):
        m.scale = scale


def lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


@torch.no_grad()
def merge_lora(model):
    """W ← W0 + s·ΔW，再把 LoRA 层换回普通层：推理时零额外开销"""
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, (LoRALinear, LoRAConv2d)):
                child.base.weight += child.scale * child.delta_weight()
                setattr(parent, name, child.base)
    return model
```

几个检查，确认实现和公式一致（用的是 [DDPM 篇](/deep-learning/ddpm-from-scratch/)的 `ClassConditionedTinyUNet`）：

```python
torch.manual_seed(0)
unet = ClassConditionedTinyUNet(10, in_channels=1, base_channels=48, time_dim=128)
x, t, y = torch.randn(4, 1, 28, 28), torch.tensor([1, 50, 100, 200]), torch.tensor([0, 1, 2, 10])
with torch.no_grad():
    out_base = unet(x, t, y)

inject_lora(unet, r=4, alpha=4)
n_lora = sum(p.numel() for p in unet.parameters() if p.requires_grad)
print(n_lora)                                                  # 25708（原模型 461633 的 5.6%）
with torch.no_grad():
    assert torch.equal(unet(x, t, y), out_base)                # B = 0：初始输出逐位相同

for m in lora_modules(unet):                                   # 假装训练过：给 B 一些非零值
    nn.init.normal_(m.lora_B if isinstance(m, LoRALinear) else m.lora_B.weight, std=0.02)
with torch.no_grad():
    out_lora = unet(x, t, y)
    set_lora_scale(unet, 0.0)
    assert torch.equal(unet(x, t, y), out_base)                # s = 0：精确回到原模型
    set_lora_scale(unet, 1.0)
    merge_lora(unet)
    assert torch.allclose(unet(x, t, y), out_lora, atol=1e-4)  # 合并前后输出一致（只差浮点误差）
```

> **易错点（实现）**
>
> 1. **先冻结、再注入。** `inject_lora` 第一行把整个模型 `requires_grad_(False)`，之后新建的 `lora_A`、`lora_B` 默认 `requires_grad=True`。顺序反过来，就会把刚建的 LoRA 参数也冻住。优化器只放 `requires_grad` 的参数。
> 2. **只保存 LoRA 权重。** `lora_state_dict` 只挑名字里带 `lora_` 的张量；把整个 `state_dict` 存下来就失去了"文件很小"的意义。
> 3. **合并后不能再调强度。** 合并把 $$s$$ 烘焙进了权重；要调强度或者叠加别的 LoRA，就先别合并。
> 4. **bias、归一化层、Embedding 默认不加 LoRA。** 它们要么参数很少（可以直接全量训练），要么不是矩阵乘法。本文的 Tiny U-Net 里转置卷积 `up` 也没有加，需要时可以仿照 `LoRAConv2d` 写一个。

## 4. 实验：给小扩散模型训练一个"线稿风格" LoRA

模拟一个最常见的 LoRA 场景：**已有一个训练好的生成模型，想让它学会一种新画风**。

- **底模**：[DDPM 篇](/deep-learning/ddpm-from-scratch/)里训练好的类别条件 Tiny U-Net（FashionMNIST，$$T = 300$$，基础通道 48，46 万参数），它生成的是普通的灰度衣服；
- **新风格**："线稿"：对每张训练图做 Canny 边缘检测，得到白线黑底的图，作为新的训练目标，类别标签不变；
- **四种微调**：全量微调（学习率 $$10^{-4}$$），以及在所有卷积层和线性层上加 LoRA，$$r = 1, 4, 16$$（$$\alpha = r$$，学习率 $$10^{-3}$$）；
- 每种都训练 1000 步，batch 64，10% 丢标签；评估用 512 张测试图的线稿，在固定的时间步和噪声上算去噪损失；
- 采样：DDIM 50 步，CFG $$w = 3$$，每类一张，四种方法用同一个初始噪声。

### 4.1 强度 s：从原模型连续过渡到新风格

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/lora/lora_scale_sweep.png" alt="LoRA 强度从 0 到 1 的生成结果">
  <figcaption>同一个 r = 4 的 LoRA、同一个初始噪声，只改推理时的强度 s。最后一行是每个类别的一张真实线稿（训练目标的样子）。</figcaption>
</figure>

- **$$s = 0$$ 和底模逐像素相同**（程序里 `torch.equal` 为真）。LoRA 没有改动底模的任何权重，关掉它就精确回到原模型。全量微调做不到这一点：要回到原模型只能重新加载另一份权重。
- $$s$$ 从 0.25 到 1，图像**连续地**从实心的灰度衣服变成白色线条，而衣服的整体形状、位置基本跟着底模走（同一个初始噪声）。这就是 SD 里"LoRA 权重 0.6"这类旋钮的作用。
- 训练 1000 步后，线稿的风格已经学到了，但线条比真实线稿杂乱，衣服内部多了很多碎线。原因一是训练步数少，二是 1 像素宽的线条对这个小模型本来就难（底模自己生成的衣服纹理也偏噪），不是 LoRA 特有的问题：全量微调的结果也是这样（见下图）。

### 4.2 和全量微调比

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/lora/lora_vs_full.png" alt="全量微调与不同 rank 的 LoRA 生成结果">
  <figcaption>四种微调方法，同一个初始噪声，强度都是 1。</figcaption>
</figure>

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/lora/lora_val_curves.png" alt="四种微调方法在线稿验证集上的损失曲线">
  <figcaption>线稿验证集上的去噪损失（越低越好）。第 0 步是底模本身。</figcaption>
</figure>

| 方法 | 可训练参数 | 占底模 | 要保存的文件（fp32） | 线稿验证损失（1000 步） |
|---|---|---|---|---|
| 全量微调 | 461,633 | 100% | 1.85 MB（整个模型） | 0.0764 |
| LoRA $$r = 1$$ | 6,427 | 1.4% | 26 KB | 0.0951 |
| LoRA $$r = 4$$ | 25,708 | 5.6% | 103 KB | 0.0802 |
| LoRA $$r = 16$$ | 102,832 | 22% | 411 KB | 0.0760 |

（底模在线稿上的损失是 0.224。）

- **$$r = 4$$ 用 5.6% 的参数就达到了全量微调的大部分效果**，$$r = 16$$ 和全量微调基本持平；$$r = 1$$ 明显差一些，但也已经从 0.224 降到了 0.095，生成的图也是线稿风格。
- 这个底模很小（46 万参数），所以 LoRA 占的比例看起来不算小。参数量是 $$r(d + k)$$ 对 $$dk$$，层越宽，比例越小：同样 $$r = 4$$，在 SD 1.5 上只占 0.09%（第 5 节）。
- 合并检查：把 $$r = 4$$ 的 LoRA 合并进权重后重新采样，和不合并的结果最大差 $$4 \times 10^{-5}$$（浮点误差），合并后模型里已经没有任何 LoRA 层，推理和原模型一样快。

> **易错点（实验）**
>
> 1. **LoRA 的学习率要单独设。** 这里 LoRA 用 $$10^{-3}$$、全量微调用 $$10^{-4}$$，这是常见的搭配。LoRA 直接沿用全量微调的学习率时通常收敛慢得多，同样步数下容易得出"LoRA 不如全量微调"的错误结论。
> 2. **训练时的强度要是 1。** 推理时调的 $$s$$ 只是乘在 $$\Delta W$$ 上的系数；训练时一直用 1，否则 $$B$$、$$A$$ 学到的幅度会跟着变。
> 3. **比较时用同一个初始噪声。** 不同方法、不同强度之间的差异本来就细，换了噪声以后图像差别主要来自采样随机性，看不出方法本身的区别。

## 5. Stable Diffusion 上的 LoRA

### 5.1 插在哪

SD 社区最早（2022 年底）把 LoRA 用到扩散模型上时，沿用了论文"只适配注意力"的思路：给 U-Net 里**每个注意力模块的 $$W_q, W_k, W_v, W_{\text{out}}$$** 加 LoRA，self-attention 和 cross-attention 都加。Diffusers 训练脚本的默认目标就是这四个（在 Diffusers 里叫 `to_q`、`to_k`、`to_v`、`to_out.0`）。

<figure class="ddpm-fig">
<svg viewBox="0 0 760 250" role="img" aria-label="SD U-Net 的 Transformer block 中 LoRA 的位置">
  <defs><marker id="sa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker></defs>
  <text x="20" y="24" class="lane">SD U-Net 里的一个 BasicTransformerBlock（共 16 个）</text>
  <rect x="20" y="50" width="90" height="40" rx="7" fill="#fff" stroke="#475467"/><text x="65" y="74" text-anchor="middle">特征 token</text>
  <rect x="140" y="40" width="190" height="60" rx="8" fill="#eef4ff" stroke="#2a6fdb"/>
  <text x="235" y="60" text-anchor="middle">Self-attention</text>
  <text x="235" y="82" text-anchor="middle" style="fill:#d92d20">q · k · v · out ＋ LoRA</text>
  <rect x="360" y="40" width="190" height="60" rx="8" fill="#eef4ff" stroke="#2a6fdb"/>
  <text x="455" y="60" text-anchor="middle">Cross-attention</text>
  <text x="455" y="82" text-anchor="middle" style="fill:#d92d20">q · k · v · out ＋ LoRA</text>
  <rect x="580" y="40" width="150" height="60" rx="8" fill="#f2f4f7" stroke="#98a2b3"/>
  <text x="655" y="60" text-anchor="middle">前馈 GEGLU</text>
  <text x="655" y="82" text-anchor="middle" class="sm">默认不加（可选）</text>
  <line x1="110" y1="70" x2="138" y2="70" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="330" y1="70" x2="358" y2="70" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="550" y1="70" x2="578" y2="70" stroke="#475467" marker-end="url(#sa)"/>
  <rect x="360" y="140" width="190" height="40" rx="7" fill="#f2f4f7" stroke="#98a2b3"/><text x="455" y="164" text-anchor="middle">CLIP 文本 token [77, 768]</text>
  <line x1="455" y1="140" x2="455" y2="102" stroke="#475467" marker-end="url(#sa)"/>
  <text x="525" y="125" class="sm">k、v 的输入</text>
  <rect x="20" y="200" width="300" height="34" rx="7" fill="#f2f4f7" stroke="#98a2b3"/><text x="170" y="222" text-anchor="middle" class="sm">ResNet 块的卷积：默认不加；加了就是 LoCon</text>
  <rect x="360" y="200" width="370" height="34" rx="7" fill="#f2f4f7" stroke="#98a2b3"/><text x="545" y="222" text-anchor="middle" class="sm">CLIP 文本编码器：可选，q/k/v/out_proj 也可以加 LoRA</text>
</svg>
<figcaption>红字是默认加 LoRA 的位置。cross-attention 的 k、v 直接作用在文本 token 上，所以 LoRA 在这里改变的是"文字 → 画面"的对应关系，这对学新概念、新风格最有效。</figcaption>
</figure>

用 [SD 篇第 6 节](/deep-learning/stable-diffusion/)的 `SD15UNet` 实际数一下（只适配四个注意力投影，128 个线性层）：

```python
def inject_lora_by_name(model, names=("to_q", "to_k", "to_v", "to_out"), r=4, alpha=4.0):
    """只把名字在 names 里的线性层换成 LoRA（SD 的默认做法）"""
    model.requires_grad_(False)
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in names and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha))
    return model


unet = inject_lora_by_name(SD15UNet(), r=4)
print(sum(p.numel() for p in unet.parameters() if p.requires_grad))   # 797184
```

| rank $$r$$ | 可训练参数 | 占 U-Net（8.6 亿） | 文件大小（fp16） |
|---|---|---|---|
| 1 | 19.9 万 | 0.023% | 0.4 MB |
| 4 | 79.7 万 | 0.093% | 1.6 MB |
| 16 | 319 万 | 0.37% | 6.4 MB |
| 64 | 1275 万 | 1.5% | 26 MB |
| 128 | 2551 万 | 3.0% | 51 MB |

社区里常见的 SD 1.5 LoRA 文件从几 MB 到一两百 MB 不等，差别就在 rank、是否也加了卷积层和文本编码器、以及存的是 fp16 还是 fp32。和 1.7 GB 的完整 U-Net 相比都很小。

### 5.2 用 Diffusers 训练

训练循环和 [SD 篇第 7 节](/deep-learning/stable-diffusion/)的完整微调完全一样，只有三处不同：U-Net 冻结后挂上 LoRA、优化器只放 LoRA 参数、只保存 LoRA 权重。Diffusers 通过 PEFT 库实现 LoRA [[3]](#ref-3)：

```python
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers import StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers

unet.requires_grad_(False)                       # vae、text_encoder 同样冻结
unet.add_adapter(LoraConfig(
    r=4, lora_alpha=4, init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
))
lora_params = [p for p in unet.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)   # 比全量微调（约 1e-5）大一个量级

# ……训练循环：VAE 编码 → 加噪 → unet 预测噪声 → MSE，和全量微调一样……

StableDiffusionPipeline.save_lora_weights(       # 只存 LoRA：几 MB
    "outputs/my_lora",
    unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)),
)
```

也可以直接用 Diffusers 自带的脚本（`examples/text_to_image/train_text_to_image_lora.py`）：

```bash
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --dataset_name="YOUR_IMAGE_CAPTION_DATASET" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --max_train_steps=15000 --learning_rate=1e-04 --lr_scheduler="cosine" \
  --rank=4 --mixed_precision="fp16" \
  --validation_prompt="YOUR TEST PROMPT" \
  --output_dir="outputs/my_lora"
```

### 5.3 用 Diffusers 推理

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("outputs/my_lora", adapter_name="style")
pipe.set_adapters(["style"], adapter_weights=[0.8])        # 强度 s = 0.8
image = pipe("a castle on a hill, in my style", num_inference_steps=30).images[0]

pipe.fuse_lora(lora_scale=0.8)       # 合并进权重：之后推理没有额外开销
pipe.unfuse_lora()                   # 减回去
pipe.unload_lora_weights()           # 完全卸载，回到原模型
```

> **易错点（SD 上的 LoRA）**
>
> 1. **LoRA 和底模是绑定的。** 在 SD 1.5 上训的 LoRA 只能用在 SD 1.5 及其微调版本上（层名、形状都得对上）；SDXL、SD 3 的结构不同，不能混用。用在和训练底模差别很大的微调模型上，效果也会打折扣。
> 2. **学习率要比全量微调大。** LoRA 常用 $$10^{-4}$$ 量级，全量微调常用 $$10^{-5}$$ 量级；用全量微调的学习率训 LoRA，往往几千步也看不出变化。
> 3. **训练时可以用 fp16，LoRA 参数最好保持 fp32。** 底模用半精度节省显存，LoRA 参数很少，用 fp32 存可以避免小更新被舍入掉（Diffusers 脚本默认就是这样做的）。
> 4. **文本编码器要不要加 LoRA。** 学新的人物或新词（例如一个罕见 token）时，给文本编码器也加 LoRA 往往更好；纯画风迁移通常只加 U-Net 就够。

## 6. 经典应用一：人物、物体、风格定制

LoRA 在图像生成里最常见的用途是**定制**：让模型学会一个它原本不认识的具体对象（某个人、某只狗、某个产品）或一种具体风格，然后在新的 prompt 里自由组合。这类方法的代表是 DreamBooth（Ruiz 等人，CVPR 2023）。原版 DreamBooth 全量微调整个模型，社区很快把它和 LoRA 结合起来（DreamBooth-LoRA），成为最常用的定制方式。

### 6.1 DreamBooth 的做法

只有 3～5 张某个对象的照片，怎么让模型学会它？ [[4]](#ref-4)

1. **用一个稀有 token 当名字**：给所有训练图配同一句 prompt，例如 "a photo of **sks** dog"。`sks` 是一个模型几乎没有先验含义的 token，训练后它就"绑定"到这只狗上；后面的 "dog" 是类别词，让模型借用自己对"狗"的已有知识。
2. **先验保留损失（prior preservation）**：只用这几张图训练，模型很快会出现**语言漂移**，即"dog"这个词本身也被带偏，所有狗都开始像这只狗，同时多样性下降。DreamBooth 的办法是：先用**原模型**按 "a photo of dog" 生成几百张"普通的狗"，训练时和实例图一起训：

$$
\mathcal{L} =
\underbrace{\mathbb{E}\Big[\big\|\epsilon - \epsilon_\theta(z_t, t, c_{\text{inst}})\big\|^2\Big]}_{\text{学会这只 sks 狗}}
+ \lambda\,
\underbrace{\mathbb{E}\Big[\big\|\epsilon' - \epsilon_\theta(z'_{t'}, t', c_{\text{class}})\big\|^2\Big]}_{\text{"dog" 的含义别变}}
$$

其中 $$c_{\text{inst}}$$ 是 "a photo of sks dog"，$$c_{\text{class}}$$ 是 "a photo of dog"，$$z'$$ 来自原模型生成的类别图，$$\lambda$$ 常取 1。

<figure class="ddpm-fig">
<svg viewBox="0 0 760 270" role="img" aria-label="DreamBooth LoRA 的训练数据流">
  <defs><marker id="da" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker></defs>
  <text x="20" y="22" class="lane">训练前：生成类别图（一次）</text>
  <rect x="20" y="36" width="160" height="40" rx="7" fill="#fff" stroke="#475467"/><text x="100" y="60" text-anchor="middle">"a photo of dog"</text>
  <rect x="210" y="36" width="150" height="40" rx="7" fill="#f2f4f7" stroke="#98a2b3"/><text x="285" y="54" text-anchor="middle">原模型（不带 LoRA）</text><text x="285" y="69" text-anchor="middle" class="sm">采样 200～1000 张</text>
  <rect x="390" y="36" width="150" height="40" rx="7" fill="#fff" stroke="#475467"/><text x="465" y="60" text-anchor="middle">类别图（普通的狗）</text>
  <line x1="180" y1="56" x2="208" y2="56" stroke="#475467" marker-end="url(#da)"/>
  <line x1="360" y1="56" x2="388" y2="56" stroke="#475467" marker-end="url(#da)"/>
  <line x1="10" y1="96" x2="750" y2="96" stroke="#eaecf0" stroke-width="2"/>
  <text x="20" y="122" class="lane">训练：每个 batch 一半实例、一半类别</text>
  <rect x="20" y="138" width="160" height="40" rx="7" fill="#fff" stroke="#475467"/><text x="100" y="155" text-anchor="middle">3～5 张实例图</text><text x="100" y="170" text-anchor="middle" class="sm">"a photo of sks dog"</text>
  <rect x="20" y="198" width="160" height="40" rx="7" fill="#fff" stroke="#475467"/><text x="100" y="215" text-anchor="middle">类别图</text><text x="100" y="230" text-anchor="middle" class="sm">"a photo of dog"</text>
  <rect x="220" y="160" width="190" height="56" rx="8" fill="#eef4ff" stroke="#2a6fdb" stroke-width="2"/>
  <text x="315" y="183" text-anchor="middle">冻结的 SD + LoRA</text>
  <text x="315" y="202" text-anchor="middle" class="sm">只训练 LoRA（可含文本编码器）</text>
  <path d="M180,158 L200,158 L200,180 L218,180" fill="none" stroke="#475467" marker-end="url(#da)"/>
  <path d="M180,218 L200,218 L200,198 L218,198" fill="none" stroke="#475467" marker-end="url(#da)"/>
  <rect x="450" y="140" width="140" height="40" rx="7" fill="#fef3f2" stroke="#d92d20"/><text x="520" y="157" text-anchor="middle">实例损失</text><text x="520" y="172" text-anchor="middle" class="sm">学会 sks 狗</text>
  <rect x="450" y="196" width="140" height="40" rx="7" fill="#fef3f2" stroke="#d92d20"/><text x="520" y="213" text-anchor="middle">先验损失 × λ</text><text x="520" y="228" text-anchor="middle" class="sm">保住 "dog" 的含义</text>
  <path d="M410,180 L430,180 L430,160 L448,160" fill="none" stroke="#475467" marker-end="url(#da)"/>
  <path d="M410,198 L430,198 L430,216 L448,216" fill="none" stroke="#475467" marker-end="url(#da)"/>
  <rect x="620" y="168" width="120" height="40" rx="7" fill="#ecfdf3" stroke="#12b76a"/><text x="680" y="185" text-anchor="middle">LoRA 文件</text><text x="680" y="200" text-anchor="middle" class="sm">几 MB</text>
  <path d="M590,188 L618,188" fill="none" stroke="#475467" marker-end="url(#da)"/>
</svg>
<figcaption>DreamBooth-LoRA：用原模型先生成类别图，训练时实例图和类别图各占一半 batch，只更新 LoRA。推理时 "a sks dog on the beach" 就能画出这只狗。</figcaption>
</figure>

### 6.2 训练一步

实例和类别拼成一个 batch 一起前向，再把预测拆成两半分别算损失（这也是 Diffusers 的 DreamBooth 脚本的做法）：

```python
def dreambooth_lora_step(unet, vae, text_encoder, batch, noise_scheduler, optimizer,
                         prior_loss_weight=1.0):
    # batch 前一半是实例图（"a photo of sks dog"），后一半是类别图（"a photo of dog"）
    with torch.no_grad():
        latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    noise = torch.randn_like(latents)
    num_t = noise_scheduler.config.num_train_timesteps
    t = torch.randint(0, num_t, (latents.shape[0],), device=latents.device)
    noisy = noise_scheduler.add_noise(latents, noise, t)
    text = text_encoder(batch["input_ids"])[0]              # 文本编码器带 LoRA 时不能 no_grad
    pred = unet(noisy, t, encoder_hidden_states=text).sample

    pred_inst, pred_prior = pred.chunk(2)
    noise_inst, noise_prior = noise.chunk(2)
    loss = F.mse_loss(pred_inst.float(), noise_inst.float())                   # 实例损失
    loss = loss + prior_loss_weight * F.mse_loss(pred_prior.float(), noise_prior.float())  # 先验损失
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()
```

对应的 Diffusers 脚本是 `examples/dreambooth/train_dreambooth_lora.py`，关键参数是 `--instance_prompt="a photo of sks dog"`、`--with_prior_preservation --class_prompt="a photo of dog" --num_class_images=200`、`--rank`、`--train_text_encoder`。

### 6.3 风格 LoRA 和人物 LoRA 的区别

| | 对象 / 人物 LoRA | 风格 LoRA |
|---|---|---|
| 数据 | 同一对象的 3～20 张图，角度、背景尽量多样 | 同一风格的 20～几百张图，内容尽量多样 |
| 文本 | 稀有 token + 类别词："a photo of sks dog" | 每张图都写描述性 caption，再加风格触发词 |
| 先验保留 | 常用，防止类别词漂移 | 一般不用 |
| 主要风险 | 过拟合：背景、姿势、衣服都被"记住" | 风格学不进去，或者把训练集的内容也学进去 |
| 推理强度 | 常用 0.7～1.0 | 常用 0.5～1.0，可和别的 LoRA 叠加 |

> **易错点（定制）**
>
> 1. **数据要多样，caption 要描述"不想被学进去的东西"。** 如果人物照片全是白背景，模型会把白背景也当成"sks"的一部分；在 caption 里写出背景、动作、衣服，这些因素就交给文字控制，不会绑到触发词上。
> 2. **步数不是越多越好。** 几张图的 DreamBooth 很容易过拟合，表现为生成结果几乎复制训练图、换 prompt 也换不掉姿势。用固定种子每隔几百步看一次样张，选最早"像了"的那个检查点。
> 3. **类别图要用原模型生成。** 用别的数据集的图做先验，等于又引入了一个新分布；先验损失的目的就是让模型对"dog"的理解保持在原模型的状态。

## 7. 经典应用二：视频生成里的 LoRA

视频扩散模型比图像模型大得多、训练数据也更难得，所以"冻结大模型、只训练小增量"在视频上更有吸引力。LoRA 在视频生成里有两种典型用法。

### 7.1 AnimateDiff：让已有的图像 LoRA 直接动起来

AnimateDiff（Guo 等人，ICLR 2024）的出发点是：社区里已经有成千上万个基于 SD 1.5 的个性化模型（DreamBooth 模型和 LoRA），能不能**不重新训练它们**，就让它们生成视频？ [[5]](#ref-5)

它的做法是在**冻结的** SD 1.5 U-Net 的每个分辨率层里插入一个**运动模块**（motion module）：一组沿时间维做注意力的 Transformer 层。U-Net 原有的层逐帧处理（把帧数并到 batch 维），运动模块把同一空间位置在不同帧上的特征拿出来做 self-attention，负责"帧与帧之间怎么动"。运动模块在视频数据上训练一次，之后可以插到任何同源的个性化模型上。

于是一个视频生成系统由四部分**叠加**而成：

<figure class="ddpm-fig">
<svg viewBox="0 0 760 290" role="img" aria-label="AnimateDiff 中图像 LoRA、运动模块和 MotionLoRA 的叠加关系">
  <rect x="20" y="20" width="460" height="250" rx="10" fill="#eef4ff" stroke="#2a6fdb"/>
  <text x="250" y="44" text-anchor="middle" class="lane">冻结的 SD 1.5 U-Net（逐帧处理空间信息）</text>
  <rect x="40" y="60" width="200" height="190" rx="8" fill="#fff" stroke="#98a2b3"/>
  <text x="140" y="82" text-anchor="middle">空间层（每帧独立）</text>
  <text x="140" y="104" text-anchor="middle" class="sm">ResNet + self/cross-attention</text>
  <rect x="58" y="120" width="164" height="50" rx="7" fill="#fff7ed" stroke="#f79009"/>
  <text x="140" y="142" text-anchor="middle">图像 LoRA</text>
  <text x="140" y="160" text-anchor="middle" class="sm">人物 / 风格（社区已有的）</text>
  <text x="140" y="200" text-anchor="middle" class="sm">决定"每一帧画什么、</text>
  <text x="140" y="216" text-anchor="middle" class="sm">长什么样"</text>
  <rect x="260" y="60" width="200" height="190" rx="8" fill="#fff" stroke="#12b76a"/>
  <text x="360" y="82" text-anchor="middle">运动模块（插在每层之后）</text>
  <text x="360" y="104" text-anchor="middle" class="sm">沿时间维的 self-attention</text>
  <rect x="278" y="120" width="164" height="50" rx="7" fill="#fff7ed" stroke="#f79009"/>
  <text x="360" y="142" text-anchor="middle">MotionLoRA</text>
  <text x="360" y="160" text-anchor="middle" class="sm">一种镜头运动（推拉摇移）</text>
  <text x="360" y="200" text-anchor="middle" class="sm">决定"帧与帧之间</text>
  <text x="360" y="216" text-anchor="middle" class="sm">怎么动"</text>
  <rect x="510" y="20" width="230" height="250" rx="10" fill="#f9fafb" stroke="#d0d5dd"/>
  <text x="625" y="44" text-anchor="middle" class="lane">各部分怎么来</text>
  <text x="525" y="76" class="sm">SD 1.5：图像预训练，冻结</text>
  <text x="525" y="104" class="sm">图像 LoRA：图像上训练</text>
  <text x="525" y="120" class="sm">（直接复用，不用再训练）</text>
  <text x="525" y="148" class="sm">运动模块：大规模视频上训练一次，</text>
  <text x="525" y="164" class="sm">训练时 SD 冻结</text>
  <text x="525" y="192" class="sm">MotionLoRA：在运动模块上加 LoRA，</text>
  <text x="525" y="208" class="sm">用约 20～50 段同一种镜头运动的</text>
  <text x="525" y="224" class="sm">视频训练</text>
  <text x="525" y="252" class="sm">橙色两块都是 LoRA，可调强度、可组合</text>
</svg>
<figcaption>AnimateDiff 的分工：空间层管内容和外观，运动模块管时间上的变化。两侧都可以挂 LoRA，互不干扰。</figcaption>
</figure>

图像 LoRA 能直接复用的原因正是 LoRA 的"增量"性质：它只改空间层的权重，运动模块只读空间层的输出，两者在不同的层上工作。

### 7.2 MotionLoRA：用 LoRA 学一种镜头运动

运动模块学的是"一般的运动"；想要特定的镜头运动（推近、拉远、左右平移、旋转），AnimateDiff 给运动模块的注意力层加 LoRA，**只用几十段同类镜头运动的视频**训练，就能学会这种运动模式。不同的 MotionLoRA 还可以按权重叠加，得到组合的镜头运动（例如边拉远边平移）。 [[5]](#ref-5)

用 Diffusers 推理（运动模块叫 `MotionAdapter`，MotionLoRA 和图像 LoRA 用同一套 `load_lora_weights` / `set_adapters` 接口）：

```python
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
pipe = AnimateDiffPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",      # 也可以换成任何 SD 1.5 的个性化模型
    motion_adapter=adapter, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config, beta_schedule="linear", clip_sample=False,
    timestep_spacing="linspace", steps_offset=1)

pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out")
pipe.load_lora_weights("path/to/my_style_lora", adapter_name="style")   # 普通的图像 LoRA
pipe.set_adapters(["zoom-out", "style"], adapter_weights=[0.8, 0.7])

frames = pipe("a lighthouse on a cliff at sunset", num_frames=16,
              num_inference_steps=25, guidance_scale=7.5).frames[0]
export_to_gif(frames, "lighthouse.gif")
```

### 7.3 大型视频模型上的 LoRA

2024 年以后的开源视频扩散模型（DiT 结构，参数从几十亿到上百亿）全量微调的代价更高，LoRA 几乎是社区定制它们的唯一现实方式：训练人物一致性、特定画风、特效（比如"爆炸""融化"这类动作效果）。做法和图像上一样，LoRA 加在 Transformer 的注意力（有时也包括前馈层）上；区别是训练数据换成短视频片段加描述，单步显存远大于图像，因此常配合低分辨率、少帧数、gradient checkpointing 和量化底模（9.1 节的 QLoRA 思路）一起用。

> **易错点（视频）**
>
> 1. **MotionLoRA 和运动模块的版本要对应。** 它是加在特定运动模块上的增量，换了运动模块版本，层名或权重分布对不上，效果就会失常。
> 2. **图像 LoRA 强度可能需要调低。** 逐帧都受同一个 LoRA 影响，强度过高时容易出现帧间闪烁或者动作僵硬。
> 3. **训练视频 LoRA 时运动要一致。** MotionLoRA 的几十段视频应该是同一种镜头运动、内容各异；运动杂乱时学到的只是平均的抖动。

## 8. 组合、强度和常见问题

### 8.1 多个 LoRA 叠加

LoRA 是加在权重上的增量，多个 LoRA 叠加就是把增量按权重相加：

$$
W = W_0 + \sum_i s_i \cdot \frac{\alpha_i}{r_i} B_i A_i
$$

例如一个人物 LoRA 加一个风格 LoRA，就能画"这个人物、这种画风"。叠加是线性的，和加载顺序无关；但网络整体是非线性的，**两个 LoRA 改动同一批层时会互相干扰**：各自单独效果都很好，一叠加人物就不像了、或者风格变弱。常用的缓解办法是把各自的强度降到 0.5～0.8，或者在训练时就让一个只管内容、一个只管风格（比如风格 LoRA 只加在部分层上）。

用第 3 节的模块把几个 LoRA 依次合并进同一个模型：

```python
def merge_loras_into(base_model_fn, loras, r=4, alpha=4.0):
    """loras: [(lora_state_dict, scale), ...]；依次注入、加载、设强度、合并"""
    model = base_model_fn()
    for state, scale in loras:
        inject_lora(model, r=r, alpha=alpha)
        model.load_state_dict(state, strict=False)       # 只加载 lora_A / lora_B
        set_lora_scale(model, scale)
        merge_lora(model)                                # 把 s·ΔW 加进 W0，换回普通层
    return model
```

（这里假设各个 LoRA 用同一套注入规则训练，层名能对上。）

### 8.2 调强度

推理时的强度 $$s$$ 是最常用的旋钮：

- $$s = 0$$ 精确等于原模型（第 4 节的实验可以直接看到）；
- $$s$$ 从 0 增到 1，效果连续地从原模型过渡到训练时的效果；
- $$s > 1$$ 是外推，风格更重，但容易出现过饱和、结构崩坏；
- 也可以只在部分去噪步上用 LoRA，或者对不同层用不同强度（社区叫 "LoRA block weight"），细粒度地控制"改构图"还是"改细节"。

### 8.3 训练时的常见问题

| 现象 | 常见原因 | 处理 |
|---|---|---|
| 训练很久也看不出变化 | 学习率太小（用了全量微调的学习率）、$$\alpha$$ 太小、LoRA 没挂上 | 学习率 $$10^{-4}$$ 量级；检查可训练参数数量和 $$\alpha/r$$ |
| 生成结果几乎复制训练图 | 数据太少、步数太多、$$r$$ 太大 | 早停，降 $$r$$，增加数据多样性，写好 caption |
| 换 prompt 改不掉背景或姿势 | 这些因素被绑到了触发词上 | 在 caption 里描述它们；数据里让它们变化起来 |
| 叠加别的 LoRA 后效果变差 | 两个 LoRA 改了同一批层 | 降低各自强度；分层训练 |
| 换个底模就失效 | LoRA 依赖于训练时的底模权重 | 在目标底模（或它的直接祖先）上训练 |

> **易错点（组合）**
>
> 1. **合并顺序不影响结果，但强度会叠加。** 两个强度都为 1 的风格 LoRA 叠在一起，相当于总的风格更新幅度翻倍，容易过头。
> 2. **合并进权重后就"拆不开"了。** 要保留随时调整、随时卸载的灵活性，就不要合并；要部署速度，就合并后另存一份。

## 9. 常见变体与要点回顾

### 9.1 常见变体

| 方法 | 改了什么 | 解决的问题 |
|---|---|---|
| **QLoRA**（2023） | 底模量化成 4 bit（NF4 格式）冻结，LoRA 仍用 16 bit 训练；再加分页优化器 | 显存：单张 48 GB 显卡可以微调 65B 的语言模型 |
| **LoCon**（LyCORIS） | 卷积层也加 LoRA（3.2 节的写法） | 只改注意力时风格类细节学不全 |
| **LoHa / LoKr**（LyCORIS） | $$\Delta W$$ 用两个低秩矩阵的逐元素乘积 / Kronecker 积表示 | 同样参数量下能表示秩更高的 $$\Delta W$$ |
| **DoRA**（2024） | 把权重拆成"幅度 × 方向"，方向用 LoRA 更新，幅度单独学 | 让 LoRA 的学习行为更接近全量微调 |
| **LoRA+**（2024） | $$B$$ 的学习率设得比 $$A$$ 大很多 | $$A$$、$$B$$ 用同一学习率时收敛偏慢 |
| **LCM-LoRA**（2023） | 把"少步采样"能力蒸馏成一个 LoRA | 挂上后 SD 用 4～8 步就能出图，可以和风格 LoRA 叠加 |

最后一行说明 LoRA 不只能学"内容"，也能学"采样行为"：任何能表示成一个小的权重增量的能力，都可以做成 LoRA 来分发和组合。 [[6]](#ref-6)

### 9.2 要点回顾

- LoRA 冻结 $$W_0$$，只学一个低秩增量 $$\Delta W = \frac{\alpha}{r} B A$$；参数从 $$dk$$ 降到 $$r(d + k)$$。
- $$B$$ 初始化为 0、$$A$$ 随机：起点和原模型完全一样，并且两个矩阵都能学。
- 训练完可以合并进权重，推理零开销；不合并时可以随时调强度 $$s$$，$$s = 0$$ 精确回到原模型。
- 它省的是梯度、优化器状态和存储，不省激活；学习率要比全量微调大约一个量级。
- 低秩之所以够用：微调主要是放大原模型已有但未被强调的方向，$$\Delta W$$ 的有效秩本来就很低。
- 在 SD 里默认加在注意力的 q/k/v/out 上；$$r = 4$$ 只有约 80 万参数（U-Net 的 0.09%）。
- 经典用法：DreamBooth 式的人物/物体定制（稀有 token + 先验保留）、风格 LoRA、视频里的 AnimateDiff（图像 LoRA 直接复用，MotionLoRA 学镜头运动）；多个 LoRA 可以按权重叠加。
