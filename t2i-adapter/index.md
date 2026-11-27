---
date: 2026-11-27 07:00:00 -0800
keywords: >-
  T2I-Adapter, spatial conditioning, sketch, canny, depth, pose, segmentation, pixel unshuffle, adapter, frozen U-Net, composable adapters, ControlNet comparison, Stable Diffusion 1.5, SDXL, diffusers, PyTorch.
summary: >-
  T2I-Adapter 怎么用一个只有 7700 万参数、和时间步无关的小网络，给冻结的 Stable Diffusion 加上边缘、深度、姿态这类空间控制：结构（pixel unshuffle + 4 级 ResBlock）、特征加在 U-Net 编码器的哪里、为什么整个采样只算一次、多个 adapter 怎么组合、训练时的三次方时间步采样。给出 SD 1.5 上逐层精确的实现，在 FashionMNIST 小扩散模型上实际训练一个 Canny adapter 并和 ControlNet 对比参数量、效果和推理开销，最后是 Diffusers 的推理和训练写法。
layout: note
title: "T2I-Adapter：用一个轻量小网络控制扩散模型"
description: "How T2I-Adapter adds spatial control to a frozen Stable Diffusion with a small, timestep-independent network: architecture, where features are added, why it runs once per generation, composing adapters, cubic timestep sampling, the exact SD 1.5 adapter, a FashionMNIST experiment against ControlNet, and diffusers usage."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig svg .lane{font-size:14px;font-weight:600}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
</style>

# T2I-Adapter：用一个轻量小网络控制扩散模型

[ControlNet](/deep-learning/controlnet/) 给 Stable Diffusion 加空间控制的办法是：把 U-Net 的前半段整个复制一份来处理条件图。效果很好，但代价也大：SD 1.5 上 ControlNet 有 3.6 亿参数，而且**每个去噪步都要跑一遍**。

T2I-Adapter 问的是：控制信号真的需要这么大的网络吗？它的回答是一个很小的、**和时间步无关**的卷积网络就够了：

1. 条件图（草图、边缘、深度、姿态……）经过一个约 7700 万参数的小网络，得到 4 个分辨率上的特征图；
2. 这 4 个特征图**直接加**到冻结 U-Net 编码器对应分辨率的特征上；
3. 小网络不看 $$z_t$$、不看时间步 $$t$$，所以**整个采样过程只需要算一次**，之后每一步复用。

代价是控制精度一般略低于 ControlNet，好处是训练和推理都便宜得多，多个 adapter 还能直接加权相加组合使用。

本文主要参考 T2I-Adapter 原论文 [[1]](#ref-1) 和官方仓库 [[2]](#ref-2)，Diffusers 的用法参考 [[3]](#ref-3)。正文里用 [n] 标出具体出处，完整列表见文末「参考资料」。

## 0. 全文路线

| 节 | 内容 |
|---|---|
| 1 | 要解决什么问题；条件的三种形态；和 ControlNet 的出发点区别 |
| 2 | 结构和公式：特征加在哪、为什么只算一次、多个 adapter 怎么组合 |
| 3 | SD 1.5 上逐层精确的 T2I-Adapter（7737 万参数）以及接进 U-Net 的写法 |
| 4 | 训练：数据、冻结什么、三次方时间步采样 |
| 5 | 实验：FashionMNIST 上训练一个 Canny adapter，和 ControlNet 对比 |
| 6 | 用 Diffusers 推理、组合多个 adapter、自己写训练一步 |
| 7 | 和 ControlNet 的对比；要点回顾 |

## 1. 要解决什么问题

### 1.1 给"画在哪"一个信号

文本很擅长说"画什么"，但不擅长说"画在哪、什么形状、什么姿势"。给扩散模型加条件，按条件的形态大致分三类：

| 形态 | shape | 例子 | 一般怎么注入 |
|---|---|---|---|
| 全局向量 | `[B, D]` | 类别、CLIP 图像嵌入 | 加到时间嵌入上，或投影成几个 token |
| token 序列 | `[B, L, D]` | 文本、图像 patch 特征 | cross-attention |
| 空间图 | `[B, C, H, W]` | 草图、边缘、深度、姿态、分割 | **按分辨率对齐后加到卷积特征上** |

T2I-Adapter 和 ControlNet 处理的都是第三类：条件图和输出图像素对齐，最自然的做法就是在 U-Net 里"同一个位置"把条件特征加进去。（IP-Adapter 处理的是前两类，见 [IP-Adapter 篇](/deep-learning/ip-adapter/)。）

### 1.2 出发点：预训练模型其实已经"会"了

论文的观察是：SD 本身已经有很强的结构生成能力，只是文本这个接口没法精确地指定结构。所以要学的不是"怎么画出这个结构"，而是"**怎么把外部的结构信号翻译成 U-Net 内部能听懂的特征**"。这个翻译任务比生成本身简单得多，一个小网络就够了。 [[1]](#ref-1)

由此得到两个设计：

- 原模型**完全冻结**，不会因为加了控制而丢掉原来的能力；
- adapter 只是一个从条件图到特征的映射，**不需要知道当前的噪声图和时间步**。

第二点是它和 ControlNet 最本质的区别：ControlNet 的拷贝分支输入里有 $$z_t$$ 和 $$t$$，每步结果都不同；T2I-Adapter 的输出只取决于条件图。

> **易错点**
>
> 1. **"轻量"指的是相对 ControlNet。** SD 1.5 的完整版 adapter 也有 7700 万参数，存成 fp32 约 300 MB，比一个 LoRA 大得多。
> 2. **adapter 和底模是配套的。** 它学的是"往这个 U-Net 的这几层特征里加什么"，换成通道数、结构不同的底模（比如 SD 1.5 换 SDXL）就要重新训练一个；同结构的微调模型（社区的 SD 1.5 微调版）通常可以直接用。

## 2. 结构和公式

### 2.1 整体数据流

<figure class="ddpm-fig">
<svg viewBox="0 0 760 400" role="img" aria-label="T2I-Adapter 整体结构">
  <defs><marker id="ta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker>
  <marker id="tr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#d92d20"/></marker></defs>
  <rect x="8" y="8" width="744" height="170" rx="10" fill="#eef4ff" stroke="#c7d7fe"/>
  <text x="20" y="30" class="lane">冻结的 U-Net 编码器（每个去噪步都要跑）</text>
  <text x="740" y="30" text-anchor="end" class="sm">文本 → cross-attention（不变）</text>
  <rect x="20" y="60" width="80" height="44" rx="7" fill="#fff" stroke="#475467"/><text x="60" y="80" text-anchor="middle">z</text><text x="60" y="96" text-anchor="middle" class="sm">64×64×4</text>
  <rect x="125" y="60" width="110" height="44" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="180" y="80" text-anchor="middle">Down 1</text><text x="180" y="96" text-anchor="middle" class="sm">320 · 64×64</text>
  <rect x="260" y="60" width="110" height="44" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="315" y="80" text-anchor="middle">Down 2</text><text x="315" y="96" text-anchor="middle" class="sm">640 · 32×32</text>
  <rect x="395" y="60" width="110" height="44" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="450" y="80" text-anchor="middle">Down 3</text><text x="450" y="96" text-anchor="middle" class="sm">1280 · 16×16</text>
  <rect x="530" y="60" width="110" height="44" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="585" y="80" text-anchor="middle">Down 4</text><text x="585" y="96" text-anchor="middle" class="sm">1280 · 8×8</text>
  <rect x="665" y="60" width="78" height="44" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="704" y="80" text-anchor="middle">Mid + Up</text><text x="704" y="96" text-anchor="middle" class="sm">→ 预测噪声</text>
  <line x1="100" y1="82" x2="123" y2="82" stroke="#475467" marker-end="url(#ta)"/>
  <line x1="235" y1="82" x2="258" y2="82" stroke="#475467" marker-end="url(#ta)"/>
  <line x1="370" y1="82" x2="393" y2="82" stroke="#475467" marker-end="url(#ta)"/>
  <line x1="505" y1="82" x2="528" y2="82" stroke="#475467" marker-end="url(#ta)"/>
  <line x1="640" y1="82" x2="663" y2="82" stroke="#475467" marker-end="url(#ta)"/>
  <circle cx="180" cy="140" r="10" fill="#fff" stroke="#d92d20"/><text x="180" y="145" text-anchor="middle" style="fill:#d92d20">+</text>
  <circle cx="315" cy="140" r="10" fill="#fff" stroke="#d92d20"/><text x="315" y="145" text-anchor="middle" style="fill:#d92d20">+</text>
  <circle cx="450" cy="140" r="10" fill="#fff" stroke="#d92d20"/><text x="450" y="145" text-anchor="middle" style="fill:#d92d20">+</text>
  <circle cx="585" cy="140" r="10" fill="#fff" stroke="#d92d20"/><text x="585" y="145" text-anchor="middle" style="fill:#d92d20">+</text>
  <line x1="180" y1="130" x2="180" y2="106" stroke="#d92d20" marker-end="url(#tr)"/>
  <line x1="315" y1="130" x2="315" y2="106" stroke="#d92d20" marker-end="url(#tr)"/>
  <line x1="450" y1="130" x2="450" y2="106" stroke="#d92d20" marker-end="url(#tr)"/>
  <line x1="585" y1="130" x2="585" y2="106" stroke="#d92d20" marker-end="url(#tr)"/>
  <text x="196" y="160" class="sm">× ω</text><text x="331" y="160" class="sm">× ω</text><text x="466" y="160" class="sm">× ω</text><text x="601" y="160" class="sm">× ω</text>
  <rect x="8" y="196" width="744" height="196" rx="10" fill="#fffbeb" stroke="#fde68a"/>
  <text x="20" y="218" class="lane">T2I-Adapter（可训练）</text>
  <rect x="20" y="300" width="80" height="44" rx="7" fill="#fff" stroke="#475467"/><text x="60" y="320" text-anchor="middle">条件图</text><text x="60" y="336" text-anchor="middle" class="sm">512×512</text>
  <rect x="20" y="236" width="80" height="44" rx="7" fill="#fff" stroke="#f79009"/><text x="60" y="254" text-anchor="middle" class="sm">unshuffle ×8</text><text x="60" y="270" text-anchor="middle" class="sm">+ conv_in</text>
  <line x1="60" y1="300" x2="60" y2="282" stroke="#475467" marker-end="url(#ta)"/>
  <rect x="125" y="236" width="110" height="60" rx="7" fill="#fff" stroke="#f79009"/><text x="180" y="256" text-anchor="middle">Stage 1</text><text x="180" y="272" text-anchor="middle" class="sm">2 × ResBlock</text><text x="180" y="287" text-anchor="middle" class="sm">320 · 64×64</text>
  <rect x="260" y="236" width="110" height="60" rx="7" fill="#fff" stroke="#f79009"/><text x="315" y="256" text-anchor="middle">Stage 2</text><text x="315" y="272" text-anchor="middle" class="sm">2 × ResBlock</text><text x="315" y="287" text-anchor="middle" class="sm">640 · 32×32</text>
  <rect x="395" y="236" width="110" height="60" rx="7" fill="#fff" stroke="#f79009"/><text x="450" y="256" text-anchor="middle">Stage 3</text><text x="450" y="272" text-anchor="middle" class="sm">2 × ResBlock</text><text x="450" y="287" text-anchor="middle" class="sm">1280 · 16×16</text>
  <rect x="530" y="236" width="110" height="60" rx="7" fill="#fff" stroke="#f79009"/><text x="585" y="256" text-anchor="middle">Stage 4</text><text x="585" y="272" text-anchor="middle" class="sm">2 × ResBlock</text><text x="585" y="287" text-anchor="middle" class="sm">1280 · 8×8</text>
  <line x1="100" y1="258" x2="123" y2="258" stroke="#475467" marker-end="url(#ta)"/>
  <line x1="235" y1="266" x2="258" y2="266" stroke="#475467" marker-end="url(#ta)"/>
  <line x1="370" y1="266" x2="393" y2="266" stroke="#475467" marker-end="url(#ta)"/>
  <line x1="505" y1="266" x2="528" y2="266" stroke="#475467" marker-end="url(#ta)"/>
  <text x="315" y="320" text-anchor="middle" class="sm">Stage 2～4 先 AvgPool ↓2</text>
  <path d="M180,236 L180,151" fill="none" stroke="#d92d20" stroke-width="1.6" marker-end="url(#tr)"/>
  <path d="M315,236 L315,151" fill="none" stroke="#d92d20" stroke-width="1.6" marker-end="url(#tr)"/>
  <path d="M450,236 L450,151" fill="none" stroke="#d92d20" stroke-width="1.6" marker-end="url(#tr)"/>
  <path d="M585,236 L585,151" fill="none" stroke="#d92d20" stroke-width="1.6" marker-end="url(#tr)"/>
  <text x="460" y="350" class="sm">输入只有条件图：不看 z_t，不看时间步 t</text>
  <text x="460" y="368" class="sm">4 个特征算一次，每个去噪步都复用</text>
  <text x="125" y="350" class="sm">红线：F_c 的 4 个特征，× ω 后加进编码器</text>
</svg>
<figcaption>T2I-Adapter 的数据流（按论文图 3 重画，改成 SD 1.5 的具体通道数）。上面是冻结的 U-Net 编码器，下面是 adapter。adapter 的 4 个特征分别加到 4 个下采样 block 内部的特征上，加完之后的特征既往下传，也进入 skip 供解码器使用。</figcaption>
</figure>

### 2.2 公式

记冻结 U-Net 编码器在第 $$i$$ 个尺度上的特征为 $$F_{\text{enc}}^i$$，adapter 是 $$\mathcal{F}_{\text{AD}}$$，条件图是 $$C$$ [[1]](#ref-1)：

$$
F_c = \mathcal{F}_{\text{AD}}(C) = \{F_c^1, F_c^2, F_c^3, F_c^4\},\qquad
\hat F_{\text{enc}}^i = F_{\text{enc}}^i + \omega \cdot F_c^i,\quad i = 1, \dots, 4
$$

$$\omega$$ 是推理时的控制强度（Diffusers 里叫 `adapter_conditioning_scale`），训练时为 1。训练目标还是普通的噪声预测 MSE，只是多了条件 $$C$$：

$$
\mathcal{L} = \mathbb{E}_{z_0,\,t,\,c_{\text{text}},\,C,\,\epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta\big(z_t,\ t,\ c_{\text{text}},\ \mathcal{F}_{\text{AD}}(C)\big)\big\rVert_2^2\Big]
$$

梯度只更新 $$\mathcal{F}_{\text{AD}}$$ 的参数，U-Net、VAE、文本编码器全部冻结。

| 符号 | 含义 | SD 1.5 上的 shape |
|---|---|---|
| $$C$$ | 条件图 | `[B, 3, 512, 512]`（草图、Canny 是 1 通道） |
| $$F_c^1 \dots F_c^4$$ | adapter 输出 | `[B,320,64,64]` `[B,640,32,32]` `[B,1280,16,16]` `[B,1280,8,8]` |
| $$F_{\text{enc}}^i$$ | 第 $$i$$ 个下采样 block 最后一层的输出 | 和 $$F_c^i$$ 相同 |
| $$\omega$$ | 控制强度 | 标量，默认 1 |

### 2.3 具体加在哪一层

SD 1.5 的编码器有 4 个下采样 block，每个 block 里是"ResBlock → Transformer"重复 2 次，然后（前 3 个 block）一个下采样卷积。adapter 的特征加在**每个 block 最后一对 ResBlock/Transformer 的输出上、下采样之前**：

```
Down 1:  Res → Attn → Res → Attn → (+ ω·F_c^1) → skip → Downsample → skip
Down 2:  Res → Attn → Res → Attn → (+ ω·F_c^2) → skip → Downsample → skip
Down 3:  Res → Attn → Res → Attn → (+ ω·F_c^3) → skip → Downsample → skip
Down 4:  Res → Res → (+ ω·F_c^4) → skip
```

这和 ControlNet 很不一样：

| | 加到哪 | 加完之后 |
|---|---|---|
| ControlNet | 12 条 skip 和中间层输出 | 只给解码器用，**不改变编码器继续往下算的特征** |
| T2I-Adapter | 4 个下采样 block 内部的特征 | 既进 skip，**也往下传给更深的编码器层** |

所以 T2I-Adapter 加进去的条件会被后续的编码器层进一步处理，用 4 个特征就能影响整个网络；ControlNet 只改 skip，所以要给每条 skip 都准备一个残差。Diffusers 里这两种注入是两个不同的参数：ControlNet 用 `down_block_additional_residuals`（加到 skip），T2I-Adapter 用 `down_intrablock_additional_residuals`（加到 block 内部）。

### 2.4 为什么只需要算一次

adapter 的输入只有条件图 $$C$$。一次生成过程里 $$C$$ 不变，所以 $$F_c$$ 也不变：

$$
\underbrace{F_c = \mathcal{F}_{\text{AD}}(C)}_{\text{采样前算 1 次}}\quad\longrightarrow\quad
\epsilon_\theta(z_t,\ t,\ c_{\text{text}},\ F_c),\quad t = T, \dots, 1
$$

50 步采样时，T2I-Adapter 的额外开销是 1 次小网络前向；ControlNet 是 50 次（开 CFG 的话再乘 2）"半个 U-Net"的前向。这是论文强调"简单、轻量"的主要原因。

代价也来自这里：adapter 不知道当前噪声有多大，只能给所有时间步同一份引导。ControlNet 可以在高噪声阶段管构图、低噪声阶段管细节，adapter 不行。4.2 节的时间步采样技巧就是为了缓解这个问题。

### 2.5 多个 adapter 组合

因为 adapter 的输出只是加在特征上的增量，多个条件（比如草图 + 颜色，或姿态 + 深度）可以直接加权求和 [[1]](#ref-1)：

$$
F_c = \sum_{k=1}^{K} \omega_k\, \mathcal{F}_{\text{AD}}^k(C_k)
$$

每个 adapter 单独训练，组合时**不需要重新训练**。这在论文里叫 composable controlling。和 [LoRA 叠加](/deep-learning/lora/)一样，网络整体是非线性的，两个条件互相冲突时（比如草图和深度图不一致）效果会变差，权重一般要调小。

> **易错点**
>
> 1. **特征是加在 block 内部，不是加在 skip 上。** 自己实现时如果把 adapter 特征当成 ControlNet 残差加在 skip 上，代码能跑，但条件不会传到更深的编码器层，效果会差很多。在 Diffusers 里就是用错了参数名（见 6.3 节）。
> 2. **adapter 没有零初始化。** 和 ControlNet 不同，论文的 adapter 用普通初始化，训练开始时就会扰动 U-Net 的特征。第 5 节的小实验里，第 0 步的生成结果已经被明显打乱；好处是没有 ControlNet 那种"前几百步不动、然后突然收敛"，控制效果从第一步就开始上升。如果希望训练起点精确等于原模型，可以在每个输出后面加一个零初始化的 1×1 卷积，这是可选的改法，不是论文的原始做法。
> 3. **只算一次是推理时的事。** 训练时每个 batch 的条件图不同，adapter 每步都要前向和反向。

## 3. SD 1.5 上的精确实现

下面是和官方 SD 1.5 完整版 adapter（Diffusers 的 `FullAdapter`）逐层对应的实现。 [[2]](#ref-2)

### 3.1 adapter 本体

第一步是 **pixel unshuffle**：把 $$512 \times 512$$ 的条件图每 $$8 \times 8$$ 个像素块重排成通道，变成 $$64 \times 64 \times 192$$，没有任何参数，但分辨率直接对齐到 latent 的 $$64 \times 64$$。之后是 4 个 stage，每个 stage 是 2 个没有归一化层的残差块：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.block2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x + self.block2(F.relu(self.block1(x)))            # 没有归一化层，也不吃时间步


class AdapterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        self.downsample = nn.AvgPool2d(2, stride=2, ceil_mode=True) if down else None
        self.in_conv = (nn.Conv2d(in_channels, out_channels, 1)
                        if in_channels != out_channels else None)
        self.resnets = nn.Sequential(AdapterResnetBlock(out_channels),
                                     AdapterResnetBlock(out_channels))

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        if self.in_conv is not None:
            x = self.in_conv(x)
        return self.resnets(x)


class FullAdapter(nn.Module):
    """条件图 [B, 3, 512, 512] → 4 个特征：
    [B,320,64,64] [B,640,32,32] [B,1280,16,16] [B,1280,8,8]"""
    def __init__(self, in_channels=3, channels=(320, 640, 1280, 1280), downscale_factor=8):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)    # 512×512×3 → 64×64×192，无参数
        self.conv_in = nn.Conv2d(in_channels * downscale_factor ** 2, channels[0], 3, padding=1)
        self.body = nn.ModuleList([
            AdapterBlock(channels[0], channels[0]),
            AdapterBlock(channels[0], channels[1], down=True),
            AdapterBlock(channels[1], channels[2], down=True),
            AdapterBlock(channels[2], channels[3], down=True),
        ])

    def forward(self, cond):
        x = self.conv_in(self.unshuffle(cond))
        feats = []
        for block in self.body:
            x = block(x)
            feats.append(x)
        return feats
```

```python
adapter = FullAdapter()
print(sum(p.numel() for p in adapter.parameters()))             # 77369280，约 77M，和论文一致
print(sum(p.numel() for p in FullAdapter(in_channels=1).parameters()))   # 77000640（草图、Canny）
with torch.no_grad():
    print([tuple(f.shape) for f in adapter(torch.randn(1, 3, 512, 512))])
# [(1, 320, 64, 64), (1, 640, 32, 32), (1, 1280, 16, 16), (1, 1280, 8, 8)]
```

参数主要在 Stage 3、4 的 $$3 \times 3$$ 卷积上（1280 通道，每个约 1475 万参数）。对比一下：SD 1.5 的 ControlNet 有 3.61 亿参数，U-Net 本身 8.6 亿。

### 3.2 接进 U-Net

用 [SD 篇](/deep-learning/stable-diffusion/)里的 `SD15UNet`，把它的 forward 重写一遍，只多一步：每个下采样 block 的最后一对 ResBlock/Transformer 之后加上 $$\omega \cdot F_c^i$$。

```python
def sd15_forward_with_adapter(unet, sample, timesteps, context, adapter_feats, scale=1.0):
    """SD15UNet 的 forward，只多一步：每个下采样 block 的最后一对 resnet/attention 之后加上 F_c"""
    timesteps = torch.as_tensor(timesteps, device=sample.device).reshape(-1).expand(sample.shape[0])
    temb = unet.time_mlp(sinusoidal_timestep_embedding(timesteps, 320).to(sample.dtype))

    x = unet.conv_in(sample)
    skips = [x]
    for block, feat in zip(unet.down_blocks, adapter_feats):
        for i, resnet in enumerate(block.resnets):
            x = resnet(x, temb)
            if block.attentions is not None:
                x = block.attentions[i](x, context)
            if i == len(block.resnets) - 1:
                x = x + scale * feat                             # F_enc^i ← F_enc^i + ω·F_c^i
            skips.append(x)                                      # 加完再存：skip 里也带着条件
        if block.downsample is not None:
            x = block.downsample(x)
            skips.append(x)

    x = unet.mid_block(x, temb, context)
    for block in unet.up_blocks:
        x = block(x, skips, temb, context)
    return unet.conv_out(F.silu(unet.conv_norm_out(x)))
```

检查两件事：$$\omega = 0$$ 时和原 U-Net 完全一样；$$\omega = 1$$ 时输出被改变。用 $$16 \times 16$$ 的小 latent（对应 $$128 \times 128$$ 的条件图）就够了：

```python
torch.manual_seed(0)
unet = SD15UNet().eval()
z, t, text = torch.randn(1, 4, 16, 16), torch.tensor([500]), torch.randn(1, 77, 768)
cond = torch.randn(1, 3, 128, 128)                              # 16×16 latent 对应 128×128 条件图
with torch.no_grad():
    feats = adapter(cond)                                        # 整个采样只算一次
    ref = unet(z, t, text)
    out0 = sd15_forward_with_adapter(unet, z, t, text, feats, scale=0.0)
    assert torch.allclose(out0, ref, atol=1e-5)                 # ω = 0：等于原模型
    out1 = sd15_forward_with_adapter(unet, z, t, text, feats)
    print((out1 - ref).abs().mean() > 0)                         # tensor(True)
```

采样时，`feats = adapter(cond)` 放在采样循环**外面**，循环里每一步都把同一个 `feats` 传进去。

> **易错点**
>
> 1. **条件图分辨率必须是 latent 的 8 倍。** pixel unshuffle 之后的分辨率要和 latent 一致，4 个特征才能和 4 个 block 对上。生成非 512 的尺寸时，条件图也要 resize 到对应的 8 倍大小，且最好是 64 的倍数（3 次下采样之后不出现奇数）。
> 2. **SDXL 的结构不一样。** SDXL 的 U-Net 只有 3 个分辨率、通道数也不同，对应的 adapter（Diffusers 里的 `FullAdapterXL`）层数和下采样位置都不同，不能拿 SD 1.5 的 adapter 去套。

## 4. 训练

### 4.1 数据和冻结

每种条件单独训练一个 adapter。数据是"图像 + 文本 + 条件图"三元组，其中条件图都是**从图像自动提取**的：

| 条件 | 怎么得到 | 通道 |
|---|---|---|
| 草图（sketch） | 边缘检测模型（如 PiDiNet）提取线条 | 1 |
| Canny | `cv2.Canny` | 1 |
| 深度 | 单目深度估计（如 MiDaS） | 3 |
| 姿态（keypose） | 人体关键点检测，画成骨架图 | 3 |
| 分割 | 语义分割，按类别上色 | 3 |
| 颜色（color） | 图像大幅降采样再放大，得到色块图 | 3 |

除了这些空间条件，论文还有一个 **style adapter**：用 CLIP 图像编码器提取参考图的特征，变成几个 token 拼在文本 token 后面。它不走上面的卷积结构，思路更接近后来的 IP-Adapter（[IP-Adapter 篇](/deep-learning/ip-adapter/)讨论了为什么"拼在文本后面"效果不如解耦的 cross-attention）。

训练时 U-Net、VAE、文本编码器全部冻结，只训练 adapter。因为网络小、而且前向不经过 U-Net 的拷贝，训练成本比 ControlNet 低得多。

### 4.2 非均匀的时间步采样

论文分析了一个现象：**生成的结构主要在采样的早期（高噪声阶段）决定**。只在采样的后半段加入 adapter，控制效果很弱；只在前半段加入，结构就基本定了。 [[1]](#ref-1)

但普通训练时 $$t$$ 是从 $$[0, T)$$ 均匀采样的，高噪声的时间步只占一部分。论文改成用三次函数采样，让 $$t$$ 更多地落在大值上：

$$
u \sim \mathcal{U}(0, 1),\qquad t = \big(1 - u^3\big) \cdot T
$$

$$u^3$$ 偏向 0，所以 $$1 - u^3$$ 偏向 1，$$t$$ 偏向 $$T$$：比如 $$t > T/2$$ 的概率从均匀采样的 0.5 变成 $$0.5^{1/3} \approx 0.79$$。

```python
def sample_timesteps_cubic(batch_size, num_train_timesteps=1000, device="cpu"):
    u = torch.rand(batch_size, device=device)
    t = (1 - u ** 3) * num_train_timesteps                      # 偏向大的 t（高噪声）
    return t.long().clamp_(0, num_train_timesteps - 1)


t = sample_timesteps_cubic(100000)
print((t >= 500).float().mean())                                 # 约 0.79；均匀采样是 0.5
```

这和 2.4 节说的局限是一回事：adapter 只能给所有时间步同一份特征，那就让它优先学好最影响结构的那些时间步。

> **易错点**
>
> 1. **条件图要和图像严格对齐。** 数据增强（随机裁剪、翻转）必须对图像和条件图做完全相同的变换，否则学到的是错位的控制。
> 2. **预处理要和推理时一致。** 训练用 PiDiNet 提的草图，推理时给手绘草图，或者训练时 Canny 阈值 100/200、推理时换成别的阈值，效果都会打折扣。
> 3. **latent 要乘 `scaling_factor`。** 和所有 SD 训练一样，VAE 编码出的 latent 要乘 0.18215 再加噪声（6.3 节的代码），漏了不会报错，但训练出来的 adapter 和底模的分布对不上。

## 5. 实验：在 FashionMNIST 上训练一个 Canny adapter

用和 [ControlNet 篇](/deep-learning/controlnet/)第 4 节**完全相同**的设置，只把 ControlNet 分支换成 T2I-Adapter，这样两者可以直接比：

- 冻结的 base：DDPM 篇里训练好的类别条件 Tiny U-Net（FashionMNIST，$$T = 300$$，基础通道 48，46 万参数），它有两个分辨率（$$28 \times 28$$ 和 $$14 \times 14$$）；
- 条件：OpenCV Canny（阈值 50/150）提取的边缘图；
- 训练：batch 64，AdamW，学习率 $$10^{-4}$$，2000 步，50% 概率丢掉类别标签；时间步用均匀采样（和 ControlNet 实验保持一致）；
- 评估：测试集前 64 张图的边缘图作条件，不给类别，DDIM 20 步，对生成图提取 Canny 边缘，和条件边缘图算 F1（允许 1 像素误差）。不受控时 F1 约 0.64。

### 5.1 Tiny T2I-Adapter

base 只有两个分辨率，adapter 也只要两级：第一级在 $$28 \times 28$$ 上输出 48 通道，第二级 AvgPool 到 $$14 \times 14$$、1×1 卷积升到 96 通道。图很小，不需要 pixel unshuffle：

```python
class TinyT2IAdapter(nn.Module):
    """条件图 [B, 1, H, W] → [f0: [B, c, H, W], f1: [B, 2c, H/2, W/2]]
    对应 Tiny U-Net 的两个分辨率"""
    def __init__(self, cond_channels=1, base_channels=48):
        super().__init__()
        c = base_channels
        self.conv_in = nn.Conv2d(cond_channels, c, 3, padding=1)
        self.stage0 = nn.Sequential(AdapterResnetBlock(c), AdapterResnetBlock(c))
        self.down = nn.AvgPool2d(2)
        self.in_conv1 = nn.Conv2d(c, 2 * c, 1)
        self.stage1 = nn.Sequential(AdapterResnetBlock(2 * c), AdapterResnetBlock(2 * c))

    def forward(self, cond):
        f0 = self.stage0(self.conv_in(cond))                       # [B, c, H, W]
        f1 = self.stage1(self.in_conv1(self.down(f0)))             # [B, 2c, H/2, W/2]
        return [f0, f1]


def tiny_unet_forward_with_adapter(unet, x_t, t, labels, feats, scale=1.0):
    """冻结的 ClassConditionedTinyUNet；在两个分辨率上把 adapter 特征加到编码器特征上"""
    emb = unet.time_mlp(sinusoidal_timestep_embedding(t, unet.time_dim))
    emb = emb + unet.class_embedding(labels)
    skip = unet.encoder(x_t, emb) + scale * feats[0]               # 第 1 级：既进 skip，也往下传
    h = unet.bottleneck(unet.down(skip), emb) + scale * feats[1]   # 第 2 级
    h = unet.attention(h)
    h = torch.cat([unet.up(h), skip], dim=1)
    return unet.output(unet.decoder(h, emb))
```

这个 adapter 有 **236,160** 个参数，ControlNet 实验里的控制分支是 328,433 个。

### 5.2 结果

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/t2i-adapter/f1_vs_controlnet.png" alt="T2I-Adapter 与 ControlNet 的边缘 F1 随训练步数变化">
  <figcaption>边缘 F1 随训练步数的变化。虚线（蓝）是 ControlNet 篇里同样设置下的零初始化 ControlNet；点线是不受控时的水平。</figcaption>
</figure>

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/t2i-adapter/samples.png" alt="T2I-Adapter 训练过程中的生成结果，以及同步数的 ControlNet">
  <figcaption>同一组边缘图、同一组 $$x_T$$、不给类别。中间几行是 adapter 训练到第 N 步的结果，倒数第二行是训练 2000 步的 ControlNet。</figcaption>
</figure>

| | ControlNet | T2I-Adapter |
|---|---|---|
| 可训练参数 | 328,433 | 236,160 |
| 第 0 步 F1 | 0.62（等于 base） | 0.71（已被扰动） |
| 2000 步 F1 | 0.849 | 0.820 |
| 生成 64 张、DDIM 20 步（CPU） | 1.90 s | 1.45 s |

（不受控的水平是 0.64；只跑冻结 base 是 1.21 s。）

- **起步方式完全不同。** ControlNet 前 300 步 F1 不动（零卷积，输出就是 base），然后突然上升；adapter 没有零初始化，第 0 步就已经改变了输出，F1 从第一步开始涨，前 500 步一直领先。但第 0 步的样本也说明了代价：普通初始化的 adapter 一开始会把生成结果打乱（上图第二行）。
- **最终精度 ControlNet 更高。** 600 步以后 ControlNet 反超，2000 步时 0.849 对 0.820。从样本看，adapter 对鞋、上衣的轮廓跟得不错，但第 3 列的长裤没能画成裤子，ControlNet 画对了（第 4 列两者都没画好）。这和论文、社区的经验一致：T2I-Adapter 控制得住大结构，细节精度不如 ControlNet。原因也很直接：adapter 不知道当前的 $$z_t$$ 和 $$t$$，只能给所有时间步同一份特征。
- **推理开销小。** adapter 特征只算一次时，比 base 多 0.24 s，ControlNet 多 0.69 s。如果把 adapter 写成每步都重新计算（错误但常见的写法），时间会变成 2.38 s，比 ControlNet 还慢：这个小 adapter 在 $$28 \times 28$$ 全分辨率上算卷积，相对这个很小的 base 并不便宜。在 SD 上 U-Net 远大于 adapter，"只算一次"的优势会更明显。

推理时把特征乘 $$\omega$$ 扫一遍（训练 2000 步的 adapter）：

| $$\omega$$ | 0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| 边缘 F1（64 张） | 0.62 | 0.72 | 0.82 | 0.90 | 0.94 |

$$\omega = 0$$ 精确退回 base，$$\omega$$ 越大越贴边缘图，和 ControlNet 的 control scale 行为一致。$$\omega > 1$$ 时 F1 更高，但这是训练时没见过的设定，在 SD 上常见的副作用是纹理发硬、颜色过饱和。

> **易错点（实验）**
>
> 1. **对比要用同一套设置。** 这里 base、数据、条件、学习率、batch、丢标签比例、评估集和 $$x_T$$ 都和 ControlNet 实验完全一致，只换了控制模块，曲线才有可比性。
> 2. **单个种子、小模型。** 具体的步数和 F1 差距会随种子浮动，这里只说明趋势：adapter 起步快、上限略低、推理便宜。
> 3. **测速时把 adapter 放在采样循环外面。** 不然测出来的是"每步都算一次 adapter"，正好抹掉了它最大的优点。

## 6. 用 Diffusers

### 6.1 推理

SDXL 上的 Canny adapter [[3]](#ref-3)：

```python
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter

# 1. 条件图：Canny 边缘，单通道
image = np.array(Image.open("input.png").convert("RGB").resize((1024, 1024)))
cond_image = Image.fromarray(cv2.Canny(image, 100, 200))       # 'L' 模式，[1024, 1024]

# 2. 加载
adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", adapter=adapter, torch_dtype=torch.float16,
).to("cuda")

# 3. 生成
out = pipe(
    prompt="a futuristic building at sunset, high quality",
    negative_prompt="blurry, low quality",
    image=cond_image,
    num_inference_steps=30,
    guidance_scale=7.5,                    # CFG，管文本
    adapter_conditioning_scale=0.8,        # ω，管条件图
    adapter_conditioning_factor=1.0,       # 在前百分之多少的去噪步上使用 adapter
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
```

SD 1.5 用 `StableDiffusionAdapterPipeline`，加载方式相同，换成 SD 1.5 的 adapter 权重和底模即可：

```python
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter

adapter = T2IAdapter.from_pretrained("<SD 1.5 的 T2I-Adapter 权重>", torch_dtype=torch.float16)
pipe = StableDiffusionAdapterPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", adapter=adapter, torch_dtype=torch.float16,
).to("cuda")
print(adapter.config.in_channels)          # 条件图通道数要和它一致：草图/Canny 是 1，深度/姿态是 3
```

`adapter_conditioning_factor` 对应 4.2 节的观察：设成 0.5 表示只在前一半（高噪声）的去噪步上加 adapter 特征，后一半让模型自由生成细节。结构基本不变，纹理和细节往往更自然。

### 6.2 组合多个 adapter

```python
from diffusers import MultiAdapter

adapters = MultiAdapter([adapter_sketch, adapter_depth])         # 两个独立训练的 adapter
pipe = StableDiffusionAdapterPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", adapter=adapters, torch_dtype=torch.float16,
).to("cuda")
out = pipe(
    prompt="a cozy living room",
    image=[sketch_image, depth_image],                           # 和 adapter 一一对应
    adapter_conditioning_scale=[0.8, 0.6],                       # 2.5 节的 ω_k
).images[0]
```

### 6.3 自己写训练一步

和 [ControlNet 篇](/deep-learning/controlnet/) 7.2 节的训练一步相比，有三处不同：adapter 不吃 $$z_t$$ 和 $$t$$；时间步用三次方采样；特征通过 `down_intrablock_additional_residuals` 传给 U-Net。

```python
import torch
import torch.nn.functional as F
from diffusers import T2IAdapter

# adapter = T2IAdapter(in_channels=1, channels=[320, 640, 1280, 1280],
#                      num_res_blocks=2, downscale_factor=8, adapter_type="full_adapter")
# vae / text_encoder / unet 都 requires_grad_(False)；优化器只放 adapter.parameters()

def t2i_adapter_training_step(batch, vae, text_encoder, unet, adapter, noise_scheduler):
    # batch["pixel_values"]: 目标图像 [B, 3, 512, 512]，范围 [-1, 1]
    # batch["conditioning_pixel_values"]: 条件图 [B, 1, 512, 512]，范围 [0, 1]
    # batch["input_ids"]: [B, 77]
    with torch.no_grad():
        latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
        latents = latents * vae.config.scaling_factor           # 0.18215，别漏
        text = text_encoder(batch["input_ids"])[0]               # [B, 77, 768]

    noise = torch.randn_like(latents)
    num_t = noise_scheduler.config.num_train_timesteps
    t = sample_timesteps_cubic(latents.shape[0], num_t, latents.device)   # 4.2 节
    noisy = noise_scheduler.add_noise(latents, noise, t)

    feats = adapter(batch["conditioning_pixel_values"])          # 4 个特征；不吃 noisy 和 t
    pred = unet(noisy, t, encoder_hidden_states=text,            # 冻结，但不能 no_grad
                down_intrablock_additional_residuals=[f for f in feats]).sample
    return F.mse_loss(pred.float(), noise.float())
```

> **易错点**
>
> 1. **参数名是 `down_intrablock_additional_residuals`。** 早期版本的 Diffusers 用 `down_block_additional_residuals` 传 T2I-Adapter 特征，靠特征个数来区分是 adapter 还是 ControlNet，现在这条路已经废弃。用错参数名时，特征会被当作 ControlNet 残差加到 skip 上（2.3 节的区别），效果明显变差。
> 2. **每次调用都传一个新列表。** U-Net 内部会从这个列表里逐个 `pop` 特征，同一个列表第二次传进去就是空的。采样循环里复用特征时，传 `[f.clone() for f in feats]` 或至少 `list(feats)`；pipeline 内部就是这么做的。
> 3. **CFG 时特征也要复制。** 有条件和无条件两个分支都要加 adapter 特征（条件图不随 prompt 变），batch 拼接时特征也要沿 batch 维复制一份。

## 7. 和 ControlNet 对比、要点回顾

### 7.1 对比

| | ControlNet | T2I-Adapter |
|---|---|---|
| 结构 | 复制 U-Net 编码器 + 中间层 | 独立的小卷积网络（pixel unshuffle + 4 级 ResBlock） |
| 可训练参数（SD 1.5） | 3.61 亿 | 0.77 亿 |
| 输入 | 条件图 + $$z_t$$ + $$t$$ + 文本 | 只有条件图 |
| 每次生成的前向次数 | 每个去噪步 1 次（CFG 时 2 次） | 1 次 |
| 注入位置 | 12 条 skip + 中间层 | 4 个下采样 block 内部 |
| 初始化 | 拷贝权重 + 零卷积 | 普通初始化 |
| 控制精度 | 更高，尤其是细节 | 结构对，细节略弱 |
| 能否按时间步调节 | 能（每步结果不同） | 不能（同一份特征），只能选在哪些步上加 |
| 组合 | 多个 ControlNet 残差相加 | 多个 adapter 特征相加 |

怎么选：显存、速度敏感（比如移动端、实时预览、同时挂好几个条件）用 T2I-Adapter；对结构精度要求高（姿态、线稿上色）用 ControlNet。

### 7.2 要点回顾

- T2I-Adapter 冻结整个 SD，只训练一个把条件图翻译成 4 个多尺度特征的小网络，特征直接加到 U-Net 编码器 4 个下采样 block 内部：$$\hat F_{\text{enc}}^i = F_{\text{enc}}^i + \omega F_c^i$$。
- SD 1.5 上的结构：pixel unshuffle ×8 → conv → 4 个 stage（每个 2 个无归一化的 ResBlock，通道 320/640/1280/1280），共 7737 万参数。
- adapter 不看 $$z_t$$ 和 $$t$$，所以**整个采样只需要算一次**，推理开销几乎为零；代价是所有时间步共用同一份引导。
- 结构主要在高噪声阶段决定，所以训练用 $$t = (1 - u^3) T$$ 偏向大的时间步；推理时可以只在前一部分步上加 adapter。
- 多个 adapter 可以直接按权重相加组合，不需要联合训练。
- Diffusers 里特征走 `down_intrablock_additional_residuals`，不要和 ControlNet 的 `down_block_additional_residuals` 混用。

## 参考资料

- <a id="ref-1"></a>**[1] 主论文**：Mou et al. *T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models*. AAAI 2024. [arxiv.org/abs/2302.08453](https://arxiv.org/abs/2302.08453)。结构（论文图 3）、多尺度特征相加、组合控制、三次方时间步采样都来自这篇。
- <a id="ref-2"></a>**[2] 官方仓库**：TencentARC/T2I-Adapter [github.com/TencentARC/T2I-Adapter](https://github.com/TencentARC/T2I-Adapter)。各种条件的 adapter 权重和训练代码。
- <a id="ref-3"></a>**[3] Diffusers 文档**：Hugging Face Diffusers: T2I-Adapter [huggingface.co/docs/diffusers/en/using-diffusers/t2i_adapter](https://huggingface.co/docs/diffusers/en/using-diffusers/t2i_adapter)。第 6 节的推理和 `MultiAdapter`。
