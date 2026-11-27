---
date: 2026-11-27 00:30:00 -0800
keywords: >-
  ControlNet, spatial conditioning, zero convolution, trainable copy, frozen U-Net, Canny edge, depth, pose, segmentation, control scale, classifier-free guidance, guess mode, Multi-ControlNet, Stable Diffusion 1.5, diffusers, PyTorch.
summary: >-
  ControlNet 怎么在不改动预训练扩散模型的前提下，加上边缘、深度、姿态这类空间控制：冻结原 U-Net，复制它的前半段做可训练分支，再用零初始化卷积把控制信号逐层加回去。从公式、零卷积为什么能学，到一个可运行的 Tiny ControlNet 和 FashionMNIST 上的真实训练实验（突然收敛、零初始化对比、控制强度），再到 SD 1.5 上逐层精确的 ControlNet（参数量 3.61 亿）、CFG 与 guess mode、Diffusers 的训练和推理。
layout: note
title: "ControlNet：给扩散模型加上空间控制"
description: "How ControlNet adds edge, depth and pose control to a frozen diffusion model: trainable copy, zero convolutions, a runnable tiny ControlNet trained on FashionMNIST, the exact SD 1.5 ControlNet, CFG and guess mode, and diffusers usage."
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

# ControlNet：给扩散模型加上空间控制

[Stable Diffusion](/deep-learning/stable-diffusion/) 用文本告诉模型"画什么"，但很难用文字说清"画在哪、摆成什么姿势、轮廓长什么样"。ControlNet 补的就是这一块：给一张**和输出同样大小的条件图**（边缘图、深度图、人体骨架、分割图……），让生成结果在空间结构上照着它来。

它的做法可以概括成三句话：

1. **冻结**一个已经训练好的扩散 U-Net，一个参数都不动；
2. 把这个 U-Net 的**前半段（编码器和中间层）复制一份**，这份拷贝是可训练的，额外接收条件图；
3. 拷贝在每个分辨率上的输出经过**零初始化的卷积**，作为残差加回冻结 U-Net 的 skip 连接和中间层。

因为第 3 步的卷积初始全为 0，刚开始训练时整个模型和原模型**完全一样**；训练只是在这个起点上，逐渐学会"按条件图修改各层特征"。下面先讲清这三点为什么这样设计，再用一个能跑的小模型真的训练一遍，最后给出 SD 1.5 上逐层精确的结构和实际用法。

## 0. 全文路线

| 节 | 内容 |
|---|---|
| 1 | ControlNet 要解决什么问题，为什么不直接微调或拼接输入 |
| 2 | 结构和公式：冻结主干 + 可训练拷贝 + 零卷积；零卷积为什么能学 |
| 3 | 最小实现：在 [DDPM 篇](/deep-learning/ddpm-from-scratch/)的 Tiny U-Net 上加 ControlNet（含两张手绘结构图） |
| 4 | 实验：在 FashionMNIST 上用边缘图真的训练一个，观察"突然收敛"、零初始化的作用、控制强度 |
| 5 | SD 1.5 上的精确 ControlNet：条件编码器、12 + 1 个零卷积、参数量 |
| 6 | 推理：control scale、CFG、guess mode、多个 ControlNet、分时段控制 |
| 7 | 用 Diffusers 训练和推理 |
| 8 | 不同的条件图怎么准备 |
| 9 | 相关方法、要点回顾 |

**符号约定**：

| 符号 | 含义 | 代码 | shape（SD 1.5，512×512） |
|---|---|---|---|
| $$z_t$$ | 带噪 latent（Tiny 版里是带噪图像 $$x_t$$） | `x_t` / `sample` | `[B, 4, 64, 64]` |
| $$t$$ | 时间步 | `t` | `[B]` |
| $$c_{\text{text}}$$ | 文本条件 | `encoder_hidden_states` | `[B, 77, 768]` |
| $$c_{\text{f}}$$ | 空间条件图（边缘、深度、姿态……） | `cond` / `controlnet_cond` | `[B, 3, 512, 512]`，范围 $$[0, 1]$$ |
| $$r_i$$ | 第 $$i$$ 条 skip 上的控制残差 | `down_residuals[i]` | 和第 $$i$$ 条 skip 同 shape |
| $$r_{\text{mid}}$$ | 中间层的控制残差 | `mid_residual` | `[B, 1280, 8, 8]` |
| $$s$$ | 控制强度 | `control_scale` / `controlnet_conditioning_scale` | 标量 |

## 1. ControlNet 要解决什么问题

### 1.1 文本管"画什么"，条件图管"怎么摆"

| 条件类型 | 控制的是 | 例子 |
|---|---|---|
| 类别、文本 | 内容和语义 | "a dog running on the beach" |
| 空间条件图 | 布局和几何 | 人在画面左边、举着右手；物体的轮廓；前后景的深度；每块区域属于什么类别 |

条件图和输出图**像素对齐**：条件图上 $$(i, j)$$ 处的信息，约束的就是输出图 $$(i, j)$$ 附近的内容。这种约束用文字几乎表达不了，而用卷积网络处理却很自然。

### 1.2 为什么不直接微调整个模型

最直接的想法是收集"条件图–图像"配对数据，把条件图拼到 U-Net 的输入上（例如 `torch.cat([z_t, cond], dim=1)`），再微调整个网络。问题在于：

- **数据量差距太大。** Stable Diffusion 是在几十亿张图文对上训练的，一个 Canny 或姿态数据集通常只有几万到几百万对。拿小数据微调整个 8.6 亿参数的 U-Net，很容易**过拟合、灾难性遗忘**，画质和对文本的理解一起变差。
- **改了输入层。** 拼接会改变第一层卷积的输入通道数，这一层的新权重只能随机初始化，训练一开始就破坏了原模型。
- **条件只在入口进入一次。** 要经过很多层才能影响到深层特征，信号可能被冲淡。

ControlNet 的目标是：**原模型的能力一点不丢，同时学会听条件图的话**。办法是把"保留能力"和"学习控制"分给两套参数。

### 1.3 什么冻结、什么训练

| 模块 | 状态 | 说明 |
|---|---|---|
| VAE、文本编码器 | 冻结 | 和 SD 训练时一样 |
| 原 U-Net（全部） | **冻结**，但仍然要前向 | 最终的噪声预测还是它输出的 |
| 可训练拷贝（编码器 + 中间层） | 训练 | 用原 U-Net 的权重初始化 |
| 条件编码器 | 训练 | 新建，把条件图变成特征 |
| 零卷积 | 训练 | 新建，初始化为 0 |

"冻结"只是优化器不更新这些参数，**前向计算一步都不少**，而且梯度还要从 loss 穿过冻结的解码器，传回控制残差（见 3.4 的易错点）。

## 2. 结构和公式

### 2.1 一个 block 上的 ControlNet

把网络里的一个模块（例如一组 ResBlock）记作 $$\mathcal{F}(\cdot;\Theta)$$，输入特征 $$x$$，输出 $$y = \mathcal{F}(x;\Theta)$$。ControlNet 对它做三件事：冻结 $$\Theta$$；复制一份参数 $$\Theta_c$$ 作为可训练拷贝；新增两个零卷积 $$\mathcal{Z}(\cdot;\Theta_{z1})$$、$$\mathcal{Z}(\cdot;\Theta_{z2})$$。输出变成：

$$
y_c = \mathcal{F}(x;\Theta) + \mathcal{Z}\Big(\mathcal{F}\big(x + \mathcal{Z}(c;\Theta_{z1});\ \Theta_c\big);\ \Theta_{z2}\Big)
$$

- 第一项：冻结的原模块，原样保留；
- 括号里：可训练拷贝看到的是"原输入 + 经过零卷积的条件"；
- 第二项整体：拷贝的输出再经过一个零卷积，作为残差加回去。

初始化时 $$\Theta_{z1} = \Theta_{z2} = 0$$，于是 $$\mathcal{Z}(\cdot) = 0$$，$$y_c = y$$：**加上 ControlNet 之后，模型在第 0 步和原模型逐位相同**。

### 2.2 放到整个 U-Net 上

对 U-Net 来说，"模块"就是整个编码器加中间层。可训练拷贝在每个分辨率产生一个特征，各自过一个零卷积，加到冻结 U-Net **对应的 skip 连接**和**中间层输出**上：

<figure class="ddpm-fig">
<svg viewBox="0 0 760 420" role="img" aria-label="ControlNet 整体结构">
  <defs><marker id="ca" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker>
  <marker id="cr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#d92d20"/></marker></defs>
  <rect x="8" y="8" width="744" height="170" rx="10" fill="#eef4ff" stroke="#c7d7fe"/>
  <text x="20" y="30" class="lane">冻结的 U-Net（参数不更新）</text>
  <rect x="20" y="70" width="80" height="40" rx="7" fill="#fff" stroke="#475467"/><text x="60" y="94" text-anchor="middle">带噪 latent</text>
  <rect x="120" y="70" width="70" height="40" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="155" y="94" text-anchor="middle">conv_in</text>
  <rect x="210" y="70" width="90" height="40" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="255" y="94" text-anchor="middle">4 个 Down</text>
  <rect x="320" y="120" width="80" height="40" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="360" y="144" text-anchor="middle">Mid</text>
  <rect x="440" y="70" width="90" height="40" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="485" y="94" text-anchor="middle">4 个 Up</text>
  <rect x="560" y="70" width="80" height="40" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="600" y="94" text-anchor="middle">conv_out</text>
  <rect x="660" y="70" width="84" height="40" rx="7" fill="#ecfdf3" stroke="#12b76a"/><text x="702" y="94" text-anchor="middle">预测噪声</text>
  <line x1="100" y1="90" x2="118" y2="90" stroke="#475467" marker-end="url(#ca)"/>
  <line x1="190" y1="90" x2="208" y2="90" stroke="#475467" marker-end="url(#ca)"/>
  <path d="M255,110 L255,140 L318,140" fill="none" stroke="#475467" marker-end="url(#ca)"/>
  <path d="M400,140 L485,140 L485,112" fill="none" stroke="#475467" marker-end="url(#ca)"/>
  <line x1="530" y1="90" x2="558" y2="90" stroke="#475467" marker-end="url(#ca)"/>
  <line x1="640" y1="90" x2="658" y2="90" stroke="#475467" marker-end="url(#ca)"/>
  <path d="M255,70 L255,50 L485,50 L485,68" fill="none" stroke="#475467" stroke-dasharray="5 4" marker-end="url(#ca)"/>
  <text x="370" y="35" text-anchor="middle" class="sm">12 条 skip（+ 控制残差）</text>
  <circle cx="370" cy="50" r="9" fill="#fff" stroke="#d92d20"/><text x="370" y="54" text-anchor="middle" style="fill:#d92d20">+</text>
  <circle cx="420" cy="140" r="9" fill="#fff" stroke="#d92d20"/><text x="420" y="144" text-anchor="middle" style="fill:#d92d20">+</text>
  <rect x="8" y="196" width="744" height="216" rx="10" fill="#fffbeb" stroke="#fde68a"/>
  <text x="20" y="218" class="lane">ControlNet（可训练）</text>
  <rect x="20" y="330" width="80" height="40" rx="7" fill="#fff" stroke="#475467"/><text x="60" y="354" text-anchor="middle">条件图</text>
  <rect x="120" y="330" width="90" height="40" rx="7" fill="#fff" stroke="#f79009"/><text x="165" y="348" text-anchor="middle">条件编码器</text><text x="165" y="363" text-anchor="middle" class="sm">末层零卷积</text>
  <rect x="120" y="260" width="70" height="40" rx="7" fill="#fff" stroke="#f79009"/><text x="155" y="279" text-anchor="middle">conv_in</text><text x="155" y="293" text-anchor="middle" class="sm">拷贝</text>
  <circle cx="232" cy="280" r="9" fill="#fff" stroke="#475467"/><text x="232" y="284" text-anchor="middle">+</text>
  <rect x="255" y="260" width="90" height="40" rx="7" fill="#fff" stroke="#f79009"/><text x="300" y="279" text-anchor="middle">4 个 Down</text><text x="300" y="293" text-anchor="middle" class="sm">拷贝</text>
  <rect x="370" y="260" width="80" height="40" rx="7" fill="#fff" stroke="#f79009"/><text x="410" y="279" text-anchor="middle">Mid</text><text x="410" y="293" text-anchor="middle" class="sm">拷贝</text>
  <rect x="300" y="215" width="140" height="30" rx="7" fill="#fef3f2" stroke="#d92d20"/><text x="370" y="235" text-anchor="middle">13 个零卷积</text>
  <path d="M60,110 L60,240 L155,240 L155,258" fill="none" stroke="#475467" stroke-dasharray="5 4" marker-end="url(#ca)"/>
  <text x="66" y="234" class="sm">同一个 latent</text>
  <line x1="190" y1="280" x2="221" y2="280" stroke="#475467" marker-end="url(#ca)"/>
  <path d="M210,350 L232,350 L232,291" fill="none" stroke="#475467" marker-end="url(#ca)"/>
  <line x1="100" y1="350" x2="118" y2="350" stroke="#475467" marker-end="url(#ca)"/>
  <line x1="241" y1="280" x2="253" y2="280" stroke="#475467" marker-end="url(#ca)"/>
  <line x1="345" y1="280" x2="368" y2="280" stroke="#475467" marker-end="url(#ca)"/>
  <line x1="300" y1="260" x2="320" y2="247" stroke="#475467" marker-end="url(#ca)"/>
  <line x1="410" y1="260" x2="400" y2="247" stroke="#475467" marker-end="url(#ca)"/>
  <path d="M340,215 L340,62 L362,55" fill="none" stroke="#d92d20" stroke-width="1.6" marker-end="url(#cr)"/>
  <path d="M420,215 L420,151" fill="none" stroke="#d92d20" stroke-width="1.6" marker-end="url(#cr)"/>
  <text x="470" y="290" class="sm">时间步、文本条件两边都输入</text>
  <text x="470" y="308" class="sm">拷贝的输出只当残差用，不经过解码器</text>
  <text x="470" y="384" class="sm">红线：控制残差 × control scale</text>
</svg>
<figcaption>ControlNet 的整体数据流。上面的 U-Net 全部冻结；下面的拷贝只有编码器和中间层，没有解码器。红色是经过零卷积的残差，加到 skip 和中间层输出上。</figcaption>
</figure>

写成公式：设冻结 U-Net 的第 $$i$$ 条 skip 是 $$h_i$$、中间层输出是 $$h_{\text{mid}}$$，ControlNet 输出残差 $$r_1, \dots, r_{12}, r_{\text{mid}}$$，则解码器看到的是

$$
\tilde h_i = h_i + s \cdot r_i\quad (i = 1, \dots, 12),\qquad
\tilde h_{\text{mid}} = h_{\text{mid}} + s \cdot r_{\text{mid}}
$$

其他一切照旧：训练目标还是预测噪声的 MSE，采样器还是 DDPM / DDIM / DPM-Solver，**ControlNet 只改变"噪声预测网络"这个黑盒内部的计算**。

$$
\mathcal{L} = \mathbb{E}_{z_0,\,t,\,c_{\text{text}},\,c_{\text{f}},\,\epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta(z_t,\ t,\ c_{\text{text}},\ c_{\text{f}})\big\rVert_2^2\Big]
$$

> **易错点**
>
> 1. **残差加在 skip 和中间层上，不加在编码器的主路径上。** 冻结 U-Net 的编码器照常计算，控制残差只影响"传给解码器的东西"。加到编码器主路径上会改变后面所有层的输入，这已经不是 ControlNet 的结构了。
> 2. **拷贝不需要解码器。** ControlNet 的任务是产生修正量，最终输出仍由冻结的解码器负责。所以拷贝只是"半个 U-Net"，参数量约为原 U-Net 的 42%（第 5 节）。
> 3. **条件图只有一张，但 ControlNet 每一步都要重新算。** 残差依赖当前的 $$z_t$$ 和 $$t$$，不能只算一次缓存起来。推理时间因此增加约一半。

### 2.3 为什么要零初始化

如果最后一层用普通的随机初始化，训练第 0 步拷贝就会输出随机残差，加到 13 个位置上：

$$
\tilde h_i = h_i + \underbrace{\text{随机噪声}}_{\text{立刻破坏原模型}}
$$

原模型的输出被打乱，训练的前一段时间要先花在"把这些随机扰动消掉"上，而且大梯度会一路传回拷贝，把从原模型继承来的好权重也搅乱。零初始化保证起点就是原模型，训练只在它的基础上一点点加控制。4.3 节的实验会直接对比两种初始化。

零初始化不是数学上的必需品：任何"新增路径初始输出为 0"的设计都能达到同样目的，例如零初始化的标量门控、LoRA 里把 $$B$$ 矩阵初始化为 0。关键是那个性质，不是卷积本身。

### 2.4 全零的权重为什么还能学

直觉上会担心：权重全是 0，梯度会不会也是 0，永远学不动？看一个 1×1 卷积（就是逐位置的线性层）$$y = W x + b$$，链式法则给出

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y}\, x^\top,\qquad
\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial y},\qquad
\frac{\partial \mathcal{L}}{\partial x} = W^\top \frac{\partial \mathcal{L}}{\partial y}
$$

- $$\partial\mathcal{L}/\partial W$$ 只取决于输入 $$x$$ 和上游梯度，和 $$W$$ 本身无关。$$x$$ 是拷贝输出的特征，不为 0，所以**零卷积在第一步就有梯度**，一步之后就不再是 0。
- 但 $$\partial\mathcal{L}/\partial x = W^\top(\cdot)$$ 在 $$W = 0$$ 时恰好是 0：**第一步时，零卷积前面的所有参数（拷贝、条件编码器、输入端的零卷积）都拿不到梯度**。

所以训练的真实顺序是：第 1 步只有输出端的零卷积在动；它们变成非零之后，梯度才开始流进拷贝；再之后输入端的零卷积和条件编码器才开始学。3.4 节的代码里可以直接验证这一点。

> **易错点**
>
> 1. **零初始化要同时清零 weight 和 bias。** 只清零 weight，bias 的默认初始化会让输出变成一个非零常数，第 0 步就不再等于原模型。
> 2. **"零卷积学得慢"是个误解。** 它在第一步就更新；论文里拷贝的学习速度和普通微调相当。真正慢的是"学会听条件图"这件事本身，见 4.2 的"突然收敛"。

## 3. 最小实现：Tiny ControlNet

这一节在 [DDPM 篇](/deep-learning/ddpm-from-scratch/) 5.4 节的 Tiny U-Net 上加 ControlNet。Tiny U-Net 只有一层下采样，于是 skip 只有一条、中间层一个，ControlNet 只需要输出两个残差。结构和 SD 上的完全一样，只是规模小到可以在 CPU 上训练。

回顾 Tiny U-Net 的前向（`c` 是基础通道数）：

```text
x_t [B,C,H,W] → encoder → skip [B,c,H,W] → down → bottleneck → h [B,2c,H/2,W/2]
              → up → 和 skip 拼接 [B,2c,H,W] → decoder → output → ε̂ [B,C,H,W]
```

### 3.1 零卷积

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_zero_conv(in_channels, out_channels=None, kernel_size=1):
    """weight 和 bias 都为 0 的卷积：[B, C_in, H, W] → [B, C_out, H, W]，初始输出恒为 0"""
    layer = nn.Conv2d(in_channels, out_channels or in_channels, kernel_size,
                      padding=kernel_size // 2)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer
```

### 3.2 控制分支

<figure class="sd-img">
<a href="/deep-learning/assets/controlnet/tiny_control_branch.png"><img src="/deep-learning/assets/controlnet/tiny_control_branch.png" alt="Tiny ControlNet 控制分支流程图"></a>
<figcaption>控制分支：条件图经过条件编码器和零卷积，加到 x_t 上，再走一遍复制来的 encoder、下采样和 bottleneck；两个输出各过一个零卷积，就是要加回 U-Net 的两个残差。图中 time_cond 来自共享的时间编码；下面的代码给拷贝复制了一份自己的时间 MLP，和 SD 上的 ControlNet 一致。</figcaption>
</figure>

对应 2.1 节的公式：$$\mathcal{Z}(c;\Theta_{z1})$$ 是 `cond_zero(cond_encoder(cond))`，$$\mathcal{F}(\cdot;\Theta_c)$$ 是复制来的 `encoder → down → bottleneck`，$$\mathcal{Z}(\cdot;\Theta_{z2})$$ 是 `skip_zero`、`mid_zero`。

```python
class TinyControlBranch(nn.Module):
    def __init__(self, base_unet, condition_channels=1, hint_channels=16):
        super().__init__()
        c = base_unet.encoder.conv1.out_channels        # 基础通道数
        in_ch = base_unet.encoder.conv1.in_channels     # x_t 的通道数
        self.time_dim = base_unet.time_dim
        # Θ_c：复制原 U-Net 的前半段（数值相同、参数独立）
        self.time_mlp = copy.deepcopy(base_unet.time_mlp)
        self.encoder = copy.deepcopy(base_unet.encoder)
        self.down = copy.deepcopy(base_unet.down)
        self.bottleneck = copy.deepcopy(base_unet.bottleneck)
        # 条件编码器：普通卷积，正常初始化
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, hint_channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hint_channels, hint_channels, 3, padding=1), nn.SiLU(),
        )
        self.cond_zero = make_zero_conv(hint_channels, in_ch)     # Θ_z1：条件 → x_t 的通道数
        self.skip_zero = make_zero_conv(c)                        # Θ_z2：两个输出各一个
        self.mid_zero = make_zero_conv(2 * c)

    def forward(self, x_t, t, cond):                 # cond: [B, C_cond, H, W]，和 x_t 同分辨率
        time_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))
        h = x_t + self.cond_zero(self.cond_encoder(cond))         # x + Z(c)
        skip = self.encoder(h, time_emb)                          # [B, c, H, W]
        mid = self.bottleneck(self.down(skip), time_emb)          # [B, 2c, H/2, W/2]
        return self.skip_zero(skip), self.mid_zero(mid)           # 两个残差，初始都是 0
```

### 3.3 和冻结的 U-Net 合起来

<figure class="sd-img">
<a href="/deep-learning/assets/controlnet/tiny_controlnet_framework.png"><img src="/deep-learning/assets/controlnet/tiny_controlnet_framework.png" alt="Tiny ControlNet 整体流程图"></a>
<figcaption>Tiny ControlNet 的完整数据流。橙色是输入，绿色是输出，紫色是模块和运算，蓝色是中间 tensor。上半部分是冻结的 U-Net，下半部分是可训练的控制分支：ctrl_zero_skip 乘 ctrl_scale 后加到 skip 上（在下采样之前取出的那份特征），再和上采样结果拼接送进 decoder；ctrl_zero_mid 加到 bottleneck 的输出上。</figcaption>
</figure>

外层模型把 U-Net 的前向拆开，在两个位置各插一次加法：

```python
class TinyControlNet(nn.Module):
    def __init__(self, base_unet, condition_channels=1):
        super().__init__()
        # 顺序很重要：先复制，再冻结（见易错点 1）
        self.control = TinyControlBranch(base_unet, condition_channels)
        self.base = base_unet.requires_grad_(False).eval()

    def train(self, mode=True):                      # 训练模式只作用于控制分支
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x_t, t, cond, control_scale=1.0):
        r_skip, r_mid = self.control(x_t, t, cond)                   # 可训练分支：两个残差
        b = self.base
        time_emb = b.time_mlp(sinusoidal_timestep_embedding(t, b.time_dim))
        skip = b.encoder(x_t, time_emb)                              # [B, c, H, W]
        h = b.attention(b.bottleneck(b.down(skip), time_emb))        # [B, 2c, H/2, W/2]
        h = h + control_scale * r_mid                                # h̃_mid = h_mid + s·r_mid
        skip = skip + control_scale * r_skip                         # h̃_skip = h_skip + s·r_skip
        h = torch.cat([b.up(h), skip], dim=1)
        return b.output(b.decoder(h, time_emb))                      # ε̂，[B, C, H, W]
```

> **易错点**
>
> 1. **先 `deepcopy` 再冻结。** `copy.deepcopy` 会把 `requires_grad` 一起复制。如果先对 base 调 `requires_grad_(False)`，再复制，拷贝出来的 encoder、down、bottleneck 也全是冻结的，真正在训练的只剩条件编码器和零卷积，ControlNet 退化成一个很弱的适配器。**这个错误不会报错**，loss 也会下降，只是控制效果差很多。建好之后用 `sum(p.numel() for p in model.parameters() if p.requires_grad)` 核对一下可训练参数量。
> 2. **冻结 ≠ `torch.no_grad()`。** 冻结 U-Net 的前向不能整体包在 `no_grad` 里：残差 $$r$$ 要经过冻结的 decoder 才影响 loss，梯度必须从 loss 穿过 decoder 传回 $$r$$。只有和残差无关的计算（VAE 编码、文本编码）可以放进 `no_grad`。
> 3. **`model.train()` 会把冻结的 base 也切到训练模式。** 如果 base 里有 Dropout 或 BatchNorm，冻结的模型在训练时就会表现得和原模型不一样。重写 `train()`，让 base 始终 `eval()`。
> 4. **拷贝要有自己的时间 MLP。** 手绘图里两边共用一个冻结的时间编码，也能工作；但 SD 上的 ControlNet 复制的是整个前半段，包括时间嵌入层，拷贝可以学自己的时间响应。

### 3.4 训练一步

训练和 DDPM 完全一样，只是多传一个条件图，优化器里只放可训练参数：

```python
def controlnet_training_step(model, optimizer, x0, cond, sched):
    model.train()                                                  # base 仍是 eval（见上）
    t = torch.randint(0, sched["betas"].shape[0], (x0.shape[0],), device=x0.device)   # [B]
    noise = torch.randn_like(x0)                                   # ε
    a = sched["sqrt_alpha_bars"][t].view(-1, 1, 1, 1)
    s = sched["sqrt_one_minus_alpha_bars"][t].view(-1, 1, 1, 1)
    x_t = a * x0 + s * noise                                       # x_t = √ᾱ x0 + √(1−ᾱ) ε
    loss = F.mse_loss(model(x_t, t, cond), noise)                  # ‖ε − ε_θ(x_t, t, c)‖²
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                              # 梯度穿过冻结的 decoder 回到残差
    optimizer.step()
    return loss.item()


# 只把可训练参数交给优化器
# optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
```

用一个随机初始化的 Tiny U-Net 就能验证 2.3、2.4 节的几个结论：

```python
torch.manual_seed(0)
base = TinyUNet(in_channels=1, base_channels=32, time_dim=128).eval()
model = TinyControlNet(base)
x_t, t = torch.randn(4, 1, 28, 28), torch.randint(0, 300, (4,))
cond = (torch.rand(4, 1, 28, 28) > 0.8).float()

# 1. 第 0 步：加了 ControlNet 的输出和原模型逐位相同
with torch.no_grad():
    assert torch.equal(model(x_t, t, cond), base(x_t, t))

# 2. 先复制后冻结：可训练的只有控制分支，base 一个都没有
assert all(not p.requires_grad for p in model.base.parameters())
assert all(p.requires_grad for p in model.control.parameters())

# 3. 第一次反向：只有输出端的零卷积有梯度，它前面的参数梯度都是 0
F.mse_loss(model(x_t, t, cond), torch.randn_like(x_t)).backward()
grad = {n: p.grad.abs().sum().item() for n, p in model.control.named_parameters()}
assert grad["skip_zero.weight"] > 0 and grad["mid_zero.weight"] > 0
assert grad["encoder.conv1.weight"] == 0 and grad["cond_zero.weight"] == 0
```

### 3.5 采样

采样器一行不用改，只要把"噪声预测网络"换成带条件图的版本。条件图在整条反向链里保持不变：

```python
@torch.no_grad()
def sample_controlnet_ddim(model, cond, sched, num_steps=50, control_scale=1.0):
    model.eval()
    steps = ddim_timesteps(sched["betas"].shape[0], num_steps)      # DDIM 篇 6 节
    x = torch.randn(cond.shape[0], 1, *cond.shape[-2:])             # x_T ~ N(0, I)
    for i, t_index in enumerate(steps):
        prev = steps[i + 1] if i + 1 < len(steps) else -1
        t = torch.full((x.shape[0],), t_index, dtype=torch.long)
        eps = model(x, t, cond, control_scale)      # 唯一的变化：多了 cond 和 scale
        abar_t = sched["alpha_bars"][t_index]
        abar_s = sched["alpha_bars"][prev] if prev >= 0 else abar_t.new_tensor(1.0)
        x0_hat = (x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()    # x̂0
        x = abar_s.sqrt() * x0_hat + (1 - abar_s).sqrt() * eps      # η = 0 的 DDIM 一步
    return x
```

## 4. 实验：在 FashionMNIST 上训练一个 Canny ControlNet

拿 DDPM 篇里训练好的类别条件 Tiny U-Net（FashionMNIST，$$T = 300$$，基础通道 48，46 万参数）做冻结的 base，用 OpenCV 的 Canny（阈值 50/150）从每张训练图提取边缘图作为条件，训练 3.2 节的控制分支（33 万可训练参数，包括复制的类别嵌入）：

- batch 64，AdamW，学习率 $$10^{-4}$$，3000 步（CPU 上约 12 分钟）；
- 和论文一样，训练时以 50% 的概率**丢掉文本条件**，这里换成丢掉类别标签（用 ∅ 代替），逼着 ControlNet 只看边缘图也能认出是什么东西；
- 评估：取测试集前 64 张图的边缘图作条件，不给类别，DDIM 20 步生成，再对生成图提取 Canny 边缘，和条件边缘图算 F1（允许 1 像素误差）。

作为参照，真实图片和它自己的边缘图 F1 = 1.0；而**完全不受控**的生成图拿去和随便一张别的边缘图比，F1 也有约 0.64（衣服大致都在画面中间，边缘总会碰巧对上一部分）。所以下面要看的是 F1 **比 0.64 高出多少**，而不是它的绝对值。

### 4.1 突然收敛

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/controlnet/samples_over_steps.png" alt="训练过程中同一组边缘图、同一组初始噪声的生成结果">
  <figcaption>第一行是条件边缘图，最后一行是原图；中间每行是训练到第 N 步时、用同一个 $$x_T$$、不给类别生成的结果。</figcaption>
</figure>

几个现象：

- **第 0 步和第 200 步几乎一模一样**，都等于冻结 base 的无条件生成。这正是零卷积的效果：训练开始时 ControlNet 对输出没有任何影响，base 的生成能力原样保留，不会因为加了一个新分支就先被"打坏"。
- **400 到 600 步之间突然学会**：鞋、长裤先对上轮廓，之后其余样本陆续跟上。论文里把这叫 *sudden convergence*（SD 上一般在 1 万步以内出现，前面几千步看起来"什么都没学会"，然后某一步突然开始听条件图）。原因是零卷积的权重要先从 0 长到一定大小，残差才开始实质影响输出；在那之前 loss 几乎不动。
- 3000 步后轮廓基本都对上了，但纹理（"Lee" 字样、格子衬衫）没有出现：边缘图只给了"在哪里有边"，Tiny base 本身也画不出这么细的纹理。

### 4.2 零初始化 vs 随机初始化

把两个输出零卷积 `skip_zero`、`mid_zero` 换成普通随机初始化（`cond_zero` 仍是零），其余完全不变，再训 1500 步：

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/controlnet/f1_curve.png" alt="零初始化与随机初始化的边缘 F1 随训练步数变化">
  <figcaption>边缘 F1 随训练步数的变化。虚线：不受控时的水平（生成图和不匹配的边缘图比较）。</figcaption>
</figure>

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/controlnet/step0_zero_vs_random.png" alt="第 0 步时零初始化与随机初始化的生成结果">
  <figcaption>第 0 步（还没训练）的生成结果：零初始化完全等于 base；随机初始化已经被一些随机残差扰动了。</figcaption>
</figure>

- 零初始化在约 300 步开始起飞，随机初始化要到约 750 步，**晚了一倍多**；1500 步时零初始化 0.84，随机初始化 0.83，后期差不多能追上。
- 在这么小的模型上，随机初始化的初始扰动不大，图没有被完全打坏，所以差距主要体现在"起飞时间"上。在 SD 这种大模型上，13 路随机残差同时注入 U-Net 的每一层 skip，初始输出会直接变成噪声或色块，前期的梯度主要花在"把残差压回去"上，这正是论文选择零卷积的原因。
- 注意这只是单个随机种子、CPU 上的小实验，具体步数会随种子浮动，这里只说明趋势。

### 4.3 control scale

推理时把残差乘一个系数 $$s$$（6.1 节），用训练好的 3000 步模型、同一组 $$x_T$$ 扫 $$s$$：

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/controlnet/control_scale_sweep.png" alt="不同 control scale 下的生成结果">
  <figcaption>同一组边缘图、同一个 $$x_T$$，只改 control scale。</figcaption>
</figure>

| control scale | 0 | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| 边缘 F1（8 张） | 0.68 | 0.72 | 0.78 | 0.88 | 0.94 | 0.97 |

- $$s = 0$$ 精确退化回 base（第一行和 4.1 里的第 0 步一致），$$s$$ 越大越贴边缘图。
- $$s > 1$$ 是训练时没见过的设定：这里边缘更贴了，但亮度和质感开始变"硬"（裤子变细、颜色变暗）。在 SD 上，scale 过大常见的表现是过饱和、纹理发脆，所以实践里一般在 0.5 到 1.2 之间调。

> **易错点（实验）**
> 1. **F1 别看绝对值。** 不受控时 F1 就有约 0.64，要和"不匹配边缘图"这个基线比；不做这个对比，很容易以为第 0 步就已经"有 60% 的控制效果"。
> 2. **前几百步 loss 和样本都不动，不代表代码错了。** 这是零卷积 + 突然收敛的正常表现；判断代码对不对，用 3.4 节的检查（初始输出等于 base、第一步梯度只落在输出零卷积上）。
> 3. **评估时要固定 $$x_T$$ 和条件图。** 不然不同步数之间的差异会被采样噪声淹没。
> 4. **类别丢弃不能省。** 训练时一直给类别的话，ControlNet 会偷懒依赖类别信息，推理时不给类别就控不住；这对应论文里 50% 置空 prompt 的做法（7.2 节）。

## 5. SD 1.5 上的精确 ControlNet

Tiny 版和 SD 1.5 版只有三处不同：U-Net 是 latent 空间里 4 层的大 U-Net；条件图比 latent 大 8 倍，需要一个会下采样的条件编码器；残差不是 2 个，而是 12 + 1 个。下面的代码直接复用 [Stable Diffusion 篇](/deep-learning/stable-diffusion/) 第 6 节的 `SD15UNet` 和它的各个 block。

### 5.1 条件编码器：把 512×512 的条件图降到 64×64

条件图是像素空间的 `[B, 3, 512, 512]`，而 U-Net 在 `[B, 4, 64, 64]` 的 latent 上工作。ControlNet 用一个很小的卷积网络把条件图降 8 倍，输出 320 通道，加到拷贝的 `conv_in` 输出上（对应 2.1 节公式里的 $$x + \mathcal{Z}(c)$$）。注意它**不经过 VAE**：条件图直接在像素空间被这个小网络编码。

```text
cond [B,3,512,512] → conv 3→16 → (conv 16→16, conv 16→32 stride 2) → [B,32,256,256]
                   → (conv 32→32, conv 32→96 stride 2) → [B,96,128,128]
                   → (conv 96→96, conv 96→256 stride 2) → [B,256,64,64]
                   → 零卷积 3×3 256→320 → [B,320,64,64]
```

```python
class ControlNetConditioningEmbedding(nn.Module):
    """条件图 [B, 3, 8h, 8w] → [B, 320, h, w]：3 次 stride 2 把分辨率降到 latent 大小"""
    def __init__(self, cond_channels=3, block_channels=(16, 32, 96, 256), out_channels=320):
        super().__init__()
        self.conv_in = nn.Conv2d(cond_channels, block_channels[0], 3, padding=1)
        blocks = []
        for c_in, c_out in zip(block_channels[:-1], block_channels[1:]):
            blocks.append(nn.Conv2d(c_in, c_in, 3, padding=1))              # 同分辨率
            blocks.append(nn.Conv2d(c_in, c_out, 3, padding=1, stride=2))   # 分辨率减半
        self.blocks = nn.ModuleList(blocks)
        # 最后一层零初始化
        self.conv_out = make_zero_conv(block_channels[-1], out_channels, kernel_size=3)

    def forward(self, cond):
        h = F.silu(self.conv_in(cond))
        for block in self.blocks:
            h = F.silu(block(h))
        return self.conv_out(h)                       # 初始化时恒为 0
```

### 5.2 ControlNet 本体：13 个零卷积

SD 1.5 的 U-Net 编码器往栈里压 12 条 skip（SD 篇 6.5 节的表），ControlNet 就给每条 skip 配一个 1×1 零卷积，通道数一一对应，再给中间层配一个：

| 残差 | 来源（拷贝里的位置） | shape（512×512） |
|---|---|---|
| 1 | `conv_in`（已加上条件） | `[B, 320, 64, 64]` |
| 2、3、4 | DownBlock 1 的两层、下采样 | `[B, 320, 64, 64]` ×2，`[B, 320, 32, 32]` |
| 5、6、7 | DownBlock 2 的两层、下采样 | `[B, 640, 32, 32]` ×2，`[B, 640, 16, 16]` |
| 8、9、10 | DownBlock 3 的两层、下采样 | `[B, 1280, 16, 16]` ×2，`[B, 1280, 8, 8]` |
| 11、12 | DownBlock 4 的两层 | `[B, 1280, 8, 8]` ×2 |
| mid | MidBlock | `[B, 1280, 8, 8]` |

```python
class SD15ControlNet(nn.Module):
    def __init__(self, unet, cond_channels=3):
        super().__init__()
        # 1. 复制 U-Net 的前半段：时间 MLP、conv_in、4 个 DownBlock、MidBlock
        self.time_mlp = copy.deepcopy(unet.time_mlp)
        self.conv_in = copy.deepcopy(unet.conv_in)
        self.down_blocks = copy.deepcopy(unet.down_blocks)
        self.mid_block = copy.deepcopy(unet.mid_block)
        # 2. 条件编码器（新建，最后一层零初始化）
        self.cond_embedding = ControlNetConditioningEmbedding(cond_channels)
        # 3. 12 条 skip 各一个零卷积，通道和 U-Net 的 skip 一一对应；mid 再一个
        skip_channels = [320] * 4 + [640] * 3 + [1280] * 5
        self.zero_convs = nn.ModuleList(
            [make_zero_conv(c) for c in skip_channels])
        self.mid_zero_conv = make_zero_conv(1280)

    def forward(self, sample, timesteps, encoder_hidden_states, cond,
                scale=1.0):
        timesteps = torch.as_tensor(timesteps, device=sample.device)
        timesteps = timesteps.reshape(-1).expand(sample.shape[0])
        temb = self.time_mlp(sinusoidal_timestep_embedding(timesteps, 320).to(sample.dtype))
        # x + Z(c)：条件在 conv_in 之后加入
        x = self.conv_in(sample) + self.cond_embedding(cond)
        skips = [x]
        for block in self.down_blocks:
            x, new_skips = block(x, temb, encoder_hidden_states)       # 文本也送进拷贝
            skips.extend(new_skips)
        x = self.mid_block(x, temb, encoder_hidden_states)
        down_residuals = [scale * zc(s) for zc, s in zip(self.zero_convs, skips)]   # r_1 … r_12
        mid_residual = scale * self.mid_zero_conv(x)                                 # r_mid
        return down_residuals, mid_residual
```

冻结的 U-Net 只需要在前向里多两处加法。Diffusers 的 `UNet2DConditionModel` 就是用 `down_block_additional_residuals` 和 `mid_block_additional_residual` 两个参数接收它们：

```python
def unet_forward_with_control(unet, sample, timesteps, encoder_hidden_states,
                              down_residuals=None, mid_residual=None):
    """SD 篇 6.8 节的 SD15UNet.forward，只多了两处加法"""
    timesteps = torch.as_tensor(timesteps, device=sample.device)
    timesteps = timesteps.reshape(-1).expand(sample.shape[0])
    temb = unet.time_mlp(sinusoidal_timestep_embedding(timesteps, 320).to(sample.dtype))
    x = unet.conv_in(sample)
    skips = [x]
    for block in unet.down_blocks:
        x, new_skips = block(x, temb, encoder_hidden_states)
        skips.extend(new_skips)
    if down_residuals is not None:
        skips = [h + r for h, r in zip(skips, down_residuals)]        # h̃_i = h_i + s·r_i
    x = unet.mid_block(x, temb, encoder_hidden_states)
    if mid_residual is not None:
        x = x + mid_residual                                          # h̃_mid = h_mid + s·r_mid
    for block in unet.up_blocks:
        x = block(x, skips, temb, encoder_hidden_states)
    return unet.conv_out(F.silu(unet.conv_norm_out(x)))
```

检查 shape 和零初始化（用 16×16 的小 latent，条件图相应是 128×128）：

```python
unet = SD15UNet().eval()
controlnet = SD15ControlNet(unet)            # 先复制
unet.requires_grad_(False)                   # 再冻结
print(sum(p.numel() for p in controlnet.parameters()))     # 361279120

z = torch.randn(1, 4, 16, 16)
context = torch.randn(1, 77, 768)
cond = torch.rand(1, 3, 128, 128)            # 条件图是 latent 的 8 倍大
with torch.no_grad():
    down_res, mid_res = controlnet(z, 500, context, cond)
    print([tuple(r.shape[1:]) for r in down_res])   # (320,16,16)×3, (320,8,8), (640,8,8)×2, ...
    same = torch.equal(unet_forward_with_control(unet, z, 500, context, down_res, mid_res),
                       unet(z, 500, context))
print(same)                                  # True：第 0 步和原 U-Net 完全一样
```

参数量：

| 部分 | 参数量 |
|---|---|
| 复制的前半段（时间 MLP、conv_in、4 个 DownBlock、MidBlock） | 348,712,960 |
| 条件编码器 | 1,086,480 |
| 13 个零卷积 | 11,479,680 |
| **合计** | **361,279,120**（约为 U-Net 8.6 亿的 42%） |

> **易错点**
>
> 1. **条件图不进 VAE。** 条件图直接以像素形式（`[0, 1]`）进入 ControlNet 自己的条件编码器；只有目标图像要经过 VAE 编码成 latent。尺寸必须正好是 latent 的 8 倍，否则加到 `conv_in` 输出上时对不齐。
> 2. **ControlNet 也看文本。** 拷贝里的 cross-attention 照样接收 `encoder_hidden_states`。所以同一张边缘图，prompt 不同，残差也不同。
> 3. **残差的个数、顺序、通道、分辨率必须和 U-Net 的 skip 一一对应。** 想换一个层数或通道不同的 base（例如 SDXL），ControlNet 必须从那个 base 重新复制、重新训练，不能只改 ControlNet 的配置。SD 1.5 的 ControlNet 也不能直接接到 SD 2.x 上（文本维度从 768 变成 1024）。
> 4. **最低分辨率的 DownBlock 4 没有 attention，它的两个残差也来自 ResBlock**，这和 base U-Net 完全一致，因为拷贝就是照着它复制的。

## 6. 推理：control scale、CFG 与它们的组合

### 6.1 control scale 和 CFG 是两个独立的旋钮

| | 控制什么 | 作用位置 | 典型值 |
|---|---|---|---|
| control scale $$s$$ | 条件图的约束强度 | 残差乘 $$s$$ 再加回 U-Net | 0.5～1.5（默认 1） |
| CFG scale $$w$$ | 文本的作用强度 | 两次噪声预测的线性组合 | 7.5 |

两者同时使用时，默认做法是**有条件和无条件两次 U-Net 前向都加上同一份控制残差**：

$$
\hat\epsilon = \epsilon_\theta(z_t, t, c_{\text{neg}}, c_{\text{f}}) + w\,\big(\epsilon_\theta(z_t, t, c_{\text{text}}, c_{\text{f}}) - \epsilon_\theta(z_t, t, c_{\text{neg}}, c_{\text{f}})\big)
$$

于是条件图同时影响两个分支，CFG 放大的只是"文本带来的差异"，空间结构由两边共同的残差决定。

### 6.2 Guess mode：没有 prompt 时怎么办

如果完全不给 prompt（只给条件图），上式两个分支的文本都是空串，CFG 无从放大，ControlNet 也失去了文本提示。论文给的方案（Diffusers 里叫 `guess_mode=True`）是：

1. **只给有条件分支加残差**，无条件分支不加，这样 CFG 放大的就是"条件图带来的差异"；
2. 但这样放大太猛，于是**按分辨率给残差加权**：越浅（分辨率越高）的残差权重越小。论文的权重是 $$w_i = 64 / h_i$$ 的思路（$$h_i$$ 是该层的边长）；Diffusers 的实现是从 0.1 到 1.0 按对数均匀分布的 13 个系数，依次乘到 12 个 skip 残差和中间层残差上。

```python
# guess mode 下的残差权重（Diffusers 的做法）：浅层小、深层大
scales = torch.logspace(-1, 0, 13)          # [0.1, 0.12, ..., 1.0]
# down_residuals = [r * s for r, s in zip(down_residuals, scales[:12])]
# mid_residual = mid_residual * scales[12]
# 并且只加到 CFG 的有条件那一半 batch 上
```

这时 ControlNet 要自己从条件图"猜"出内容（例如从一张房间的深度图猜出这是卧室），所以叫 guess mode。

### 6.3 只在一段时间步里施加控制

早期的反向步骤决定整体布局，后期决定细节。Diffusers 的 `control_guidance_start` / `control_guidance_end` 让 ControlNet 只在采样进度的一个区间里起作用（例如 0～0.6：先把结构定下来，后面放开让模型自由补细节）：

```python
def control_scale_at(step_index, num_steps, scale, start=0.0, end=1.0):
    progress = step_index / num_steps                # 0 = 第一步（噪声最大），接近 1 = 最后一步
    return scale if start <= progress < end else 0.0
```

### 6.4 多个 ControlNet

同时用姿态、深度、边缘时，每种条件用自己的 ControlNet，**同一位置的残差加权求和**后再加回 U-Net：

$$
\tilde h_i = h_i + \sum_k s_k\, r_i^{(k)},\qquad
\tilde h_{\text{mid}} = h_{\text{mid}} + \sum_k s_k\, r_{\text{mid}}^{(k)}
$$

```python
def combine_residuals(controlnets, conds, scales, sample, t, text):
    total_down, total_mid = None, None
    for net, cond, s in zip(controlnets, conds, scales):
        down, mid = net(sample, t, text, cond, scale=s)              # 每个已经乘了自己的 s_k
        total_down = down if total_down is None else [a + b for a, b in zip(total_down, down)]
        total_mid = mid if total_mid is None else total_mid + mid
    return total_down, total_mid
```

> **易错点**
>
> 1. **多个条件图必须来自同一个空间布局。** 姿态说人在左边、边缘图说人在右边，残差互相抵消，结果往往两边都不像，还容易出伪影。冲突时调低次要条件的 $$s_k$$，或者让它只在部分时间步起作用。
> 2. **control scale 太大会"描边"。** 结构跟得很死，但纹理和光影变差，甚至把条件图的线条直接画进图里；太小则不听条件图。$$s$$ 不改变参数，可以每次推理随意调。
> 3. **每多一个 ControlNet，每一步就多一次"半个 U-Net"的前向。** 两个 ControlNet 时，推理计算量大约是原来的两倍。

## 7. 用 Diffusers 训练和推理

### 7.1 推理

```python
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)

# 1. 准备条件图：Canny 边缘，单通道复制成 3 通道
#    范围 [0, 255] 的 PIL 图（pipeline 内部会除以 255）
image = np.array(Image.open("input.png").convert("RGB").resize((512, 512)))
edges = cv2.Canny(image, 100, 200)                          # [512, 512]，0 或 255
cond_image = Image.fromarray(np.stack([edges] * 3, axis=-1))

# 2. 加载：SD 1.5 的完整仓库 + 一个 Canny ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)   # 20 步左右就够

# 3. 生成
out = pipe(
    prompt="a futuristic building at sunset, high quality",
    negative_prompt="blurry, low quality",
    image=cond_image,
    num_inference_steps=20,
    guidance_scale=7.5,                       # CFG，管文本
    controlnet_conditioning_scale=1.0,        # control scale，管条件图
    control_guidance_start=0.0, control_guidance_end=1.0,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
```

多个 ControlNet 时，`controlnet` 传一个列表，`image` 和 `controlnet_conditioning_scale` 也传相同长度的列表，pipeline 内部就是 6.4 节的加权求和。

### 7.2 训练一步

和 SD 的训练一步（SD 篇第 7 节）相比，只多了 ControlNet 的前向，以及把它的输出传给冻结的 U-Net：

```python
import torch
import torch.nn.functional as F
from diffusers import ControlNetModel

# controlnet = ControlNetModel.from_unet(unet)      # 从 U-Net 复制前半段，零卷积自动初始化为 0
# vae / text_encoder / unet 都 requires_grad_(False)
# 优化器只放 controlnet.parameters()

def controlnet_sd_training_step(batch, vae, text_encoder, unet, controlnet,
                                noise_scheduler):
    # batch["pixel_values"]: 目标图像 [B, 3, 512, 512]，范围 [-1, 1]
    # batch["conditioning_pixel_values"]: 条件图 [B, 3, 512, 512]，范围 [0, 1]
    # batch["input_ids"]: [B, 77]，其中一部分 caption 已被换成空串（见易错点 2）
    # 下面这部分和残差无关，可以 no_grad
    with torch.no_grad():
        latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        text = text_encoder(batch["input_ids"])[0]  # [B, 77, 768]

    noise = torch.randn_like(latents)
    num_t = noise_scheduler.config.num_train_timesteps
    t = torch.randint(0, num_t, (latents.shape[0],), device=latents.device)
    noisy = noise_scheduler.add_noise(latents, noise, t)             # z_t

    down_res, mid_res = controlnet(
        noisy, t, encoder_hidden_states=text,
        controlnet_cond=batch["conditioning_pixel_values"], return_dict=False)
    pred = unet(noisy, t, encoder_hidden_states=text,                # 冻结，但不能 no_grad
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res).sample
    return F.mse_loss(pred.float(), noise.float())                   # SD 1.5 是 ε-prediction
```

完整训练（多卡、混合精度、梯度累积、断点续训、验证）建议直接用 Diffusers 官方的 ControlNet 训练脚本，它的核心就是上面这一步。

> **易错点**
>
> 1. **目标图和条件图的数值范围不同。** 目标图要进 VAE，归一化到 $$[-1, 1]$$；条件图进 ControlNet 自己的编码器，Diffusers 约定是 $$[0, 1]$$。弄反了不会报错，只会学不好。
> 2. **训练时随机把一部分 caption 换成空串。** 论文用的比例是 50%（官方脚本里是 `--proportion_empty_prompts`）。这会逼 ControlNet 只靠条件图也能理解画面内容，推理时对 prompt 不完整、不准确更鲁棒，也让 guess mode 成为可能。
> 3. **"突然收敛"。** 论文观察到模型不是逐渐学会听条件图的，而是在某一步（通常不到 1 万步）突然开始遵守条件图，之前生成的图看上去和条件图无关（4.2 节在小模型上复现了这个现象）。所以训练前期看不到效果不要急着调参，用固定的验证条件图和种子定期出图观察。
> 4. **学习率要小。** 拷贝是从好权重开始的，论文和官方脚本用 $$10^{-5}$$ 左右；太大会把继承来的特征很快破坏掉。
> 5. **显存。** ControlNet 有 3.6 亿可训练参数，AdamW 还要为每个参数存两份状态。显存不够时依次考虑：梯度检查点、梯度累积、混合精度、8-bit Adam、更小的 batch。

## 8. 不同的条件图

ControlNet 的结构和训练目标对所有条件都一样，每种条件训练一个自己的 ControlNet 权重。不同的只是**条件图怎么得到、怎么预处理**：

| 条件 | 怎么得到 | 通道 | 预处理要点 | 约束的是 |
|---|---|---|---|---|
| Canny 边缘 | Canny 算子 | 1（常复制成 3） | 阈值决定细节多少；推理和训练的阈值风格要接近 | 轮廓、局部边缘 |
| 深度 | 单目深度估计（如 MiDaS）或传感器 | 1 | 归一化方式（逐图 min-max、近处亮还是远处亮）必须和训练一致 | 前后关系、整体几何 |
| 人体姿态 | 关键点检测（如 OpenPose）后画成骨架图 | 3 | 骨架颜色编码就是关节的身份，不能乱改 | 人物姿势 |
| 分割图 | 语义分割（如 ADE20K 类别） | 3（颜色编码）或 K（one-hot） | 缩放只能用最近邻，双线性会插出不存在的类别 | 每块区域是什么 |
| 涂鸦 / 线稿 | 用户手绘，或 HED、线稿提取 | 1 | 训练时要对线条做随机加粗、断裂等增强，适应手绘的不规则 | 粗略形状 |
| 法线图 | 法线估计 | 3 | 坐标系约定（哪个通道朝哪）必须一致 | 表面朝向、局部几何 |

```python
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def canny_condition(image_uint8, low=100, high=200):
    """[H, W, 3] uint8 → [1, 3, H, W] float，范围 [0, 1]"""
    edges = cv2.Canny(image_uint8, low, high)                       # [H, W]，0 或 255
    edges = torch.from_numpy(edges).float() / 255.0
    return edges[None, None].repeat(1, 3, 1, 1)


def depth_condition(depth):
    """[B, 1, H, W] 任意尺度的深度 → 逐图归一化到 [0, 1]"""
    d_min = depth.amin(dim=(2, 3), keepdim=True)
    d_max = depth.amax(dim=(2, 3), keepdim=True)
    return (depth - d_min) / (d_max - d_min + 1e-6)


def segmentation_condition(labels, num_classes, size):
    """[B, H, W] 的类别下标 → [B, K, h, w] 的 one-hot；缩放必须用最近邻"""
    labels = F.interpolate(labels[:, None].float(), size=size, mode="nearest")[:, 0].long()
    return F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
```

> **易错点**
>
> 1. **训练和推理用同一个预处理器。** 用 MiDaS 深度训练的 ControlNet，推理时换成另一个深度估计器，深度的尺度和细节风格都不同，效果会明显下降。
> 2. **条件图的错误会原样传给结果。** 姿态检测把手腕检测错了，生成的手就是错的；ControlNet 会忠实地服从一张错的条件图。
> 3. **条件图分辨率要和输出一致**（SD 1.5 是 512×512，即 latent 的 8 倍），pipeline 会自动缩放，但缩放方式（尤其是分割图）要注意。

## 9. 相关方法与要点回顾

### 9.1 相关方法

| 方法 | 条件 | 注入方式 | 特点 |
|---|---|---|---|
| ControlNet | 空间条件图 | 复制前半个 U-Net，13 个零卷积残差 | 控制最强；参数和推理开销最大（+42% 参数，每步多半个 U-Net） |
| T2I-Adapter | 空间条件图 | 一个小卷积网络产生多尺度特征，加到 U-Net 编码器上 | 约 7700 万参数；特征和时间步无关，整条采样链只算一次 |
| ControlNet-XS、ControlLoRA | 空间条件图 | 更小的控制分支，或在拷贝上用 LoRA | 用少得多的参数接近 ControlNet 的效果 |
| IP-Adapter | 参考图像（风格、身份） | 新增一组 cross-attention 读图像特征，和文本的 cross-attention 并联 | 管"像什么"，不管像素对齐的结构；可以和 ControlNet 同时用 |
| Uni-ControlNet、ControlNet-Union | 多种条件 | 一个网络同时接收多种条件 | 省掉"每种条件一个模型" |

### 9.2 要点回顾

- **ControlNet 解决什么问题？** 在不损坏预训练扩散模型的前提下，加入像素对齐的空间条件（边缘、深度、姿态、分割等）。文本管"画什么"，条件图管"怎么摆"。
- **结构是什么？** 冻结原 U-Net；复制它的编码器和中间层作为可训练拷贝，额外接收条件图；拷贝在每个分辨率的输出经过零卷积，作为残差加到原 U-Net 的 skip 连接和中间层输出上。最终的噪声预测仍由冻结的解码器输出。
- **为什么零初始化？** 保证训练第 0 步模型和原模型逐位相同，训练从一个好的起点出发，不会被随机残差破坏。全零权重也能学：$$\partial\mathcal{L}/\partial W = (\partial\mathcal{L}/\partial y)\,x^\top$$ 和 $$W$$ 无关；但第一步时零卷积前面的参数梯度为 0，梯度是逐渐向前传开的。
- **为什么复制而不是新建？** 拷贝从原模型的权重出发，已经会提取有用的视觉特征，只需学会"如何结合条件图"，比从头训练快得多、对小数据也更稳。
- **训练和采样改了什么？** 训练目标还是噪声 MSE，只训练 ControlNet；采样器完全不变，只是噪声预测多了一个条件图输入。50% 的 caption 置空，训练中会出现"突然收敛"。
- **SD 1.5 上多大？** 条件编码器把 512×512 的条件图降到 64×64；12 条 skip + 1 个中间层共 13 个残差；约 3.61 亿参数，是 U-Net 的 42%。
- **control scale 和 CFG 的区别？** 前者缩放控制残差，管条件图的强度；后者组合有、无文本条件的两次预测，管文本的强度。guess mode 只给有条件分支加残差，并按分辨率加权。
- **常见的坑？** 先冻结再 `deepcopy` 会让拷贝也被冻结；冻结的 U-Net 前向不能放进 `no_grad`；`model.train()` 会把冻结部分也切到训练模式；条件图不进 VAE，范围是 $$[0, 1]$$；残差的数量、通道、分辨率必须和 base 的 skip 一一对应。
