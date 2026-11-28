---
date: 2026-11-27 03:00:00 -0800
tags: [Diffusion, VAE, Stable Diffusion]
keywords: >-
  Latent Diffusion Model, LDM, perceptual compression, semantic compression, autoencoder, KL-reg, VQ-reg, LPIPS, PatchGAN, downsampling factor, scale factor, cross-attention, concatenation conditioning, Stable Diffusion, PyTorch.
summary: >-
  Latent Diffusion（LDM）为什么要把扩散搬进 autoencoder 的 latent 空间：感知压缩与语义压缩的分工、论文图 3 的整体结构、两阶段训练、自编码器的完整损失（L1 + LPIPS + PatchGAN + KL 或 VQ 正则）、压缩倍数 f 的取舍、latent 缩放因子、cross-attention 与拼接两种条件方式。附一个可运行的 Tiny LDM：在 FashionMNIST 上先训自编码器、再在 8×8 latent 里训扩散，和像素空间对比速度和效果。
layout: note
title: "Latent Diffusion（LDM）：把扩散搬进 latent 空间"
description: "Why latent diffusion runs DDPM inside an autoencoder's latent space: perceptual vs semantic compression, the LDM architecture, two-stage training, autoencoder losses, the downsampling factor, latent scaling, cross-attention and concat conditioning, plus a runnable tiny LDM on FashionMNIST."
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

# Latent Diffusion（LDM）：把扩散搬进 latent 空间

[DDPM](/deep-learning/ddpm-from-scratch/) 在像素上加噪、去噪，理论上没有问题，但放到 $$512 \times 512$$ 这样的分辨率上，U-Net 每一步都要处理几十万个像素，采样还要重复几十上百步，训练和推理都非常贵。Latent Diffusion Model（LDM，Rombach 等人，CVPR 2022，论文 *High-Resolution Image Synthesis with Latent Diffusion Models*）的做法很直接：

1. 先单独训练一个**自编码器**，把图像压缩成一个小得多、但在感知上几乎无损的 latent（例如 $$512 \times 512 \times 3 \to 64 \times 64 \times 4$$）；
2. 冻结它，然后**在 latent 空间里做完全标准的 DDPM**；
3. 生成时在 latent 里去噪，最后用解码器解码一次得到图像。

扩散的公式、训练目标、采样器一个都没改，改的只是"在哪个空间里扩散"。[Stable Diffusion](/deep-learning/stable-diffusion/) 就是这篇论文的放大版。本文介绍论文的思路、结构图和两阶段训练的细节，并在 CPU 上训练一个 Tiny LDM 作为实验。

## 参考资料

本文主要参考 LDM 原论文 [1] 和官方仓库 [2]；自编码器的训练方式来自 VQGAN [3]。正文中用 [n] 标注具体出处。

- <a id="ref-1"></a>**[1] 主论文**：Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)。结构图（论文图 3）、压缩倍数 f 的实验、两种正则、条件机制都来自这篇及其附录。
- <a id="ref-2"></a>**[2] 官方仓库**：CompVis/latent-diffusion [github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)。自编码器和 LDM 的配置、损失函数实现。
- <a id="ref-3"></a>**[3] VQGAN**：Esser, Rombach, Ommer. *Taming Transformers for High-Resolution Image Synthesis*. CVPR 2021. [arxiv.org/abs/2012.09841](https://arxiv.org/abs/2012.09841)。第 3 节 L1 + LPIPS + PatchGAN 的自编码器训练方式。
- <a id="ref-4"></a>**[4] LPIPS**：Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR 2018. [arxiv.org/abs/1801.03924](https://arxiv.org/abs/1801.03924)。感知损失。

## 0. 全文路线

| 节 | 内容 |
|---|---|
| 1 | 为什么不在像素空间做：感知压缩和语义压缩 |
| 2 | 论文的整体结构图（图 3），两阶段训练 |
| 3 | 第一阶段：自编码器怎么训（L1 + LPIPS + PatchGAN + KL/VQ 正则） |
| 4 | 压缩倍数 $$f$$ 怎么选 |
| 5 | 第二阶段：在 latent 里训练扩散，latent 缩放因子 |
| 6 | 条件机制：cross-attention 和拼接 |
| 7 | 实验：FashionMNIST 上的 Tiny LDM，和像素空间对比 |
| 8 | 论文结果，从 LDM 到 Stable Diffusion |
| 9 | 代价、后续发展、要点回顾 |

**符号约定**：

| 符号 | 含义 | 代码 | shape（SD 1.x，512×512） |
|---|---|---|---|
| $$x$$ | 图像，归一化到 $$[-1, 1]$$ | `x` | `[B, 3, 512, 512]` |
| $$\mathcal{E},\ \mathcal{D}$$ | 自编码器的编码器、解码器 | `encoder`, `decoder` | — |
| $$f$$ | 空间压缩倍数，$$f = H/h$$ | `2 ** num_down` | 8 |
| $$z_0 = s \cdot \mathcal{E}(x)$$ | 干净 latent（已乘缩放因子 $$s$$） | `z0` | `[B, 4, 64, 64]` |
| $$z_t$$ | 时间步 $$t$$ 的带噪 latent | `z_t` | `[B, 4, 64, 64]` |
| $$y,\ \tau_\theta(y)$$ | 条件（文本、类别、图像……）及其编码 | `labels` / `context` | 文本：`[B, 77, 768]` |
| $$\epsilon_\theta$$ | latent 空间里的去噪 U-Net | `unet` | 输入输出同 $$z_t$$ |

## 1. 为什么不在像素空间做

### 1.1 算力花在哪了

一张 $$512 \times 512$$ 的 RGB 图有 $$3 \times 512 \times 512 = 786{,}432$$ 个数。像素空间的 DDPM 每个采样步都要在这个分辨率上跑一遍 U-Net（卷积、注意力都随像素数增长，自注意力更是平方增长），训练时还要为反向传播存下每层的激活。SD 1.x 的 latent 是 $$4 \times 64 \times 64 = 16{,}384$$ 个数，**少了 48 倍**。U-Net 不会因此严格快 48 倍（通道数、层数还在），但空间尺寸小了 8 倍，卷积和注意力的成本都大幅下降，这正是 LDM 能在单卡上训练、能生成 $$512^2$$ 甚至更大图像的原因。

### 1.2 感知压缩和语义压缩

论文更深一层的论点是：**一张数字图像里大部分比特对应的是人眼察觉不到的细节**。作者分析了一个训练好的像素空间扩散模型的率失真曲线（rate–distortion，横轴是"用了多少比特"，纵轴是"重建误差"），发现学习过程可以粗略分成两段 [[1]](#ref-1)：

- **感知压缩**（perceptual compression）：去掉高频细节，误差下降很多，但学到的"语义"很少；
- **语义压缩**（semantic compression）：真正的生成模型学的部分，学的是物体、构图、概念之间的关系。

<figure class="ddpm-fig">
<svg viewBox="0 0 760 300" role="img" aria-label="感知压缩与语义压缩示意">
  <defs><marker id="fa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker></defs>
  <line x1="80" y1="250" x2="700" y2="250" stroke="#475467" marker-end="url(#fa)"/>
  <line x1="80" y1="250" x2="80" y2="30" stroke="#475467" marker-end="url(#fa)"/>
  <text x="700" y="275" text-anchor="end" class="sm">码率：用了多少比特（越往右越接近原图）</text>
  <text x="70" y="40" text-anchor="end" class="sm">失真</text>
  <rect x="80" y="40" width="210" height="210" fill="#eef4ff" opacity=".8"/>
  <rect x="290" y="40" width="410" height="210" fill="#fff7ed" opacity=".8"/>
  <path d="M95,125 C150,165 210,192 290,206 C380,226 520,236 690,240" fill="none" stroke="#2a6fdb" stroke-width="2.5"/>
  <text x="185" y="70" text-anchor="middle" class="lane">语义压缩</text>
  <text x="185" y="90" text-anchor="middle" class="sm">少量比特决定"画的是什么"</text>
  <text x="185" y="106" text-anchor="middle" class="sm">交给扩散模型（生成）</text>
  <text x="495" y="70" text-anchor="middle" class="lane">感知压缩</text>
  <text x="495" y="90" text-anchor="middle" class="sm">大量比特只是人眼看不出的高频细节</text>
  <text x="495" y="106" text-anchor="middle" class="sm">交给自编码器（压缩 + 重建）</text>
  <line x1="290" y1="40" x2="290" y2="250" stroke="#d92d20" stroke-dasharray="5 4"/>
  <text x="298" y="150" class="sm" style="fill:#d92d20">LDM 在这里切开：</text>
  <text x="298" y="166" class="sm" style="fill:#d92d20">只去掉感知上不重要的部分</text>
</svg>
<figcaption>按论文图 2 的思想重画的示意图（不是原数据）。像素空间扩散把两段都放在一个模型里学；LDM 用自编码器先吃掉右边那一大段。</figcaption>
</figure>

像素空间的扩散模型虽然能通过损失函数"不太在乎"那些不可察觉的细节，但**梯度和网络计算仍然要覆盖每一个像素**。LDM 的想法是把两段分开：先用一个温和的压缩阶段只去掉感知上不重要的信息，再让扩散模型在剩下的、更紧凑的空间里专心学语义。

### 1.3 为什么是二维 latent，而不是一串 token

在 LDM 之前，"先压缩再生成"的路线已经有了：VQ-VAE、VQGAN、DALL·E 把图像压成离散 token，再用自回归 Transformer 生成。但自回归模型要把二维 latent 展平成一维序列，序列越长越难训，所以这些方法被迫**压得很狠**（例如 $$f = 16$$ 甚至 32），图像细节损失严重。

LDM 的 latent 是**保留空间结构的二维特征图**，生成器是卷积 U-Net，天然擅长处理二维结构，所以可以只做温和的压缩（$$f = 4$$ 或 8），细节保留得好很多。这也意味着 latent 不是"语义向量"：它更像一张低分辨率、多通道的特征图，仍然带着位置、轮廓、颜色这些局部信息。

> **易错点**
>
> 1. **latent 不是 resize。** 把图缩小到 $$64 \times 64$$ 会直接丢信息；编码器输出的 4 个通道是学出来的压缩表示，解码器可以从中恢复远超 $$64 \times 64$$ 的细节。
> 2. **latent 也不是纯语义表示。** 它保留了很强的空间结构（第 7 节的实验里，直接看 latent 通道就能认出衣服的轮廓）。语义是扩散模型在这个空间里学的。
> 3. **48 倍是元素数，不是加速比。** 实际加速取决于 U-Net 结构；论文的结论是 $$f = 4 \sim 16$$ 时效率和生成质量平衡得最好（第 4 节）。

## 2. 论文的整体结构

### 2.1 结构图

下面按论文图 3 的结构重画：左边是像素空间，中间是 latent 空间，右边是条件。 [[1]](#ref-1)

<figure class="ddpm-fig">
<svg viewBox="0 0 760 470" role="img" aria-label="LDM 整体结构，按论文图 3 重画">
  <defs>
    <marker id="la" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker>
    <marker id="lr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#d92d20"/></marker>
    <marker id="lg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#12b76a"/></marker>
  </defs>
  <!-- regions -->
  <rect x="8" y="8" width="150" height="372" rx="10" fill="#fdf2f2" stroke="#f4b4b4"/>
  <text x="83" y="30" text-anchor="middle" class="lane">像素空间</text>
  <rect x="168" y="8" width="420" height="372" rx="10" fill="#ecfdf3" stroke="#a6e9c5"/>
  <text x="378" y="30" text-anchor="middle" class="lane">latent 空间</text>
  <rect x="598" y="8" width="154" height="372" rx="10" fill="#f2f4f7" stroke="#d0d5dd"/>
  <text x="675" y="30" text-anchor="middle" class="lane">条件</text>
  <!-- pixel space: x, E, D, x~ -->
  <text x="40" y="85" text-anchor="middle" class="it">x</text>
  <polygon points="70,60 120,72 120,98 70,110" fill="#fff" stroke="#d92d20"/>
  <text x="95" y="90" text-anchor="middle" class="it">ℰ</text>
  <line x1="50" y1="85" x2="68" y2="85" stroke="#475467" marker-end="url(#la)"/>
  <polygon points="70,292 120,280 120,330 70,318" fill="#fff" stroke="#d92d20"/>
  <text x="95" y="310" text-anchor="middle" class="it">𝒟</text>
  <text x="40" y="310" text-anchor="middle" class="it">x̃</text>
  <line x1="68" y1="305" x2="52" y2="305" stroke="#475467" marker-end="url(#la)"/>
  <!-- latent: forward diffusion on top -->
  <text x="200" y="90" text-anchor="middle" class="it">z</text>
  <line x1="120" y1="85" x2="188" y2="85" stroke="#475467" marker-end="url(#la)"/>
  <rect x="225" y="66" width="250" height="38" rx="8" fill="#fff" stroke="#98a2b3"/>
  <text x="350" y="90" text-anchor="middle">扩散过程（逐步加噪，无参数）</text>
  <line x1="210" y1="85" x2="223" y2="85" stroke="#475467" marker-end="url(#la)"/>
  <line x1="475" y1="85" x2="520" y2="85" stroke="#475467" marker-end="url(#la)"/>
  <text x="540" y="90" text-anchor="middle" class="it">z<tspan baseline-shift="sub" font-size="11">T</tspan></text>
  <!-- denoising U-Net -->
  <rect x="200" y="150" width="340" height="190" rx="10" fill="#fff" stroke="#2a6fdb" stroke-width="2"/>
  <text x="370" y="172" text-anchor="middle" class="lane">去噪 U-Net ε<tspan baseline-shift="sub" font-size="10">θ</tspan></text>
  <!-- encoder side boxes -->
  <rect x="220" y="190" width="40" height="120" rx="4" fill="#eef4ff" stroke="#2a6fdb"/>
  <rect x="275" y="210" width="34" height="80" rx="4" fill="#eef4ff" stroke="#2a6fdb"/>
  <rect x="324" y="228" width="30" height="44" rx="4" fill="#eef4ff" stroke="#2a6fdb"/>
  <rect x="386" y="228" width="30" height="44" rx="4" fill="#eef4ff" stroke="#2a6fdb"/>
  <rect x="431" y="210" width="34" height="80" rx="4" fill="#eef4ff" stroke="#2a6fdb"/>
  <rect x="480" y="190" width="40" height="120" rx="4" fill="#eef4ff" stroke="#2a6fdb"/>
  <!-- QKV mini blocks -->
  <g class="sm">
    <rect x="281" y="296" width="22" height="16" rx="3" fill="#fff7ed" stroke="#f79009"/><text x="292" y="308" text-anchor="middle" class="sm">QKV</text>
    <rect x="328" y="276" width="22" height="16" rx="3" fill="#fff7ed" stroke="#f79009"/><text x="339" y="288" text-anchor="middle" class="sm">QKV</text>
    <rect x="390" y="276" width="22" height="16" rx="3" fill="#fff7ed" stroke="#f79009"/><text x="401" y="288" text-anchor="middle" class="sm">QKV</text>
    <rect x="437" y="296" width="22" height="16" rx="3" fill="#fff7ed" stroke="#f79009"/><text x="448" y="308" text-anchor="middle" class="sm">QKV</text>
  </g>
  <line x1="260" y1="250" x2="273" y2="250" stroke="#475467" marker-end="url(#la)"/>
  <line x1="309" y1="250" x2="322" y2="250" stroke="#475467" marker-end="url(#la)"/>
  <line x1="354" y1="250" x2="384" y2="250" stroke="#475467" marker-end="url(#la)"/>
  <line x1="416" y1="250" x2="429" y2="250" stroke="#475467" marker-end="url(#la)"/>
  <line x1="465" y1="250" x2="478" y2="250" stroke="#475467" marker-end="url(#la)"/>
  <!-- skip connections -->
  <path d="M240,190 L240,182 L500,182 L500,188" fill="none" stroke="#475467" stroke-dasharray="4 3" marker-end="url(#la)"/>
  <path d="M292,210 L292,200 L448,200 L448,208" fill="none" stroke="#475467" stroke-dasharray="4 3" marker-end="url(#la)"/>
  <path d="M339,228 L339,218 L401,218 L401,226" fill="none" stroke="#475467" stroke-dasharray="4 3" marker-end="url(#la)"/>
  <!-- z_T into unet, output z_{T-1} -->
  <path d="M540,96 L540,130 L180,130 L180,250 L218,250" fill="none" stroke="#475467" marker-end="url(#la)"/>
  <text x="190" y="122" class="sm">训练：z<tspan baseline-shift="sub" font-size="9">t</tspan>；采样：从 z<tspan baseline-shift="sub" font-size="9">T</tspan> 开始</text>
  <line x1="520" y1="250" x2="556" y2="250" stroke="#475467" marker-end="url(#la)"/>
  <text x="572" y="255" text-anchor="middle" class="it">z<tspan baseline-shift="sub" font-size="10">t−1</tspan></text>
  <path d="M572,262 L572,360 L190,360 L190,272 L218,272" fill="none" stroke="#475467" stroke-dasharray="2 3" marker-end="url(#la)"/>
  <text x="380" y="354" text-anchor="middle" class="sm">重复去噪 T 次（或 DDIM 的 N 步），得到 z₀</text>
  <path d="M200,330 L150,330 L150,305 L122,305" fill="none" stroke="#475467" marker-end="url(#la)"/>
  <text x="160" y="345" class="sm">z₀</text>
  <!-- conditioning -->
  <rect x="612" y="50" width="126" height="28" rx="6" fill="#fff" stroke="#98a2b3"/><text x="675" y="69" text-anchor="middle">语义分割图</text>
  <rect x="612" y="86" width="126" height="28" rx="6" fill="#fff" stroke="#98a2b3"/><text x="675" y="105" text-anchor="middle">文本</text>
  <rect x="612" y="122" width="126" height="28" rx="6" fill="#fff" stroke="#98a2b3"/><text x="675" y="141" text-anchor="middle">表征（类别等）</text>
  <rect x="612" y="158" width="126" height="28" rx="6" fill="#fff" stroke="#98a2b3"/><text x="675" y="177" text-anchor="middle">图像</text>
  <polygon points="640,214 710,214 698,250 652,250" fill="#fff" stroke="#12b76a"/>
  <text x="675" y="237" text-anchor="middle" class="it">τ<tspan baseline-shift="sub" font-size="10">θ</tspan></text>
  <line x1="675" y1="188" x2="675" y2="212" stroke="#475467" marker-end="url(#la)"/>
  <path d="M652,240 L600,240 L600,320 L470,320 L460,314" fill="none" stroke="#12b76a" stroke-width="1.6" marker-end="url(#lg)"/>
  <path d="M600,320 L410,320 L404,294" fill="none" stroke="#12b76a" stroke-width="1.6" marker-end="url(#lg)"/>
  <path d="M600,320 L600,326 L340,326 L340,294" fill="none" stroke="#12b76a" stroke-width="1.6" marker-end="url(#lg)"/>
  <path d="M340,326 L292,326 L292,314" fill="none" stroke="#12b76a" stroke-width="1.6" marker-end="url(#lg)"/>
  <text x="675" y="272" text-anchor="middle" class="sm">cross-attention：</text>
  <text x="675" y="287" text-anchor="middle" class="sm">进每个 QKV 的 K、V</text>
  <path d="M612,64 L594,64 L594,142 L190,142 L190,236 L216,236" fill="none" stroke="#d92d20" stroke-dasharray="6 3" marker-end="url(#lr)"/>
  <!-- legend -->
  <line x1="20" y1="410" x2="50" y2="410" stroke="#475467" marker-end="url(#la)"/><text x="56" y="414" class="sm">数据流 / 去噪一步</text>
  <rect x="175" y="402" width="22" height="16" rx="3" fill="#fff7ed" stroke="#f79009"/><text x="203" y="414" class="sm">cross-attention</text>
  <line x1="310" y1="410" x2="340" y2="410" stroke="#12b76a" stroke-width="1.6" marker-end="url(#lg)"/><text x="346" y="414" class="sm">条件 τθ(y)</text>
  <line x1="430" y1="410" x2="460" y2="410" stroke="#475467" stroke-dasharray="4 3"/><text x="466" y="414" class="sm">skip 连接</text>
  <line x1="545" y1="410" x2="575" y2="410" stroke="#d92d20" stroke-dasharray="6 3"/><text x="581" y="414" class="sm">拼接到输入（空间对齐的条件）</text>
  <text x="20" y="446" class="sm">红色梯形是第一阶段训练好、之后冻结的自编码器；蓝色 U-Net 和绿色 τθ 是第二阶段一起训练的。</text>
</svg>
<figcaption>LDM 整体结构，按论文图 3 的布局重画。上面一行是训练时的前向加噪，下面是去噪 U-Net；条件要么经 τθ 编码后通过 cross-attention 注入（文本、类别），要么直接拼到 U-Net 输入上（分割图、低分辨率图像这类和输出空间对齐的条件）。</figcaption>
</figure>

图里的每一块：

| 部件 | 作用 | 何时训练 |
|---|---|---|
| $$\mathcal{E}$$ | 图像 $$x$$ → latent $$z$$，空间缩小 $$f$$ 倍 | 第一阶段，之后冻结 |
| $$\mathcal{D}$$ | latent → 图像 $$\tilde x$$ | 第一阶段，之后冻结 |
| 扩散过程 | $$z \to z_t$$，和 DDPM 完全一样的加噪公式，无参数 | — |
| 去噪 U-Net $$\epsilon_\theta$$ | 预测 $$z_t$$ 里的噪声；中间层带 cross-attention | 第二阶段 |
| $$\tau_\theta$$ | 把条件 $$y$$ 编码成 token 序列（文本用 Transformer，类别用 embedding） | 第二阶段，和 U-Net 一起训（SD 换成了冻结的 CLIP，见 8.2） |
| 拼接（switch） | 空间对齐的条件直接和 $$z_t$$ 在通道维拼接 | — |

### 2.2 两阶段训练和推理

整个系统分两个阶段训练，两个阶段的损失完全不同：

| 阶段 | 训练什么 | 输入 | 损失 |
|---|---|---|---|
| 1. 感知压缩 | $$\mathcal{E}$$、$$\mathcal{D}$$（和判别器） | 图像 | 重建 + 感知 + 对抗 + 正则（第 3 节） |
| 2. latent 扩散 | $$\epsilon_\theta$$（和 $$\tau_\theta$$） | $$z_0 = s \cdot \mathcal{E}(x)$$，冻结的编码器给出 | latent 上预测噪声的 MSE（第 5 节） |
| 推理：文生图 | 无 | 随机 $$z_T$$ | 去噪 → $$\mathcal{D}(z_0 / s)$$，解码器只跑一次 |
| 推理：图生图、修补 | 无 | 输入图经 $$\mathcal{E}$$ 编码 | 同上 |

为什么第二阶段要冻结自编码器：

- **目标固定**：如果编码器也在变，同一张图的 latent 一直在动，扩散模型等于在追一个移动的靶子；
- **可以缓存**：训练集可以事先全部编码好存起来，扩散训练时完全不用再跑编码器；
- **可以复用**：同一个自编码器可以给很多个扩散模型用（论文里所有任务都共用几个自编码器，SD 1.x 的各个版本也共用一个 VAE）；
- **分工清楚**：重建质量和生成质量分开优化、分开评估。

这是工程上的选择，不是数学上的必须；但联合训练会让 latent 空间不稳定，后来的主流工作基本都沿用了"先训 tokenizer，再冻结"的做法。

> **易错点**
>
> 1. **"VAE 只在推理时用"是不对的。** 扩散训练要用编码器把每张训练图变成 $$z_0$$（或者事先缓存），只是它不接收梯度。纯文生图推理才不需要编码器。
> 2. **起点 $$z_T \sim \mathcal{N}(0, I)$$，终点不是。** 扩散模型负责把高斯噪声变成"像真实 latent 的样本"；解码器只会解码"像真实 latent"的输入。直接把一个高斯样本送进解码器，得到的是一团没有意义的纹理（第 7 节有图）。

## 3. 第一阶段：感知压缩的自编码器

### 3.1 完整的训练目标

论文的自编码器沿用了 VQGAN 的训练方式。只用 L1 或 L2 这种逐像素损失时，模型会对不确定的高频细节"取平均"，重建结果发糊；所以要再加两项让重建在**感知上**锐利 [[3]](#ref-3)[[4]](#ref-4)：

$$
\mathcal{L}_{\text{AE}} =
\underbrace{\| x - \tilde x \|_1}_{\text{重建}}
+ \underbrace{\mathcal{L}_{\text{LPIPS}}(x, \tilde x)}_{\text{感知}}
+ \lambda_{\text{adv}} \underbrace{\mathcal{L}_{\text{GAN}}(\tilde x)}_{\text{对抗}}
+ \underbrace{\mathcal{L}_{\text{reg}}}_{\text{正则}},
\qquad \tilde x = \mathcal{D}(\mathcal{E}(x))
$$

- **LPIPS 感知损失**：把 $$x$$ 和 $$\tilde x$$ 都送进一个预训练的 VGG，比较多层特征的距离。它在乎轮廓、纹理这些人眼敏感的结构，不要求每个像素都一样。
- **PatchGAN 对抗损失**：判别器不对整张图打一个分，而是对每个局部 patch 打分（输出一张"真假图"）。它逼着解码器生成真实的局部纹理，而不是模糊的平均值。判别器用 hinge 损失，并且训练开始一段时间后才打开（先让重建稳定下来）。
- **正则项**：防止 latent 空间的尺度任意漂移，有两种选择，见 3.2 节。

对抗项的权重 $$\lambda_{\text{adv}}$$ 不是常数，而是**自适应**地按解码器最后一层上两种梯度的大小比例算出来，让对抗梯度和重建梯度保持同一量级：

$$
\lambda_{\text{adv}} = \frac{\big\| \nabla_{G_L} \mathcal{L}_{\text{rec}} \big\|}{\big\| \nabla_{G_L} \mathcal{L}_{\text{GAN}} \big\| + \delta},
\qquad G_L = \text{解码器最后一层的权重},\ \delta = 10^{-4}
$$

```python
def adaptive_gan_weight(rec_loss, gan_loss, last_layer_weight, max_weight=1e4):
    # λ_adv = ‖∇ L_rec‖ / (‖∇ L_GAN‖ + δ)，两个梯度都对解码器最后一层权重求
    rec_grad = torch.autograd.grad(rec_loss, last_layer_weight, retain_graph=True)[0]
    gan_grad = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]
    weight = rec_grad.norm() / (gan_grad.norm() + 1e-4)
    return weight.clamp(0.0, max_weight).detach()          # 只是一个系数，不传梯度


def hinge_d_loss(logits_real, logits_fake):
    # 判别器：真 patch 的打分推到 ≥ 1，假 patch 推到 ≤ −1
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def generator_gan_loss(logits_fake):
    # 解码器（生成器）这一侧：让判别器给重建图高分
    return -logits_fake.mean()
```

### 3.2 两种正则：KL-reg 和 VQ-reg

论文同时实验了两种正则，对应两类模型（名字里的 KL / VQ 就是指这个） [[1]](#ref-1)：

**KL-reg**：编码器输出一个对角高斯 $$q(z \mid x) = \mathcal{N}(\mu, \operatorname{diag}\sigma^2)$$，用重参数化采样，加一个对 $$\mathcal{N}(0, I)$$ 的 KL 惩罚：

$$
z = \mu + \sigma \odot \xi,\ \ \xi \sim \mathcal{N}(0, I),\qquad
\mathcal{L}_{\text{reg}} = \lambda_{\text{KL}} \cdot \tfrac12 \sum \big( \mu^2 + \sigma^2 - 1 - \log \sigma^2 \big)
$$

和教科书 VAE 不同的是，$$\lambda_{\text{KL}}$$ **非常小**（约 $$10^{-6}$$）。它只起"别让 latent 的尺度跑飞"的作用，并不要求 latent 分布接近标准正态，也不要求解码器能把任意高斯样本解码成好图。所以训练好的 $$\sigma$$ 很小，编码器几乎是确定性的。SD 用的就是这一种（KL-f8，4 通道）。

**VQ-reg**：在解码器里放一个向量量化层，把每个位置的特征换成码本里最近的向量：

$$
z_q[i, j] = e_k,\quad k = \arg\min_m \| z[i, j] - e_m \|_2
$$

它和 VQGAN 的区别在于：**量化层被"吸收"进了解码器**。扩散模型操作的是量化**之前**的连续 latent $$z$$，量化只在解码时发生。这样扩散仍然在连续空间里进行，同时量化层起到了正则的作用。

```python
class VectorQuantizer(nn.Module):
    """VQ-reg 用：在解码器入口把连续 latent 换成最近的码本向量"""
    def __init__(self, num_codes=8192, dim=4, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, dim)
        self.beta = beta

    def forward(self, z):                                          # z: [B, d, h, w]
        flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])       # [B·h·w, d]
        idx = torch.cdist(flat, self.codebook.weight).argmin(dim=1)
        z_q = self.codebook(idx).view(z.shape[0], z.shape[2], z.shape[3], -1).permute(0, 3, 1, 2)
        # 码本损失 + commitment 损失（β 控制编码器向码本靠拢的力度）
        loss = F.mse_loss(z_q, z.detach()) + self.beta * F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()                               # straight-through：反向时梯度直通
        return z_q, loss
```

论文的观察：VQ 正则的模型重建指标略差，但**有时生成质量更好**；在超分、修补等任务上论文用了 VQ-f4，文生图大模型用了 KL-f8。

### 3.3 一个最小的 KL 自编码器

下面是第 7 节实验用的自编码器，把上面的 KL-reg 写全了。为了能在 CPU 上几分钟训完，只保留 L1 + KL，去掉了 LPIPS 和判别器（第 7 节会看到代价：重建偏糊）：

```python
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.block(x)


class TinyEncoder(nn.Module):
    """x [B, C, H, W] → (μ, log σ²)，各 [B, z_ch, H/f, W/f]，f = 2 ** num_down"""
    def __init__(self, in_ch=1, ch=64, z_ch=4, num_down=2):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ch, 3, padding=1), ResBlock(ch)]
        for _ in range(num_down):
            layers += [nn.Conv2d(ch, ch, 3, stride=2, padding=1), ResBlock(ch)]   # 分辨率减半
        layers += [nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 2 * z_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        mean, logvar = self.net(x).chunk(2, dim=1)        # 前一半通道是 μ，后一半是 log σ²
        return mean, logvar.clamp(-30.0, 20.0)


class TinyDecoder(nn.Module):
    """z [B, z_ch, h, w] → x̃ [B, C, h·f, w·f]"""
    def __init__(self, out_ch=1, ch=64, z_ch=4, num_down=2):
        super().__init__()
        layers = [nn.Conv2d(z_ch, ch, 3, padding=1), ResBlock(ch)]
        for _ in range(num_down):
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(ch, ch, 3, padding=1), ResBlock(ch)]   # 分辨率加倍
        layers += [nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class TinyAutoencoderKL(nn.Module):
    def __init__(self, in_ch=1, ch=64, z_ch=4, num_down=2):
        super().__init__()
        self.encoder = TinyEncoder(in_ch, ch, z_ch, num_down)
        self.decoder = TinyDecoder(in_ch, ch, z_ch, num_down)

    def encode(self, x, sample=True):
        mean, logvar = self.encoder(x)
        if not sample:
            return mean, mean, logvar                     # 用均值（众数），结果确定
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)   # z = μ + σ·ξ
        return z, mean, logvar

    def decode(self, z):
        return self.decoder(z)


def autoencoder_loss(model, x, kl_weight=1e-6):
    z, mean, logvar = model.encode(x)
    x_hat = model.decode(z)
    rec = (x - x_hat).abs().mean()                                        # ‖x − x̃‖₁
    kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=(1, 2, 3)).mean()
    return rec + kl_weight * kl, rec, kl
```

和 SD 的 VAE 对比：结构是一样的（卷积 + ResBlock + 下采样，出口给 $$2 \times$$ 通道的 $$\mu, \log\sigma^2$$），SD 的版本更深更宽（通道 128/256/512/512，$$f = 8$$，在最低分辨率还有一层自注意力，约 8400 万参数），并且用了完整的 L1 + LPIPS + PatchGAN 损失。编码器输出分布的写法见 [SD 篇 3.3 节](/deep-learning/stable-diffusion/)。

> **易错点（自编码器）**
>
> 1. **只用 L1/L2 会糊。** 逐像素损失对不确定的细节取平均；感知损失和对抗损失才是让重建锐利的关键。代价是训练更麻烦（GAN 要晚点打开、要自适应权重）。
> 2. **KL 权重别按教科书 VAE 设。** 设成 1 时 latent 被强行拉向标准正态，信息被压掉很多，重建明显变差；LDM 只要一个极弱的约束。
> 3. **VQ 模型里扩散用的是量化前的 latent。** 量化是解码器的第一步，不要在扩散的训练数据上先量化。
> 4. **对数方差要 clamp。** `exp(logvar)` 很容易溢出，训练初期尤其如此。

## 4. 压缩倍数 f 怎么选

$$f$$ 越大，latent 越小，扩散越便宜；但压得越狠，自编码器丢的信息越多，生成质量的上限越低。论文在类别条件 ImageNet 上用相同的参数量和训练步数比较了 $$f \in \{1, 2, 4, 8, 16, 32\}$$（$$f = 1$$ 就是像素空间扩散） [[1]](#ref-1)：

- **$$f = 1, 2$$ 训练进展慢**：大部分感知压缩的工作还留给扩散模型做，算力花在了像素级细节上；
- **$$f = 32$$ 质量停滞**：第一阶段压得太狠，细节在自编码器里就丢了，扩散模型再好也补不回来；
- **$$f = 4 \sim 16$$ 效率和质量平衡得最好**；训练 200 万步后，$$f = 8$$ 和像素空间（$$f = 1$$）的 FID 差了 38；
- 结论：**$$f = 4$$ 和 $$f = 8$$** 是最好的选择。SD 选了 $$f = 8$$、4 通道。

latent 的通道数 $$c$$ 也要配合 $$f$$ 选：$$f$$ 越大，每个位置要装的信息越多，通道数也要多一些（论文里 $$f = 4$$ 配 3 通道，$$f = 8$$ 配 4 通道，$$f = 16$$ 配 8 或 16 通道等）。

在第 7 节的小实验里也能看到同样的趋势。同一个自编码器结构（3.3 节）只改下采样次数，在 FashionMNIST（补到 $$32 \times 32$$）上各训 2000 步：

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/latent-diffusion/recon_by_f.png" alt="不同压缩倍数下自编码器的重建">
  <figcaption>第一行是原图，下面三行分别是 f = 2、4、8 的重建（括号里是 latent 的 shape）。</figcaption>
</figure>

| $$f$$ | latent shape | 元素数（原图 1024） | 测试集 PSNR |
|---|---|---|---|
| 2 | $$4 \times 16 \times 16$$ | 1024（不压缩） | 37.6 dB |
| 4 | $$4 \times 8 \times 8$$ | 256（压缩 4 倍） | 26.0 dB |
| 8 | $$4 \times 4 \times 4$$ | 64（压缩 16 倍） | 22.1 dB |

$$f = 2$$ 几乎无损，但 latent 的元素数和原图一样多，等于没省算力；$$f = 8$$ 时 "Lee" 字样、格子纹理都没了，这些细节**扩散模型再好也生成不出来**。$$f = 4$$ 丢掉了最细的纹理（这里只用了 L1 损失，所以偏糊），但保留了形状和大部分明暗，第 7 节就用它。

## 5. 第二阶段：在 latent 里做扩散

### 5.1 公式只换了一个字母

DDPM 的训练目标（[DDPM 篇](/deep-learning/ddpm-from-scratch/)）是

$$
\mathcal{L}_{\text{DM}} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0, I), t}\Big[ \big\| \epsilon - \epsilon_\theta(x_t, t) \big\|_2^2 \Big]
$$

LDM 把 $$x$$ 换成编码后的 $$z$$，其余完全不变：

$$
\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0, I), t}\Big[ \big\| \epsilon - \epsilon_\theta(z_t, t) \big\|_2^2 \Big],
\qquad z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon
$$

加上条件 $$y$$（6 节）时，$$\epsilon_\theta$$ 多一个输入 $$\tau_\theta(y)$$，$$\tau_\theta$$ 和 $$\epsilon_\theta$$ 一起优化：

$$
\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(x), y, \epsilon, t}\Big[ \big\| \epsilon - \epsilon_\theta\big(z_t, t, \tau_\theta(y)\big) \big\|_2^2 \Big]
$$

噪声调度、时间步采样、CFG、DDPM/DDIM 采样器全部原样照搬。代码层面的变化只有四处：

1. 训练前（或训练中、不求梯度地）用编码器得到 $$z_0$$；
2. U-Net 的输入、输出通道数是 latent 通道数（例如 4），不是 RGB 的 3；
3. 采样得到的是 latent，最后要过一次解码器；
4. 在编码器和扩散之间乘一个缩放因子 $$s$$（5.2 节）。

### 5.2 latent 缩放因子

扩散的前向过程默认数据大致是**单位方差**：只有这样 $$z_T = \sqrt{\bar\alpha_T} z_0 + \sqrt{1 - \bar\alpha_T}\epsilon$$ 在 $$t = T$$ 时才接近 $$\mathcal{N}(0, I)$$，噪声调度在各个时间步上的"信噪比"也才和像素空间时的设计一致。但 KL 权重那么小，编码器输出的 latent 方差可以是任意的（SD 的 VAE 约为 $$5.5^2$$）。

论文的做法（附录里针对 KL 模型）是：在第一批训练数据上估计 latent 的标准差 $$\hat\sigma$$，然后

$$
z_0 = s \cdot \mathcal{E}(x),\qquad \tilde x = \mathcal{D}(z_0 / s),\qquad s = 1 / \hat\sigma
$$

SD 1.x 的 $$s = 0.18215 \approx 1 / 5.49$$ 就是这么来的。VQ 模型的 latent 天然有界，一般不需要这一步。

```python
@torch.no_grad()
def estimate_scale_factor(autoencoder, images):
    # 在一批训练图像上估计 latent 的标准差，s = 1 / σ̂
    _, mean, _ = autoencoder.encode(images, sample=False)
    return 1.0 / mean.std().item()
```

### 5.3 训练一步

```python
def ldm_training_step(unet, autoencoder, optimizer, x0, labels, sched, scale):
    # x0: [B, C, H, W] 图像；autoencoder 已冻结
    with torch.no_grad():
        z, _, _ = autoencoder.encode(x0)                      # 从 q(z|x) 采样，[B, 4, H/f, W/f]
        z0 = z * scale                                        # z0 = s · E(x)
    t = torch.randint(0, sched["betas"].shape[0], (z0.shape[0],), device=z0.device)
    noise = torch.randn_like(z0)
    a = sched["sqrt_alpha_bars"][t].view(-1, 1, 1, 1)
    b = sched["sqrt_one_minus_alpha_bars"][t].view(-1, 1, 1, 1)
    z_t = a * z0 + b * noise                                  # z_t = √ᾱ z0 + √(1−ᾱ) ε
    labels = drop_labels_for_cfg(labels, unet.null_class_id)  # 10% 丢标签，给 CFG 用
    loss = F.mse_loss(unet(z_t, t, labels), noise)            # ‖ε − ε_θ(z_t, t, y)‖²
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()
```

`drop_labels_for_cfg` 和下面的 `predict_noise_cfg` 来自 [DDPM 篇的 CFG 一节](/deep-learning/ddpm-from-scratch/)，U-Net 就是那里的 `ClassConditionedTinyUNet`，只是 `in_channels=4`。大规模训练时通常把整个数据集事先编码成 $$(\mu, \log\sigma^2)$$ 存起来，训练时只做第 4 行之后的部分（第 7 节的实验就是这样做的）。

### 5.4 采样

```python
@torch.no_grad()
def ldm_sample(unet, autoencoder, labels, sched, scale, latent_shape, num_steps=50, w=3.0):
    unet.eval()
    steps = ddim_timesteps(sched["betas"].shape[0], num_steps)        # DDIM 篇
    z = torch.randn(labels.shape[0], *latent_shape)                   # z_T ~ N(0, I)，在 latent 空间
    for i, t_index in enumerate(steps):
        prev = steps[i + 1] if i + 1 < len(steps) else -1
        t = torch.full((z.shape[0],), t_index, dtype=torch.long)
        eps = predict_noise_cfg(unet, z, t, labels, w)                 # CFG
        abar_t = sched["alpha_bars"][t_index]
        abar_s = sched["alpha_bars"][prev] if prev >= 0 else abar_t.new_tensor(1.0)
        z0_hat = (z - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()
        z = abar_s.sqrt() * z0_hat + (1 - abar_s).sqrt() * eps        # η = 0 的 DDIM 一步
    return autoencoder.decode(z / scale)                               # 解码前除回 s，只解码一次
```

> **易错点（latent 扩散）**
>
> 1. **编码后乘 $$s$$、解码前除 $$s$$，两处都不能漏。** 都不会报错：漏了乘，训练数据的尺度和噪声对不上；漏了除，解码出来颜色、对比度全错。
> 2. **$$s$$ 是自编码器的属性，不是扩散模型的超参数。** 换一个自编码器（例如 SDXL 的 VAE，$$s = 0.13025$$）就要换 $$s$$；同一个自编码器上训练的所有扩散模型共用一个 $$s$$。
> 3. **U-Net 的通道数跟着 latent 走。** `in_channels` 和输出通道都等于 latent 通道数；分辨率要能被 U-Net 的下采样次数整除（第 7 节把 28×28 补到 32×32 就是为了这个）。
> 4. **`clip_x0` 之类的像素空间技巧不适用。** 像素在 $$[-1, 1]$$ 里，可以把 $$\hat x_0$$ 截断；latent 没有这个范围，不要截断。

## 6. 条件机制：cross-attention 和拼接

### 6.1 cross-attention：通用的条件接口

为了让同一套结构能接受文本、类别、图像表征等各种条件，LDM 用一个**领域相关的编码器** $$\tau_\theta$$ 把条件 $$y$$ 变成一串 token $$\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$$，再在 U-Net 的中间层用 cross-attention 注入。把第 $$i$$ 层的特征图 $$\varphi_i(z_t)$$ 展平成 $$N$$ 个 token [[1]](#ref-1)：

$$
Q = W_Q^{(i)} \cdot \varphi_i(z_t),\qquad
K = W_K^{(i)} \cdot \tau_\theta(y),\qquad
V = W_V^{(i)} \cdot \tau_\theta(y)
$$

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\Big( \frac{Q K^\top}{\sqrt d} \Big) \cdot V
$$

**Query 来自图像特征，Key 和 Value 来自条件**：每个空间位置去条件的 $$M$$ 个 token 里"查"自己需要的信息。输出和 $$\varphi_i(z_t)$$ 同形状，以残差形式加回去。实际实现里它是 Transformer block 的一部分（self-attention → cross-attention → 前馈），完整代码见 [SD 篇 5.5 节和 6.2～6.4 节](/deep-learning/stable-diffusion/)。

$$\tau_\theta$$ 随任务变化：

| 条件 | $$\tau_\theta$$ | 论文 / SD 的做法 |
|---|---|---|
| 文本 | Transformer | 论文：BERT tokenizer + 从头训练的 Transformer，和 U-Net 一起训；SD：冻结的 CLIP ViT-L/14 文本编码器 |
| 类别 | embedding 表 | 一个 token |
| 布局（边界框） | Transformer | 每个框编码成 token |

### 6.2 拼接：空间对齐的条件

如果条件本身就是一张和输出**像素对齐**的图（语义分割图、低分辨率图像、带遮挡的图像），更简单有效的做法是把它缩放到 latent 的分辨率，**直接在通道维拼到 $$z_t$$ 上**，U-Net 的输入通道数相应增加：

$$
\hat\epsilon = \epsilon_\theta\big( [\, z_t,\ c\, ],\ t \big),\qquad c \in \mathbb{R}^{c' \times h \times w}
$$

```python
class ConcatConditionedUNet(nn.Module):
    """把空间对齐的条件拼到 z_t 上；U-Net 输入通道 = latent 通道 + 条件通道"""
    def __init__(self, unet_cls, latent_ch=4, cond_ch=1, **kw):
        super().__init__()
        self.unet = unet_cls(in_channels=latent_ch + cond_ch, **kw)
        # 输出仍然只预测 latent 的噪声：把最后一层换成 latent_ch 个输出通道
        self.unet.output = nn.Conv2d(self.unet.output.in_channels, latent_ch, 3, padding=1)

    def forward(self, z_t, t, cond):
        cond = F.interpolate(cond, size=z_t.shape[-2:], mode="nearest")  # 对齐到 latent 分辨率
        return self.unet(torch.cat([z_t, cond], dim=1), t)
```

论文里用拼接的任务：

| 任务 | 拼接的条件 | 说明 |
|---|---|---|
| 超分辨率（4×） | 低分辨率图像 | $$f = 4$$ 时 latent 和低分辨率图一样大，直接拼 |
| 图像修补（inpainting） | 被遮挡图像的 latent + 遮罩 | SD 的 inpainting 版本沿用：4（$$z_t$$）+ 4（遮挡图的 latent）+ 1（遮罩）= 9 个输入通道 |
| 语义合成 | 语义分割图 | 训练在 $$256^2$$，推理时可以直接用更大的分割图卷积式地生成 $$512^2$$～$$1024^2$$ 的图 |

最后一行是 LDM 的一个特性：U-Net 是全卷积的，空间对齐的条件可以给任意大小，于是可以在比训练分辨率更大的画布上采样。

> **易错点（条件）**
>
> 1. **cross-attention 里 Q 来自图像、K/V 来自条件，反过来就错了。** 输出要和图像特征同形状才能加回去，所以 Query 必须来自图像。
> 2. **拼接时只改输入通道，不改输出通道。** 网络仍然只预测 latent 的噪声；条件通道不加噪。
> 3. **拼接的条件要先对齐到 latent 分辨率。** 条件是 $$512^2$$ 的图、latent 是 $$64^2$$ 时，要先下采样（或者像 inpainting 那样先编码进 latent 空间）。
> 4. **"开关"不是二选一的硬规定。** 同一个模型可以两种都用：例如 SD inpainting 用拼接给遮挡信息，同时用 cross-attention 给文本。

## 7. 实验：FashionMNIST 上的 Tiny LDM

用两阶段训练一个完整的小 LDM，再用**同一个 U-Net 结构、同样的步数**在像素空间训练一个对照组：

- 数据：FashionMNIST，补边到 $$32 \times 32$$（让分辨率能被下采样整除），范围 $$[-1, 1]$$；
- 第一阶段：3.3 节的 `TinyAutoencoderKL`，$$f = 4$$，4 通道，L1 + $$10^{-6}$$ KL，Adam $$2 \times 10^{-4}$$，2000 步；
- 第二阶段：DDPM 篇的 `ClassConditionedTinyUNet`（基础通道 64，约 78 万参数），$$T = 300$$，AdamW $$2 \times 10^{-4}$$，batch 64，3000 步，10% 丢标签；
- 采样：DDIM 50 步，CFG $$w = 3$$，每个类别 8 张，固定初始噪声；
- 对照组：完全相同的 U-Net 和超参数，只是 `in_channels=1`，直接在 $$32 \times 32$$ 像素上训练。

所有时间都是在 4 核 CPU 上测的。

### 7.1 latent 长什么样

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/latent-diffusion/latent_channels.png" alt="自编码器 latent 的 4 个通道">
  <figcaption>左边是输入图，右边是编码器均值 μ 的 4 个通道（8×8，红正蓝负）。</figcaption>
</figure>

latent 不是什么抽象的语义向量：每个通道都是一张 $$8 \times 8$$ 的"特征图"，能直接看出鞋、上衣、裤子的轮廓。这就是 1.3 节说的"保留空间结构的二维 latent"，所以去噪网络仍然可以用卷积 U-Net。

整个训练集的 latent 标准差是 0.70，所以缩放因子 $$s = 1/0.70 \approx 1.43$$。这里 latent 的尺度本来就不大；SD 的 VAE 是 5.5，漏乘 $$s$$ 就是数量级的错误。

### 7.2 latent 空间 vs 像素空间

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/latent-diffusion/pixel_vs_latent_samples.png" alt="像素空间与 latent 空间扩散的生成结果">
  <figcaption>同一个 U-Net、同样训练 3000 步。每行一个类别（T 恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴）。</figcaption>
</figure>

| | 像素空间 | latent 空间 |
|---|---|---|
| U-Net 输入 | $$1 \times 32 \times 32$$ | $$4 \times 8 \times 8$$ |
| 训练每步 | 0.42 秒 | **0.031 秒（快约 13 倍）** |
| 3000 步总训练时间 | 约 21 分钟 | **约 1.6 分钟** |
| 生成 80 张（DDIM 50 步 + CFG） | 13.0 秒 | **1.1 秒**，另加解码 0.16 秒 |

- **速度**：U-Net 处理的空间位置从 1024 个变成 64 个，同样的网络每步快了 13 倍。解码器只在最后跑一次，开销可以忽略。
- **质量**：同样 3000 步，latent 版已经是清晰可辨的衣服、裤子、鞋；像素版还是一堆高频的条纹和噪点，明显没训够。这就是 1.2 节说的"感知压缩已经被自编码器做完了"：扩散模型在 latent 里只需要学"什么类别、什么形状"，而像素版要同时学形状和每个像素的纹理。
- **代价**：latent 版的结果偏平滑，纹理细节少，因为自编码器（只用了 L1）本身就重建不出细纹理（见第 4 节 $$f = 4$$ 那一行）。这正是 9.1 节说的"自编码器是质量的上限"；加上 LPIPS 和对抗损失会好很多。

注意两边的训练 loss（latent 版末尾约 0.19，像素版约 0.07）**不能直接比**：它们是不同空间里的噪声预测误差，像素图大面积是纯黑背景，噪声很好预测，所以数值偏低。比较要看生成结果。

### 7.3 解码器自己不会"生成"

<figure class="ddpm-fig">
  <img src="/deep-learning/assets/latent-diffusion/decode_noise_vs_sample.png" alt="直接解码高斯噪声与解码扩散生成的 latent">
  <figcaption>上：把 N(0, I) 的随机 latent 直接送进解码器；下：扩散模型从同样分布的噪声出发、去噪后再解码。</figcaption>
</figure>

直接解码高斯噪声只能得到一团没有意义的纹理。KL 权重只有 $$10^{-6}$$，真实 latent 的分布离标准正态差得很远，解码器只"认识"真实 latent。把噪声变成"像真实 latent 的东西"是扩散模型的工作，这也是 2.2 节易错点 2 的直观版本。

> **易错点（实验）**
>
> 1. **分辨率要先对齐。** 28×28 下采样两次是 7×7，再进 U-Net 的下采样就不整除了；所以先补到 32×32 再编码。真实场景里图像边长要是"自编码器下采样倍数 × U-Net 下采样倍数"的整数倍，SD 1.x 是 64 的倍数。
> 2. **latent 要缓存，不要每步重新编码。** 这里事先把 6 万张图的 $$(\mu, \log\sigma^2)$$ 算好存下来，训练时只做一次重参数化采样；否则每步都要过一次编码器，latent 版的速度优势会被吃掉一大块。
> 3. **对照组要控制变量。** 同一个 U-Net、同样的步数、同样的采样设置，只换输入空间；否则速度和质量的差别说明不了问题。

## 8. 论文结果，从 LDM 到 Stable Diffusion

### 8.1 论文里的结果

- **无条件生成**：CelebA-HQ $$256^2$$ 上 LDM-4 的 FID 为 5.11，当时的最好结果之一； [[1]](#ref-1)
- **类别条件 ImageNet**：LDM-4 加 classifier-free guidance（LDM-4-G）FID 3.60，和当时最强的像素空间扩散模型（ADM）相当，但参数量和训练算力都小得多；
- **文生图**：在 LAION-400M 上训练了一个 14.5 亿参数的 KL-f8 模型（$$\tau_\theta$$ 是从头训的 Transformer），配合 classifier-free guidance，在 MS-COCO 上和同期大得多的模型（如 GLIDE、Make-A-Scene）有可比的 FID；
- **超分、修补、语义合成**：都用拼接条件，用同一套框架覆盖了多种图像到图像的任务；
- **采样效率**：论文比较了不同 $$f$$ 的模型在 10～200 步 DDIM 下的 FID 和吞吐量，$$f = 4 \sim 8$$ 的模型在同样的采样时间里质量明显更好。

### 8.2 Stable Diffusion 改了什么

Stable Diffusion 1.x 是论文作者和 Stability AI、Runway 合作，把文生图 LDM 放大后的版本，结构上几乎没变：

| | 论文的文生图 LDM | Stable Diffusion 1.x |
|---|---|---|
| 自编码器 | KL-f8 | KL-f8，4 通道，$$s = 0.18215$$ |
| 文本编码器 $$\tau_\theta$$ | BERT tokenizer + Transformer，和 U-Net 一起训 | **冻结的** CLIP ViT-L/14 文本编码器，输出 `[77, 768]` |
| U-Net | 约 14.5 亿参数（含 $$\tau_\theta$$） | 约 8.6 亿参数 |
| 训练数据 | LAION-400M | LAION-5B 的高质量子集 |
| 分辨率 | $$256^2$$ | $$512^2$$（latent $$64^2$$） |
| 采样 | DDIM + CFG | 同左，以及后来更快的各种 scheduler |

最重要的变化是**换成了冻结的 CLIP 文本编码器**：它已经在大量图文对上学会了把文字映射到视觉语义，比从头训一个 Transformer 好得多，也省掉了文本编码器的训练成本。各个模块的逐层实现见 [Stable Diffusion 篇](/deep-learning/stable-diffusion/)；在冻结的 SD 上加空间控制见 [ControlNet 篇](/deep-learning/controlnet/)。

## 9. 代价、后续发展、要点回顾

### 9.1 代价：自编码器是质量的上限

LDM 的整条链路是 $$x \to z \to \tilde x$$。即使扩散模型完美地生成了一个"真实 latent"，解码出来的图也只能和自编码器的重建一样好：

- 很小的文字、远处的人脸、手指这类细节容易在 $$f = 8$$ 的压缩里丢掉；
- 高频纹理变软、局部颜色偏移、自编码器特有的伪影。

总结：**像素空间扩散直接建模每个像素，但贵；latent 扩散用一点重建保真度，换来大得多的效率和可扩展性。**

### 9.2 后来的发展

- **更好的自编码器**：SDXL 重新训练了 VAE；SD 3、FLUX 把 latent 通道从 4 加到 16，显著改善了文字和细节的重建；
- **换掉 U-Net**：DiT 及其后续（SD 3、FLUX、Sora 类视频模型）在同样的 latent 空间里用 Transformer 做去噪，"先压缩再扩散"的框架没变；
- **视频**：时空自编码器把视频在时间和空间上一起压缩，再在 3D latent 里扩散；
- **回到像素空间**：也有工作用更好的网络结构和训练技巧直接在像素空间做高分辨率扩散，以避开自编码器的瓶颈。

### 9.3 要点回顾

- LDM 没有改 DDPM 的任何公式，只是把扩散变量从 $$x_t$$ 换成了 $$z_t$$。
- 两阶段：先训一个只做**感知压缩**的自编码器（L1 + LPIPS + PatchGAN + 极弱的 KL 或 VQ 正则），冻结；再在它的 latent 里训扩散模型，学**语义**。
- latent 是保留空间结构的二维特征图，所以可以只温和地压缩；$$f = 4$$ 或 8 最好，太小训练慢，太大质量上不去。
- 编码后乘缩放因子 $$s = 1/\hat\sigma$$，让 latent 接近单位方差；解码前除回去。
- 条件两种接法：$$\tau_\theta$$ + cross-attention（文本、类别），或者直接拼到输入（分割图、低分辨率图、修补）。
- 训练要用编码器（或缓存的 latent），文生图推理只要解码器，且只跑一次。
- 代价是自编码器的重建上限：细小文字、人脸、手是 latent 扩散的老问题。
