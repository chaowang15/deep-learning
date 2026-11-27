---
date: 2026-11-26 21:00:00 -0800
keywords: >-
  Stable Diffusion 1.x, latent diffusion, CLIP tokenizer, CLIP text encoder, VAE, scaling factor, U-Net, cross-attention, TimeResBlock, DownBlock, UpBlock, scheduler, classifier-free guidance, training, inference, PyTorch.
summary: >-
  按数据流拆开经典 Stable Diffusion 1.x 的每个模块：Tokenizer + CLIP 文本编码、VAE 与 scaling factor、噪声调度与 scheduler、带时间编码和 cross-attention 的 U-Net（ResBlock、Down/Mid/UpBlock、完整网络），以及训练一步和带 CFG 的推理全流程。每个模块给出 tensor shape、可运行的 PyTorch 代码和易错点，并配有手绘的模块流程图。
layout: note
title: "Stable Diffusion 拆解：从 prompt 到图像的每个模块"
description: "Classic Stable Diffusion 1.x module by module: CLIP text encoding, VAE latents, scheduler, the conditional U-Net with cross-attention, training, and CFG inference, with shapes, code and diagrams."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig svg .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;fill:#475467}
.ddpm-fig svg .lane{font-size:14px;font-weight:600}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
.sd-img{margin:1.2em 0;text-align:center}
.sd-img img{max-width:100%;height:auto;border:1px solid #eaecf0;border-radius:6px}
.sd-img figcaption{color:#667085;font-size:.9em;margin-top:.4em}
</style>

# Stable Diffusion 拆解：从 prompt 到图像的每个模块

前两篇 [DDPM 从零实现](/deep-learning/ddpm-from-scratch/) 和 [DDIM](/deep-learning/ddim/) 讲的是扩散模型本身：怎么加噪、怎么训练一个预测噪声的网络、怎么一步步（或跳步）采样。Stable Diffusion 并没有改这套数学，它是在外面加了几样东西：


- **训练目标**：和 DDPM 完全相同，预测噪声的 MSE；
- **在 latent 空间里扩散**：先用 VAE 把图像压缩成小得多的 latent，在 latent 上加噪、去噪；
- **文本条件**：用 CLIP 文本编码器把 prompt 变成一串向量；
- **cross-attention**：在 U-Net 里让图像特征去"读"这串文本向量；
- **CFG**：推理时用 classifier-free guidance 加强文本的作用。

这一篇按数据流把这些模块逐个拆开，目标是**看懂一个 prompt 是怎么一步步变成一张图的，每一步的 tensor 是什么 shape，以及每个模块的核心代码怎么写**。以经典的 **Stable Diffusion 1.x（1.4 / 1.5）** 为准，生成 $$512 \times 512$$ 的图；SD 2.x、SDXL、SD3 的改动在相应位置简单提一下。

本文主要参考 Latent Diffusion 论文 [[1]](#ref-1)（SD 就是它在大规模图文数据上的版本）和官方仓库 [[2]](#ref-2)；直观的图解推荐 [[5]](#ref-5)。正文里用 [n] 标出具体出处，完整列表见文末「参考资料」。

## 0. 全文路线

| 节 | 模块 | 一句话 | 可训练？ |
|---|---|---|---|
| 1 | 总览 | 训练和推理两条数据流 | |
| 2 | Tokenizer + CLIP 文本编码器 | prompt → `[B, 77, 768]` 的逐 token 向量 | Tokenizer 无参数；CLIP 冻结 |
| 3 | VAE | 图像 `[B, 3, 512, 512]` ↔ latent `[B, 4, 64, 64]` | 冻结 |
| 4 | 噪声调度与 scheduler | 训练时一步加噪；推理时决定怎么走下一步 | 无参数 |
| 5 | U-Net（教学版） | $$(z_t, t, c) \to \hat\epsilon$$，时间走 ResBlock，文本走 cross-attention | **唯一训练的部分** |
| 6 | 精确的 SD 1.5 U-Net | Spatial Transformer、每个 block 2～3 层、12 条 skip，参数量和官方一致 | |
| 7 | 训练一步 | 编码 → 加噪 → 预测噪声 → MSE | |
| 8 | 推理全流程与 CFG | 编码一次 prompt，循环 U-Net + scheduler，最后解码一次；用 Diffusers 直接跑 | |
| 9–11 | 对照表、自检、要点回顾 | | |

第 5 节的 U-Net 是一个缩小版的教学实现（纯 PyTorch，可以直接跑），保留了 SD 的接口和信息流；第 6 节再把每个模块换成 SD 1.5 的真实结构，拼出参数量和官方完全相同的 U-Net；第 7、8 节的端到端代码直接用 Diffusers 的官方组件，输入输出约定和前面完全一致。

## 1. 总览：训练和推理两条数据流

<figure class="ddpm-fig">
<svg viewBox="0 0 760 450" role="img" aria-label="Stable Diffusion 训练和推理的数据流">
  <defs><marker id="sa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker></defs>
  <text x="10" y="22" class="lane">训练（每个 batch 一次）</text>
  <rect x="10" y="40" width="110" height="46" rx="8" fill="#fff" stroke="#475467"/><text x="65" y="60" text-anchor="middle">真实图像</text><text x="65" y="76" text-anchor="middle" class="mono">[B,3,512,512]</text>
  <rect x="150" y="40" width="110" height="46" rx="8" fill="#f2f4f7" stroke="#98a2b3"/><text x="205" y="60" text-anchor="middle">VAE 编码器</text><text x="205" y="76" text-anchor="middle" class="sm">冻结，乘 0.18215</text>
  <rect x="290" y="40" width="110" height="46" rx="8" fill="#fff" stroke="#475467"/><text x="345" y="60" text-anchor="middle">干净 latent</text><text x="345" y="76" text-anchor="middle" class="mono">[B,4,64,64]</text>
  <rect x="430" y="40" width="110" height="46" rx="8" fill="#fff7ed" stroke="#f79009"/><text x="485" y="60" text-anchor="middle">一步加噪</text><text x="485" y="76" text-anchor="middle" class="sm">随机 t 和噪声</text>
  <rect x="570" y="40" width="80" height="46" rx="8" fill="#eef4ff" stroke="#2a6fdb" stroke-width="2"/><text x="610" y="60" text-anchor="middle">U-Net</text><text x="610" y="76" text-anchor="middle" class="sm">训练</text>
  <rect x="675" y="40" width="75" height="46" rx="8" fill="#fef3f2" stroke="#d92d20"/><text x="712" y="60" text-anchor="middle">MSE</text><text x="712" y="76" text-anchor="middle" class="sm">和真实噪声</text>
  <line x1="120" y1="63" x2="148" y2="63" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="260" y1="63" x2="288" y2="63" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="400" y1="63" x2="428" y2="63" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="540" y1="63" x2="568" y2="63" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="650" y1="63" x2="673" y2="63" stroke="#475467" marker-end="url(#sa)"/>
  <rect x="10" y="120" width="110" height="46" rx="8" fill="#fff" stroke="#475467"/><text x="65" y="140" text-anchor="middle">图像的文字描述</text><text x="65" y="156" text-anchor="middle" class="sm">10% 换成空串</text>
  <rect x="150" y="120" width="110" height="46" rx="8" fill="#f2f4f7" stroke="#98a2b3"/><text x="205" y="140" text-anchor="middle">Tokenizer</text><text x="205" y="156" text-anchor="middle" class="mono">[B,77]</text>
  <rect x="290" y="120" width="110" height="46" rx="8" fill="#f2f4f7" stroke="#98a2b3"/><text x="345" y="140" text-anchor="middle">CLIP 文本编码器</text><text x="345" y="156" text-anchor="middle" class="sm">冻结</text>
  <rect x="430" y="120" width="110" height="46" rx="8" fill="#fff" stroke="#475467"/><text x="485" y="140" text-anchor="middle">文本条件</text><text x="485" y="156" text-anchor="middle" class="mono">[B,77,768]</text>
  <line x1="120" y1="143" x2="148" y2="143" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="260" y1="143" x2="288" y2="143" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="400" y1="143" x2="428" y2="143" stroke="#475467" marker-end="url(#sa)"/>
  <path d="M540,143 L610,143 L610,88" fill="none" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="10" y1="200" x2="750" y2="200" stroke="#eaecf0" stroke-width="2"/>
  <text x="10" y="232" class="lane">推理（生成一张图）</text>
  <rect x="10" y="250" width="110" height="46" rx="8" fill="#fff" stroke="#475467"/><text x="65" y="270" text-anchor="middle">两个 prompt</text><text x="65" y="286" text-anchor="middle" class="sm">正向、负向</text>
  <rect x="150" y="250" width="110" height="46" rx="8" fill="#f2f4f7" stroke="#98a2b3"/><text x="205" y="270" text-anchor="middle">CLIP 编码</text><text x="205" y="286" text-anchor="middle" class="sm">只算一次</text>
  <rect x="290" y="250" width="110" height="46" rx="8" fill="#fff" stroke="#475467"/><text x="345" y="270" text-anchor="middle">两份文本条件</text><text x="345" y="286" text-anchor="middle" class="mono">[2B,77,768]</text>
  <line x1="120" y1="273" x2="148" y2="273" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="260" y1="273" x2="288" y2="273" stroke="#475467" marker-end="url(#sa)"/>
  <rect x="10" y="340" width="110" height="46" rx="8" fill="#fff" stroke="#475467"/><text x="65" y="360" text-anchor="middle">随机 latent 噪声</text><text x="65" y="376" text-anchor="middle" class="mono">[B,4,64,64]</text>
  <rect x="290" y="325" width="250" height="76" rx="10" fill="#eef4ff" stroke="#2a6fdb" stroke-width="2"/>
  <text x="415" y="350" text-anchor="middle">U-Net（批大小翻倍）→ CFG 合并</text>
  <text x="415" y="370" text-anchor="middle">→ scheduler 更新 latent</text>
  <text x="415" y="390" text-anchor="middle" class="sm">重复 N 步（例如 50）</text>
  <line x1="120" y1="363" x2="288" y2="363" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="345" y1="296" x2="345" y2="323" stroke="#475467" marker-end="url(#sa)"/>
  <rect x="580" y="250" width="110" height="46" rx="8" fill="#f2f4f7" stroke="#98a2b3"/><text x="635" y="270" text-anchor="middle">VAE 解码器</text><text x="635" y="286" text-anchor="middle" class="sm">先除以 0.18215</text>
  <rect x="580" y="340" width="110" height="46" rx="8" fill="#ecfdf3" stroke="#12b76a"/><text x="635" y="360" text-anchor="middle">生成的图像</text><text x="635" y="376" text-anchor="middle" class="mono">[B,3,512,512]</text>
  <path d="M540,385 L560,385 L560,273 L578,273" fill="none" stroke="#475467" marker-end="url(#sa)"/>
  <line x1="635" y1="296" x2="635" y2="338" stroke="#475467" marker-end="url(#sa)"/>
  <text x="10" y="435" class="sm">训练不用 VAE 解码器；纯文生图推理不用 VAE 编码器。负向 prompt 默认是空串。</text>
</svg>
<figcaption>灰色是冻结或无参数的模块，蓝色是唯一需要训练的 U-Net。</figcaption>
</figure>

整个系统里只有 U-Net 在训练；其余模块要么没有参数（tokenizer、scheduler），要么是预训练好之后冻结的（CLIP 文本编码器、VAE）。用公式写，推理就是三行 [[1]](#ref-1)[[5]](#ref-5)：

$$
\hat\epsilon = \epsilon_\theta(z_t, t, c),\qquad
z_{t'} = \operatorname{SchedulerStep}(z_t, \hat\epsilon, t),\qquad
\hat x = D(z_0 / 0.18215)
$$

其中 $$z_t$$ 是 latent 空间里的带噪样本，$$c$$ 是文本条件，$$t'$$ 是 scheduler 选的下一个时间步，$$D$$ 是 VAE 解码器。

**全文的 shape 约定**（SD 1.x，$$512 \times 512$$）：

| 符号 | 含义 | 代码 | shape |
|---|---|---|---|
| $$x$$ | RGB 图像，归一化到 $$[-1, 1]$$ | `images` | `[B, 3, 512, 512]` |
| $$z_0$$ | 干净 latent（已乘 scaling factor） | `latents` | `[B, 4, 64, 64]` |
| $$z_t$$ | 时间步 $$t$$ 的带噪 latent | `noisy_latents` | `[B, 4, 64, 64]` |
| $$t$$ | 时间步，$$0 \dots 999$$ | `timesteps` | `[B]` |
| — | token 下标 | `input_ids` | `[B, 77]`，int64 |
| $$c$$ | 逐 token 的文本条件 | `text_embeddings` | `[B, 77, 768]` |
| $$\epsilon,\ \hat\epsilon$$ | 真实噪声、预测噪声 | `noise`, `predicted_noise` | `[B, 4, 64, 64]` |

## 2. 文本编码：Tokenizer + CLIP

### 2.1 是什么、为什么分两步

网络不能直接吃字符串，所以文本条件分两步 [[4]](#ref-4)：

1. **Tokenizer**：把字符串切成子词（BPE），查词表得到整数下标，补齐到固定长度 77。它是一个确定的预处理算法，**没有可学习参数**。
2. **CLIP 文本编码器**：一个 12 层的 Transformer，把每个下标变成一个 768 维向量，并且通过 self-attention 让每个 token 的向量带上上下文信息。

$$
\text{prompt 列表}\ \xrightarrow{\text{Tokenizer}}\ \text{token 下标}\ [B, 77]\ \xrightarrow{\text{CLIP}}\ c\ [B, 77, 768]
$$

SD 1.x 用的是 OpenAI CLIP ViT-L/14 的**文本部分**（约 1.2 亿参数）。CLIP 原本是图文对比学习训练出来的多模态模型，但 Stable Diffusion 只借用了它的文本编码器：图像编码器、图文相似度都不参与生成。

### 2.2 Tokenizer：77 个位置

77 个位置的组成是：一个开始符 `<BOS>`（id 49406）+ 最多 75 个内容 token + 一个结束符 `<EOS>`（id 49407），不够的用 padding 补齐。以 `"a photo of a cat"` 为例：

```text
位置:            0      1     2     3    4     5      6      7      8   ...  76
token:        <BOS>    a  photo   of    a   cat  <EOS>  <PAD>  <PAD> ... <PAD>
input_ids:    49406  320  1125   539  320  2368  49407  49407  49407 ... 49407
attention_mask:   1    1     1     1    1     1      1      0      0  ...     0
```

两个细节：

- **一个词不一定是一个 token。** BPE 会把不常见的词拆成几个子词，每个子词占一个位置，所以"75 个 token"不等于"75 个词"。
- **padding 和 `<EOS>` 用的是同一个 id 49407。** 只看 `input_ids` 分不清哪个是真正的结束符，要看 `attention_mask`。

### 2.3 CLIP 文本编码器：保留全部 77 个向量

编码器内部是标准的 Transformer：

$$
h_i^{(0)} = E_{\text{token}}[\,\text{id}_i\,] + p_i,\qquad
c = \operatorname{Transformer}\big(h^{(0)}\big)\ \in \mathbb{R}^{B \times 77 \times 768}
$$

输出的每个位置都是**带上下文的**向量：同一个 token `bank` 在 "river bank" 和 "investment bank" 里会得到不同的向量。

文本分类之类的任务通常只要一个句子向量 `[B, 768]`，但 Stable Diffusion 保留全部 `[B, 77, 768]`。原因在 U-Net 的 cross-attention 里（5.5 节）：**图像的每个空间位置要能去"看"不同的词**。生成 "a red car on a snowy road" 时，车所在的区域应该更多关注 `red`、`car`，背景区域更多关注 `snowy`、`road`。如果提前平均成一个向量，这种按位置选择词的能力就没有了。

所以代码里用的是 `last_hidden_state`，而不是 `pooler_output`（`[B, 768]`，只取 `<EOS>` 位置）：

```python
@torch.no_grad()
def encode_prompts(prompts, tokenizer, text_encoder, device):
    batch = tokenizer(
        list(prompts),
        padding="max_length",                    # 补齐到 77
        max_length=tokenizer.model_max_length,   # 77
        truncation=True,                         # 超过 77 的部分直接截掉
        return_tensors="pt",
    )
    input_ids = batch.input_ids.to(device)                    # [B, 77] int64
    # SD 1.x 的官方 pipeline 只传 input_ids，不传 attention_mask
    text_embeddings = text_encoder(input_ids).last_hidden_state   # [B, 77, 768]
    return text_embeddings
```

Tokenizer 和文本编码器返回的都不是单个 tensor，而是一个带多个字段的对象。各字段的用途：

```text
tokenizer(prompts, ...)  →  BatchEncoding
├── input_ids          [B, 77]        ← SD 实际使用
├── attention_mask     [B, 77]        ← SD 1.x 不使用
└── ...

text_encoder(input_ids)  →  BaseModelOutputWithPooling
├── last_hidden_state  [B, 77, 768]   ← U-Net 的 cross-attention 使用
├── pooler_output      [B, 768]       ← 句子向量，CLIP 检索、分类使用
├── hidden_states      可选，每一层的输出（"CLIP skip" 取倒数第二层就从这里拿）
└── attentions         可选，每一层的注意力权重
```

`pooler_output` 是最后一层在 `<EOS>` 位置的向量（经过最终的 LayerNorm）。只取一个位置也能代表整句话，是因为 CLIP 训练时就用这个位置做图文对比；而且经过多层 attention 后，`<EOS>` 已经看过它左边的所有 token。

> **易错点**
>
> 1. **77 是序列长度，768 是每个 token 的维度。** `input_ids` 是整数下标，不是 embedding，也不是 one-hot；下标的数值大小没有语义。
> 2. **超长的 prompt 会被静默截断。** 75 个 token 之后的内容进不了编码器，对生成没有任何影响。很多 WebUI 的"长 prompt"是把文本切成多段 77 分别编码再拼接，这是额外的工程技巧，不是模型本身的能力。
> 3. **padding 位置的向量并不是 0，U-Net 也会看到它们。** CLIP 文本编码器用的是因果（causal）attention，每个位置只看它左边的 token，所以 padding 不会影响前面真实 token 的向量；但 padding 位置自己的输出向量带着整句的信息，而 U-Net 的 cross-attention 对 77 个位置**都不加 mask**。这也是为什么 SD 1.x 的 pipeline 不传 `attention_mask`：模型就是在"不 mask"的条件下训练的，推理时保持一致即可。
> 4. **文本编码每次生成只算一次。** 所有去噪步共用同一个 $$c$$，循环里反复跑的只有 U-Net。
> 5. **`pooler_output` 找 `<EOS>` 的方法是对 `input_ids` 取 argmax。** `<EOS>` 的 id 49407 是词表里最大的，padding 也是 49407，而 argmax 返回第一个最大值的位置，正好是真正的结束符。自己改 tokenizer 的 padding id 时这个技巧会失效。
> 6. **不同版本的文本编码器不同。** SD 2.x 换成 OpenCLIP ViT-H，维度 1024，并且取的是**倒数第二层**的输出；SDXL 同时用两个文本编码器，把两者的输出拼起来，维度 2048。WebUI 里的 "CLIP skip" 指的就是取倒数第几层。

## 3. VAE：在 latent 空间里做扩散

### 3.1 是什么、为什么

在 $$512 \times 512 \times 3$$ 的像素空间直接做扩散，U-Net 每一步都要处理 78 万个数，大部分算力花在人眼几乎看不出差别的高频细节上。Latent Diffusion（Stable Diffusion 的前身）的做法是：先训练一个自编码器，把图像压缩成小得多的 latent，再**在 latent 空间里做扩散**。 [[1]](#ref-1)

SD 1.x 的 VAE 在空间上缩小 8 倍，通道数为 4：

$$
x \in \mathbb{R}^{3 \times 512 \times 512}\ \xrightarrow{\ E\ }\ z \in \mathbb{R}^{4 \times 64 \times 64}\ \xrightarrow{\ D\ }\ \hat x \in \mathbb{R}^{3 \times 512 \times 512}
$$

元素个数少了 48 倍。VAE 是单独训练好的（重建损失 + 感知损失 + 对抗损失 + 很小的 KL 正则），训练扩散模型时冻结。注意它在 U-Net **外面**，并不替代 U-Net 自己的下采样路径。

### 3.2 Scaling factor

VAE 编码出来的 latent 标准差大约是 5.5，而扩散模型的前向过程假设数据大致是单位方差（这样 $$z_T$$ 才接近 $$\mathcal{N}(0, I)$$，见 DDPM 篇第 2 节数据归一化）。所以 SD 在 VAE latent 和扩散 latent 之间乘一个常数：

$$
z_0 = 0.18215 \cdot E(x),\qquad \hat x = D\big(z_0 / 0.18215\big)
$$

$$0.18215 \approx 1/5.49$$，是在训练数据上估计出来的 latent 标准差的倒数。代码里应该读 `vae.config.scaling_factor`，不要写死（SDXL 的是 0.13025）。

```python
@torch.no_grad()
def encode_images(images, vae, sample_posterior=True):
    # images: [B, 3, H, W]，范围 [-1, 1]
    posterior = vae.encode(images).latent_dist          # 均值、方差都是 [B, 4, H/8, W/8]
    raw = posterior.sample() if sample_posterior else posterior.mode()
    return raw * vae.config.scaling_factor              # z0 = 0.18215 · E(x)，[B, 4, H/8, W/8]


@torch.no_grad()
def decode_latents(latents, vae):
    decoded = vae.decode(latents / vae.config.scaling_factor).sample   # [B, 3, 8h, 8w]，约在 [-1, 1]
    return (decoded / 2 + 0.5).clamp(0, 1)                             # [-1, 1] → [0, 1]，用于显示或保存
```

什么时候用编码器、什么时候用解码器：

| 阶段 | VAE 编码器 | VAE 解码器 |
|---|---|---|
| 训练扩散模型 | 用（或事先把 latent 算好缓存），冻结 | 不用 |
| 文生图推理 | **不用**，直接从随机 latent 开始 | 最后用一次 |
| 图生图、局部重绘 | 用来编码输入图 | 最后用一次 |

> **易错点**
>
> 1. **编码后要乘、解码前要除 scaling factor。** 忘了乘，训练数据的方差是 30 倍左右，和噪声尺度完全不匹配；忘了除，解码出来是一片过曝、发灰或者全是色块的图。这两个错误都不会报错。
> 2. **`sample()` 还是 `mode()`？** 训练时一般 `sample()`，相当于一点数据增强；做重建检查、图像编辑时用 `mode()`（均值），结果是确定的。
> 3. **图像尺寸必须是 8 的倍数**（经过 U-Net 后实际上要求 latent 能被 8 整除，也就是图像是 64 的倍数，见 5.7 节）。
> 4. **4 个 latent 通道不是 RGBA。** 它们是 VAE 学出来的抽象特征，直接当图片显示会大致看出轮廓，但颜色没有意义。

### 3.3 编码器输出的是一个分布

上面的 `vae.encode(images).latent_dist` 不是一个 tensor，而是一个对角高斯分布 $$q(z \mid x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$$。VAE 编码器的卷积网络实际输出 8 个通道（`moments`，`[B, 8, 64, 64]`），前 4 个是均值 $$\mu$$，后 4 个是 $$\log\sigma^2$$：

$$
z = \mu + \sigma \odot \xi,\quad \xi \sim \mathcal{N}(0, I),\qquad
\operatorname{KL}\big(q(z \mid x)\,\|\,\mathcal{N}(0, I)\big) = \tfrac12 \sum \big(\mu^2 + \sigma^2 - 1 - \log\sigma^2\big)
$$

```python
class DiagonalGaussianPosterior:
    def __init__(self, moments):                          # moments: [B, 8, h, w]，VAE 编码器的输出
        self.mean, self.logvar = moments.chunk(2, dim=1)  # μ、log σ²，各 [B, 4, h, w]
        self.logvar = self.logvar.clamp(-30.0, 20.0)      # 防止 exp 溢出

    def sample(self, generator=None):                     # z = μ + σ·ξ（重参数化）
        std = torch.exp(0.5 * self.logvar)
        noise = torch.randn(self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + std * noise

    def mode(self):                                       # 高斯的众数就是均值
        return self.mean

    def kl(self):                                         # 和 N(0, I) 的 KL，每个样本一个数，[B]
        return 0.5 * (self.mean.square() + self.logvar.exp() - 1.0 - self.logvar).flatten(1).sum(dim=1)
```

KL 项只在**训练 VAE 本身**时用到。整个系统其实分两个阶段训练，各自的损失完全不同：

| 阶段 | 训练什么 | 损失 | 目的 |
|---|---|---|---|
| VAE 预训练 | VAE 编码器 + 解码器 | 重建 + 感知 + 对抗 + 很小的 KL | 学一个保留视觉细节的压缩 |
| 扩散训练（第 7 节） | 只有 U-Net | latent 上预测噪声的 MSE | 学带条件的去噪 |
| 文生图推理 | 无 | 无 | 去噪后解码一次 |

> **易错点**
>
> 1. **训练好的 VAE 的 $$\sigma$$ 很小。** KL 的权重只有 $$10^{-6}$$ 量级，编码器几乎是确定性的，所以 `sample()` 和 `mode()` 的差别很小，说它是"相当于一点数据增强"也只是一点点。
> 2. **SD 的 VAE 没有往标准正态拉。** 正因为 KL 很弱，latent 的方差远大于 1（约 $$5.5^2$$），才需要 3.2 节的 scaling factor。

## 4. 噪声调度与 scheduler

### 4.1 训练：和 DDPM 一样的一步加噪

在 latent 空间里，前向过程和 DDPM 完全相同（DDPM 篇第 4 节）：

$$
z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon \sim \mathcal{N}(0, I)
$$

Diffusers 里这一步叫 `scheduler.add_noise(z0, noise, t)`，就是 DDPM 篇里的 `q_sample`。不同的是 SD 1.x 的噪声表：$$T = 1000$$，用的是 **scaled linear** 调度，在 $$\sqrt\beta$$ 上线性插值：

$$
\beta_t = \Big(\sqrt{0.00085} + \tfrac{t-1}{T-1}\big(\sqrt{0.012} - \sqrt{0.00085}\big)\Big)^2
$$

```python
T = 1000
betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, T) ** 2   # scaled_linear，[T]
alpha_bars = torch.cumprod(1 - betas, dim=0)                    # ᾱ_t，[T]
```

![SD 1.x 噪声调度](/deep-learning/assets/stable-diffusion/schedule.png)

和 DDPM 原论文的线性调度（$$10^{-4} \to 0.02$$）相比，SD 的 $$\beta$$ 整体更小，加噪更温和：$$t = 500$$ 时 $$\bar\alpha$$ 是 0.28（DDPM 是 0.08）。代价是最后一步 $$\sqrt{\bar\alpha_{1000}} \approx 0.068$$，并没有完全变成纯噪声（DDPM 是 0.006），这一点在易错点 3 里会再提。

### 4.2 推理：scheduler 决定怎么走下一步

推理时 scheduler 做三件事：选出要走的时间步（例如从 1000 个里选 50 个）、按需缩放 U-Net 的输入、根据 U-Net 的输出算出下一个 latent。

```python
scheduler.set_timesteps(50)                       # 例如 [981, 961, ..., 1]
latents = latents * scheduler.init_noise_sigma    # DDIM 是 1；Euler 类是 σ_max
for t in scheduler.timesteps:
    model_input = scheduler.scale_model_input(latents, t)        # DDIM 不变；Euler 类要缩放
    eps = unet(model_input, t, encoder_hidden_states=text_embeddings).sample
    latents = scheduler.step(eps, t, latents).prev_sample        # 走到下一个选中的时间步
```

U-Net 决定"预测什么"，scheduler 决定"怎么走到下一步"。所以同一个 U-Net 可以换不同的 scheduler：

| Scheduler | 本质 | 常用步数 |
|---|---|---|
| DDPM | 随机的马尔可夫反向过程（[DDPM 篇](/deep-learning/ddpm-from-scratch/) 第 7 节） | 1000 |
| DDIM | $$\eta = 0$$ 时是 ODE 的一阶 Euler 解法（[DDIM 篇](/deep-learning/ddim/)） | 20～50 |
| PNDM | SD 1.x 官方默认，伪线性多步法 | 50 |
| Euler / Euler Ancestral | 在 $$\sigma$$ 参数化下的 ODE / SDE 解法 | 20～30 |
| DPM-Solver++ | 针对扩散 ODE 的高阶解法 | 15～25 |

> **易错点**
>
> 1. **`num_train_timesteps` 和 `num_inference_steps` 不是一回事。** 前者是训练时的 $$T = 1000$$，后者是推理时实际走几步。
> 2. **不同 scheduler 的循环写法不能混用。** Euler 类的 scheduler 要求初始 latent 乘 `init_noise_sigma`（约 14.6），每步输入要过 `scale_model_input`；DDIM 这两步都是恒等的。照着 DDIM 写循环、再换成 Euler scheduler，生成的图会是一片噪声。始终调用这两个接口最保险。
> 3. **SD 1.x 的终点不是纯噪声。** 训练时最大噪声水平下仍残留约 7% 的信号，但推理却从纯高斯噪声开始。这会让模型很难生成特别亮或特别暗的图（平均亮度被"锁"在中间）。后来的工作提出把调度改成终点信噪比为 0（zero terminal SNR），并配合 $$v$$-prediction 重新训练。
> 4. **预测类型要和模型一致。** SD 1.x 和 SD 2 base 是 $$\epsilon$$-prediction，SD 2 的 768 版本是 $$v$$-prediction（DDPM 篇第 10 节）。scheduler 的 `prediction_type` 配错不会报错，只会生成乱码一样的图。

### 4.3 不用 Diffusers：预测目标和手写的一步反向更新

scheduler 没有可学习参数，自己写也就几行。训练时的监督目标有三种选法，网络结构不变，只是目标不同：

| `prediction_type` | 目标 | 公式 |
|---|---|---|
| `epsilon`（SD 1.5） | 加进去的噪声 | $$\epsilon$$ |
| `v_prediction`（SD 2 768） | 速度 | $$v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,z_0$$ |
| `sample` | 干净 latent | $$z_0$$ |

```python
def training_target(z0, noise, t, alpha_bars, prediction_type="epsilon"):
    a = alpha_bars[t].view(-1, 1, 1, 1)                        # ᾱ_t，广播到 [B, 1, 1, 1]
    if prediction_type == "epsilon":
        return noise                                           # ε
    if prediction_type == "v_prediction":
        return a.sqrt() * noise - (1 - a).sqrt() * z0          # v = √ᾱ ε − √(1−ᾱ) z0
    if prediction_type == "sample":
        return z0
    raise ValueError(prediction_type)
```

推理时最简单的 scheduler 就是 DDPM 的相邻一步（推导见 [DDPM 篇](/deep-learning/ddpm-from-scratch/) 第 7 节）：先由 $$\hat\epsilon$$ 估计 $$\hat z_0$$，再代入后验均值和方差：

$$
\hat z_0 = \frac{z_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}},\qquad
\mu = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\hat z_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,z_t,\qquad
z_{t-1} = \mu + \sqrt{\tilde\beta_t}\,\xi
$$

其中 $$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$。代入 $$\hat z_0$$ 化简，$$\mu$$ 就是 DDPM 篇里更常见的 $$\frac{1}{\sqrt{\alpha_t}}\big(z_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon\big)$$。

```python
def ddpm_step(z_t, eps, t, betas, alpha_bars, generator=None):   # t 是整数，z_t → z_{t-1}
    beta, a_bar = betas[t], alpha_bars[t]
    a_bar_prev = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0)
    z0_hat = (z_t - (1 - a_bar).sqrt() * eps) / a_bar.sqrt()                       # ẑ0
    mean = (a_bar_prev.sqrt() * beta / (1 - a_bar)) * z0_hat \
         + ((1 - beta).sqrt() * (1 - a_bar_prev) / (1 - a_bar)) * z_t             # μ，α_t = 1 − β_t
    if t == 0:
        return mean                                                                 # 最后一步不加噪声
    var = beta * (1 - a_bar_prev) / (1 - a_bar)                                     # β̃_t
    noise = torch.randn(z_t.shape, generator=generator, device=z_t.device, dtype=z_t.dtype)
    return mean + var.sqrt() * noise
```

把 8.2 节循环里的 `scheduler.step` 换成它、时间步换成 `range(999, -1, -1)`，就是完全手写的 1000 步 DDPM 采样。

> **易错点**
>
> 1. **这个更新只适用于相邻时间步。** 跳步时（50 步）要把 $$\bar\alpha_{t-1}$$ 换成下一个选中时间步的 $$\bar\alpha$$，或者直接用 DDIM（[DDIM 篇](/deep-learning/ddim/) 第 4 节）。
> 2. **目标、损失和采样器必须用同一种 `prediction_type`。** 同一个 checkpoint 不能随意换：用 $$\epsilon$$ 训练的模型当 $$v$$ 去解读，采出来是噪声。

## 5. U-Net：带时间和文本条件的去噪网络

### 5.1 三条输入、一个输出

U-Net 的接口和 DDPM 篇里的 Tiny U-Net 一样是"图进、图出"，只是多了文本条件：

$$
\hat\epsilon = \epsilon_\theta(z_t, t, c):\qquad
[B, 4, 64, 64],\ [B],\ [B, 77, 768]\ \longrightarrow\ [B, 4, 64, 64]
$$

三种信息从三个地方进入网络：

| 信息 | 进入方式 | 作用 |
|---|---|---|
| 带噪 latent $$z_t$$ | 卷积编码器–解码器的主干 | 当前图像内容 |
| 时间步 $$t$$ | 正弦编码 → MLP → 加到每个 ResBlock 里 | 告诉共享的网络当前噪声有多大 |
| 文本 $$c$$ | 选定分辨率上的 cross-attention | 告诉网络要生成什么 |

输出必须和输入同 shape：网络预测的是加在 $$z_0$$ 每个元素上的噪声。

下面先实现基本模块，再拼成完整的 U-Net。所有代码共用这几个 import 和一个 GroupNorm 工厂函数：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(num_channels):
    # 扩散模型的 batch 往往很小，用 GroupNorm 而不是 BatchNorm：不依赖 batch 统计量
    return nn.GroupNorm(num_groups=32, num_channels=num_channels)
```

### 5.2 时间步编码

和 DDPM 篇第 5.2 节完全相同的正弦编码，再过一个两层 MLP，得到一个所有 ResBlock 共用的时间向量：

$$
\operatorname{emb}(t) = \big[\cos(t\,\omega_0), \dots, \cos(t\,\omega_{D/2-1}),\ \sin(t\,\omega_0), \dots, \sin(t\,\omega_{D/2-1})\big],\qquad
\omega_i = 10000^{-i/(D/2)}
$$

```python
def sinusoidal_timestep_embedding(timesteps, dim, max_period=10_000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / half)  # ω_i，[D/2]
    angles = timesteps.float()[:, None] * freqs[None, :]                                           # t·ω_i，[B, D/2]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)                                # [B, D]


class TimestepEncoder(nn.Module):
    def __init__(self, base_dim, time_dim):
        super().__init__()
        self.base_dim = base_dim
        self.mlp = nn.Sequential(nn.Linear(base_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

    def forward(self, timesteps):                                  # [B]
        return self.mlp(sinusoidal_timestep_embedding(timesteps, self.base_dim))   # [B, time_dim]
```

SD 1.5 里 `base_dim = 320`、`time_dim = 1280`（4 倍）。cos 在前、sin 在后，以及频率的分母用 $$D/2$$ 还是 $$D/2 - 1$$，不同实现各不相同，只要训练和推理一致就没有影响；但**加载别人的权重时必须完全照抄**。

### 5.3 TimeResBlock：把时间注入特征图

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/time_res_block.png"><img src="/deep-learning/assets/stable-diffusion/time_res_block.png" alt="TimeResBlock 流程图"></a>
<figcaption>TimeResBlock：时间向量投影成每个通道一个偏置，加到第一个卷积的输出上；输入通过 skip 投影直接加到输出。</figcaption>
</figure>

ResBlock 是 U-Net 的基本卷积单元。时间向量 `[B, time_dim]` 经过一个线性层投影到输出通道数，reshape 成 `[B, C_out, 1, 1]`，广播加到所有空间位置上：

$$
h = \operatorname{Conv}_1\big(\operatorname{SiLU}(\operatorname{GN}(x))\big) + W_t\,\operatorname{SiLU}(\operatorname{emb}),\qquad
\text{out} = \operatorname{Conv}_2\big(\operatorname{SiLU}(\operatorname{GN}(h))\big) + \operatorname{skip}(x)
$$

```python
class TimeResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # 通道数变了就用 1×1 卷积对齐，否则恒等
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_embedding):          # x: [B, C_in, H, W]，time_embedding: [B, time_dim]
        h = self.conv1(F.silu(self.norm1(x)))                                  # [B, C_out, H, W]
        h = h + self.time_projection(time_embedding)[:, :, None, None]         # + [B, C_out, 1, 1]，广播
        h = self.conv2(F.silu(self.norm2(h)))                                  # [B, C_out, H, W]
        return h + self.skip(x)                                                # 残差
```

和 DDPM 篇的 TimeResBlock 相比，这里用的是 **pre-activation** 顺序（GroupNorm → SiLU → Conv），也是 SD 实际使用的顺序。

### 5.4 下采样和上采样

```python
class Downsample(nn.Module):              # [B, C_in, H, W] → [B, C_out, H/2, W/2]
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):                # [B, C_in, H, W] → [B, C_out, 2H, 2W]
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # 先最近邻放大再卷积，避免转置卷积常见的棋盘格伪影
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))
```

### 5.5 Spatial cross-attention：把文本注入特征图

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/spatial_cross_attention.png"><img src="/deep-learning/assets/stable-diffusion/spatial_cross_attention.png" alt="Spatial Cross Attention 流程图"></a>
<figcaption>Spatial cross-attention：图像特征展平成 H·W 个 token 作为 query，文本的 77 个向量作为 key 和 value。</figcaption>
</figure>

这是文本控制图像的唯一入口。把特征图的每个空间位置当成一个 token，作为 **query**；文本的 77 个向量作为 **key 和 value**：

$$
Q = X W_Q\ \ [B, HW, C],\qquad K = c\,W_K,\ \ V = c\,W_V\ \ [B, 77, C]
$$

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\Big(\frac{Q K^\top}{\sqrt{d}}\Big) V
$$

注意力矩阵是 `[B, heads, HW, 77]`：**每个空间位置在 77 个词上的一个概率分布**，决定这个位置从哪些词里取信息。多头时把 $$C$$ 拆成 `heads × head_dim`，$$d$$ 是 `head_dim`。

```python
class SpatialCrossAttention(nn.Module):
    def __init__(self, channels, context_dim, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads, self.head_dim = num_heads, channels // num_heads
        self.norm = group_norm(channels)
        self.to_q = nn.Linear(channels, channels, bias=False)       # 图像 → Q
        self.to_k = nn.Linear(context_dim, channels, bias=False)    # 文本 768 → K
        self.to_v = nn.Linear(context_dim, channels, bias=False)    # 文本 768 → V
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x, context):                # x: [B, C, H, W]，context: [B, L, D_text]
        B, C, H, W = x.shape
        L = context.shape[1]
        tokens = self.norm(x).flatten(2).transpose(1, 2)           # [B, C, H, W] → [B, HW, C]

        # 拆成多头：[B, N, C] → [B, N, heads, head_dim] → [B, heads, N, head_dim]
        q = self.to_q(tokens).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # QKᵀ/√d，[B, heads, HW, L]
        weights = scores.softmax(dim=-1)                              # 每个位置在 L 个词上归一化
        out = weights @ v                                             # [B, heads, HW, head_dim]

        out = out.transpose(1, 2).reshape(B, H * W, C)                # 合并多头 → [B, HW, C]
        out = self.to_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)                 # 先换回 [B, C, HW] 再 reshape
        return x + out                                                # 残差，[B, C, H, W]
```

> **易错点**
>
> 1. **Q 来自图像，K、V 来自文本。** 反过来写 shape 也能对上（只要维度凑巧），但意思完全变了。`to_k`、`to_v` 的输入维度是文本的 768，不是图像的通道数。
> 2. **展平后要先 `transpose` 再 `reshape`。** `[B, C, H, W]` 直接 `reshape(B, HW, C)` 会把通道和空间位置混在一起，shape 对、结果错（DDPM 篇 5.5 节同样的坑）。换回来时同理。
> 3. **softmax 在最后一维（77 个词）上做。** 每个空间位置在所有词上的权重和为 1，而不是每个词在所有位置上和为 1。
> 4. **cross-attention 比 self-attention 便宜得多。** 注意力矩阵是 $$HW \times 77$$；同分辨率的 self-attention 是 $$HW \times HW$$，在 $$64 \times 64$$ 上是 $$4096^2 \approx 1700$$ 万个元素，每个头一份。这也是 SD 1.5 的最高分辨率 attention 最吃显存的原因。
> 5. **真实的 SD 用的是 Spatial Transformer。** 每个 attention 位置其实是：GroupNorm → 1×1 投影 → [LayerNorm → **self-attention** → LayerNorm → **cross-attention** → LayerNorm → 前馈网络（GEGLU）] → 1×1 投影 → 残差。这里只保留了最关键的 cross-attention。

### 5.6 DownBlock、MidBlock、UpBlock

把上面的模块组合成三种可复用的 block。每种 block 都在固定的分辨率上做"ResBlock + cross-attention"，区别在于数据怎么进出：

- **DownBlock**：处理完后**保存一份 skip 特征**，再按需下采样（`add_downsample`）。
- **MidBlock**：最低分辨率上的"ResBlock → attention → ResBlock"，不改变 shape。
- **UpBlock**：先把输入和对应分辨率的 skip **拼接**，再处理，最后按需上采样（`add_upsample`）。

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/down_block.png"><img src="/deep-learning/assets/stable-diffusion/down_block.png" alt="DownBlock 流程图"></a>
</figure>

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/mid_block.png"><img src="/deep-learning/assets/stable-diffusion/mid_block.png" alt="MidBlock 流程图"></a>
</figure>

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/up_block.png"><img src="/deep-learning/assets/stable-diffusion/up_block.png" alt="UpBlock 流程图"></a>
<figcaption>DownBlock、MidBlock、UpBlock 的手绘流程图（点击图片可查看大图）。</figcaption>
</figure>

图和下面代码有两处顺序上的差别，以代码为准：

- **DownBlock 里通道数在哪变。** 图中 ResBlock 不变通道，下采样时 $$C \to 2C$$；代码里是 ResBlock 把 `C_in` 变成 `C_out`，下采样只改分辨率。两者整体的 shape 变化一样，后一种和 Diffusers 的写法一致。
- **UpBlock 里先拼接还是先上采样。** 图中是"先上采样、再和 skip 拼接"，输出标的 `[B,2C,H/2,W/2]` 也应为 `[B,C,H,W]`；代码是"先和**同分辨率**的 skip 拼接、处理，最后再上采样"，这也是 Diffusers 的顺序。两种写法都能工作，区别只在 skip 和哪个分辨率的特征配对。

```python
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, context_dim, num_heads, add_downsample=True):
        super().__init__()
        self.res_block = TimeResBlock(in_channels, out_channels, time_dim)
        self.attention = SpatialCrossAttention(out_channels, context_dim, num_heads)
        self.downsample = Downsample(out_channels, out_channels) if add_downsample else None   # 最后一个 block 不下采样

    def forward(self, x, time_embedding, text_embeddings):     # x: [B, C_in, H, W]
        x = self.res_block(x, time_embedding)                  # [B, C_out, H, W]
        x = self.attention(x, text_embeddings)                 # [B, C_out, H, W]
        skip = x                                               # 下采样之前保存，留给同分辨率的 UpBlock
        if self.downsample is not None:
            x = self.downsample(x)                             # [B, C_out, H/2, W/2]
        return x, skip


class MidBlock(nn.Module):
    def __init__(self, channels, time_dim, context_dim, num_heads):
        super().__init__()
        self.res_block1 = TimeResBlock(channels, channels, time_dim)
        self.attention = SpatialCrossAttention(channels, context_dim, num_heads)
        self.res_block2 = TimeResBlock(channels, channels, time_dim)

    def forward(self, x, time_embedding, text_embeddings):     # shape 不变
        x = self.res_block1(x, time_embedding)
        x = self.attention(x, text_embeddings)
        return self.res_block2(x, time_embedding)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_dim, context_dim, num_heads, add_upsample=True):
        super().__init__()
        self.res_block = TimeResBlock(in_channels + skip_channels, out_channels, time_dim)   # 输入是拼接后的通道数
        self.attention = SpatialCrossAttention(out_channels, context_dim, num_heads)
        self.upsample = Upsample(out_channels, out_channels) if add_upsample else None       # 最后一个 block 不上采样

    def forward(self, x, skip, time_embedding, text_embeddings):   # x、skip 的 H、W 必须相同
        x = torch.cat([x, skip], dim=1)                        # [B, C_in + C_skip, H, W]
        x = self.res_block(x, time_embedding)                  # [B, C_out, H, W]
        x = self.attention(x, text_embeddings)                 # [B, C_out, H, W]
        if self.upsample is not None:
            x = self.upsample(x)                               # [B, C_out, 2H, 2W]
        return x
```

### 5.7 拼成完整的 U-Net

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/unet_overview.png"><img src="/deep-learning/assets/stable-diffusion/unet_overview.png" alt="Stable Diffusion U-Net 总流程图"></a>
<figcaption>以 4 个分辨率层级为例的完整 U-Net：红色虚线是时间向量，灰色虚线是文本条件，横向箭头是 skip 连接。图中 x_skip4 的 shape 应为 [B,4C,H/8,W/8]（DownBlock4 不下采样，skip 和它的输出同分辨率）。</figcaption>
</figure>

用 `channel_multipliers` 同时决定层级数和每层的通道数。$$N$$ 个分辨率层级就有 $$N$$ 个 DownBlock 和 $$N$$ 个 UpBlock，但**只有前 $$N-1$$ 个会下采样、上采样**：最后一个 DownBlock 和第一个 UpBlock 都在最低分辨率上。

```python
class ConditionalUNet(nn.Module):
    def __init__(self, latent_channels=4, base_channels=64, channel_multipliers=(1, 2, 4, 4),
                 context_dim=768, num_heads=8):
        super().__init__()
        chs = [base_channels * m for m in channel_multipliers]     # 例如 [64, 128, 256, 256]
        n = len(chs)
        self.num_sampling_operations = n - 1
        time_dim = base_channels * 4
        self.time_encoder = TimestepEncoder(base_channels, time_dim)
        self.input_conv = nn.Conv2d(latent_channels, chs[0], 3, padding=1)

        # 下行：第 i 个 block 输出 chs[i]，只有最后一个不下采样
        self.down_blocks = nn.ModuleList([
            DownBlock(chs[max(i - 1, 0)], chs[i], time_dim, context_dim, num_heads, add_downsample=i < n - 1)
            for i in range(n)
        ])
        self.mid_block = MidBlock(chs[-1], time_dim, context_dim, num_heads)
        # 上行：按分辨率从低到高，第 i 层的输入来自更深一层（最深一层来自 mid），skip 来自同一层
        self.up_blocks = nn.ModuleList([
            UpBlock(chs[min(i + 1, n - 1)], chs[i], chs[i], time_dim, context_dim, num_heads, add_upsample=i > 0)
            for i in reversed(range(n))
        ])
        self.output_norm = group_norm(chs[0])
        self.output_conv = nn.Conv2d(chs[0], latent_channels, 3, padding=1)

    def forward(self, noisy_latent, timesteps, text_embeddings):
        # noisy_latent: [B, 4, H, W]，timesteps: [B]，text_embeddings: [B, L, 768]
        factor = 2 ** self.num_sampling_operations
        assert noisy_latent.shape[-1] % factor == 0 and noisy_latent.shape[-2] % factor == 0
        t_emb = self.time_encoder(timesteps)                       # [B, time_dim]，所有 block 共用
        x = self.input_conv(noisy_latent)                          # [B, chs[0], H, W]

        skips = []
        for block in self.down_blocks:                             # 每个 DownBlock 存一个 skip
            x, skip = block(x, t_emb, text_embeddings)
            skips.append(skip)
        x = self.mid_block(x, t_emb, text_embeddings)              # [B, chs[-1], H/2^(n-1), W/2^(n-1)]
        for block, skip in zip(self.up_blocks, reversed(skips)):  # 倒序取出，分辨率一一对应
            x = block(x, skip, t_emb, text_embeddings)
        return self.output_conv(F.silu(self.output_norm(x)))       # [B, 4, H, W]
```

以 `base_channels=64`、`channel_multipliers=(1, 2, 4, 4)`、输入 `[B, 4, 64, 64]` 为例，shape 的变化是：

```text
input_conv          [B,   4, 64, 64] → [B,  64, 64, 64]
DownBlock 1         → skip1 [B,  64, 64, 64]，下采样 → [B,  64, 32, 32]
DownBlock 2         → skip2 [B, 128, 32, 32]，下采样 → [B, 128, 16, 16]
DownBlock 3         → skip3 [B, 256, 16, 16]，下采样 → [B, 256,  8,  8]
DownBlock 4         → skip4 [B, 256,  8,  8]，不下采样 → [B, 256,  8,  8]
MidBlock                                          → [B, 256,  8,  8]
UpBlock (+skip4)    cat [B, 512, 8, 8]   → [B, 256,  8,  8]，上采样 → [B, 256, 16, 16]
UpBlock (+skip3)    cat [B, 512, 16, 16] → [B, 256, 16, 16]，上采样 → [B, 256, 32, 32]
UpBlock (+skip2)    cat [B, 384, 32, 32] → [B, 128, 32, 32]，上采样 → [B, 128, 64, 64]
UpBlock (+skip1)    cat [B, 192, 64, 64] → [B,  64, 64, 64]，不上采样
output_conv                                       → [B,   4, 64, 64]
```

> **易错点**
>
> 1. **skip 要倒序取出。** 最先保存的 skip1 是最高分辨率的，要留给最后一个 UpBlock。用 `reversed(skips)` 或者栈的 `pop()`。
> 2. **UpBlock 的输入通道数是"上一层输出 + skip"。** 漏算 skip 的通道，`TimeResBlock` 的第一个卷积会报通道不匹配。
> 3. **latent 尺寸要能被 $$2^{N-1}$$ 整除。** 否则某次下采样会向下取整，上采样回来和 skip 的尺寸对不上，`torch.cat` 报错。SD 1.5 是 3 次下采样，latent 要能被 8 整除，也就是图像尺寸是 64 的倍数。
> 4. **"最后一个 block 不采样"不等于"它什么也不做"。** 它照样有 ResBlock 和 attention，只是保持分辨率。
> 5. **GroupNorm(32) 要求通道数能被 32 整除**，cross-attention 要求通道数能被头数整除。`base_channels` 不要随便取。

### 5.8 教学版和真实 SD 1.5 U-Net 的区别

真实的 SD 1.5 U-Net（约 8.6 亿参数）拓扑和上面一样，主要差在规模和每个 block 里的层数（第 6 节逐个模块实现）：

| | 本文教学版 | SD 1.5 |
|---|---|---|
| 每层通道数 | `base × (1, 2, 4, 4)` | `(320, 640, 1280, 1280)` |
| 时间向量维度 | `4 × base` | 1280 |
| 每个 block 里的层数 | 1 个 ResBlock + 1 个 attention | 下行每个 block 2 组 ResBlock + Transformer；上行每个 block 3 组 |
| 哪些 block 有 attention | 全部 | 最低分辨率（8×8）的 DownBlock 4 和 UpBlock 1 没有 cross-attention；MidBlock 有 |
| skip 数量 | 每个 DownBlock 1 个，共 4 个 | 每个 ResBlock 输出和每次下采样输出都存，加上输入卷积，共 12 个；上行 12 个 ResBlock 各取一个 |
| attention 结构 | 只有 cross-attention | Spatial Transformer：self-attention + cross-attention + 前馈网络 |
| 注意力头 | 8 个头 | 8 个头（配置里的字段名是 `attention_head_dim=8`，实际含义是头数，这是一个有名的历史命名错误） |

如果只是想用预训练的 U-Net，不需要自己写，直接 `UNet2DConditionModel.from_pretrained(..., subfolder="unet")`，调用约定和上面的教学版完全一样：

```python
predicted_noise = unet(
    sample=noisy_latents,                    # [B, 4, 64, 64]
    timestep=t,                              # [B] 或标量
    encoder_hidden_states=text_embeddings,   # [B, 77, 768]
).sample                                     # [B, 4, 64, 64]
```

## 6. 精确的 SD 1.5 U-Net：逐模块实现

第 5 节的教学版只保留了信息流。这一节把每个模块换成 **SD 1.5 里真实的结构**：ResBlock 多了 dropout，attention 位置换成完整的 Spatial Transformer（self-attention + cross-attention + 前馈网络），每个 DownBlock 有 2 层、每个 UpBlock 有 3 层，一共 12 条 skip。拼起来的网络参数量是 **859,520,964**，和官方 `UNet2DConditionModel` 的 SD 1.5 权重逐个数字相同（6.8 节末尾验证）。 [[2]](#ref-2)[[3]](#ref-3)

模块之间的包含关系：

```text
SD15UNet
├── 时间编码：正弦 [B,320] → MLP → temb [B,1280]（5.2 节）
├── conv_in：[B,4,64,64] → [B,320,64,64]
├── 4 × SD15DownBlock ─┬─ ResnetBlock2D（6.1）
├── SD15MidBlock       ├─ SpatialTransformer（6.4）
├── 4 × SD15UpBlock    │    └─ BasicTransformerBlock（6.3）
│                      │         ├─ Attention：self（6.2）
│                      │         ├─ Attention：cross（6.2）
│                      │         └─ FeedForward（GEGLU，6.3）
│                      └─ Downsample / Upsample（5.4）
└── GroupNorm → SiLU → conv_out：[B,320,64,64] → [B,4,64,64]
```

下面的代码沿用第 5 节的 import、`sinusoidal_timestep_embedding`、`Downsample`、`Upsample`；类名加了 `SD15` 前缀，和第 5 节的教学版区分开。

### 6.1 ResnetBlock2D

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/sd15_resnet_block.png"><img src="/deep-learning/assets/stable-diffusion/sd15_resnet_block.png" alt="ResnetBlock2D 流程图"></a>
<figcaption>ResnetBlock2D。上方虚线框表示 time embedding 的来历（整个网络只算一次），橙色框是这个 block 的输入。</figcaption>
</figure>

和 5.3 节的 TimeResBlock 只差一个 dropout，公式是：

$$
h = \operatorname{Conv}_1\big(\operatorname{SiLU}(\operatorname{GN}_1(x))\big) + W_t\,\operatorname{SiLU}(\text{temb})
$$

$$
y = \operatorname{Conv}_2\big(\operatorname{Dropout}(\operatorname{SiLU}(\operatorname{GN}_2(h)))\big) + \operatorname{shortcut}(x)
$$

```python
class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim=1280, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_dim, out_channels)       # W_t：1280 → 当前通道数
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-5)
        self.dropout = nn.Dropout(dropout)                           # SD 1.5 训练时为 0
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (nn.Identity() if in_channels == out_channels
                              else nn.Conv2d(in_channels, out_channels, 1))   # 通道变了才用 1×1 卷积

    def forward(self, x, temb):                      # x: [B, C_in, H, W]，temb: [B, 1280]
        h = self.conv1(F.silu(self.norm1(x)))                          # [B, C_out, H, W]
        h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]     # + W_t·SiLU(temb)，广播到 H×W
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))           # [B, C_out, H, W]
        return h + self.conv_shortcut(x)                               # 残差
```

> **易错点**
>
> 1. **`temb` 的 1280 维和特征通道数没有关系。** 所有 ResBlock 共用同一个 `temb [B, 1280]`，每个 ResBlock 用自己的 `time_emb_proj` 投影到自己的 320 / 640 / 1280 个通道。
> 2. **时间偏置加在第一个卷积之后、第二个 GroupNorm 之前。** 加在输入上或者最后的残差上，网络也能训练，但已经不是 SD 的结构，权重加载不进来。
> 3. **skip 拼接不是 ResBlock 的事。** UpBlock 里"拼接后通道数变多"由 UpBlock 负责，ResBlock 只看到一个 `C_in` 更大的输入，靠 `conv_shortcut` 对齐残差。

### 6.2 多头注意力：self 和 cross 是同一个类

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/sd15_attention.png"><img src="/deep-learning/assets/stable-diffusion/sd15_attention.png" alt="多头注意力流程图"></a>
<figcaption>多头注意力（图中名为 CrossAttnBlock）。x 是 N 个图像 token，context 是 M 个文本 token；不传 context 时 K、V 也来自 x，就是 self-attention。</figcaption>
</figure>

5.5 节把 GroupNorm、展平和残差都写在了 attention 里面。真实结构把它拆开：这里的 `Attention` 只处理 token 序列 `[B, N, C]`，**self-attention 和 cross-attention 用同一个类**，区别只是 K、V 从哪来：

$$
Q = x W_Q\ [B, N, C],\qquad K = s W_K,\ V = s W_V\ [B, M, C],\qquad
s = \begin{cases} x & \text{self-attention}\ (M = N) \\ c & \text{cross-attention}\ (M = 77) \end{cases}
$$

```python
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, num_heads=8):
        super().__init__()
        context_dim = context_dim or query_dim               # 不给就是 self-attention
        self.num_heads, self.head_dim = num_heads, query_dim // num_heads
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def split_heads(self, t):                            # [B, N, C] → [B, heads, N, head_dim]
        B, N, _ = t.shape
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, context=None):                  # x: [B, N, C]，context: [B, M, C_ctx] 或 None
        source = x if context is None else context
        q = self.split_heads(self.to_q(x))                     # Q，[B, h, N, d]
        k = self.split_heads(self.to_k(source))                # K，[B, h, M, d]
        v = self.split_heads(self.to_v(source))                # V，[B, h, M, d]
        weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(dim=-1)   # [B, h, N, M]
        out = (weights @ v).transpose(1, 2).reshape(x.shape)   # 合并多头 → [B, N, C]
        return self.to_out(out)
```

> **易错点**
>
> 1. **SD 1.5 的头数固定是 8，`head_dim` 随通道变。** 320 / 640 / 1280 通道对应 `head_dim` 40 / 80 / 160。配置里叫 `attention_head_dim=8`，实际是头数（5.8 节）；SD 2.x 改成了固定 `head_dim=64`，头数随通道变。
> 2. **Q、K、V 没有 bias，输出投影有。** 加载权重时 key 名对不上，多半是这里。
> 3. **最高分辨率的 self-attention 最贵。** $$64 \times 64$$ 特征图上 $$N = 4096$$，注意力矩阵每个头 $$4096^2$$；cross-attention 只有 $$4096 \times 77$$。xFormers、`scaled_dot_product_attention` 这类加速主要省的就是这里。

### 6.3 前馈网络（GEGLU）和 BasicTransformerBlock

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/sd15_feedforward.png"><img src="/deep-learning/assets/stable-diffusion/sd15_feedforward.png" alt="FeedForward 流程图"></a>
<figcaption>FeedForward：一个线性层同时输出 value 和 gate，value 逐元素乘 GELU(gate)，再投影回 C。</figcaption>
</figure>

前馈网络用的是 **GEGLU**（GELU-gated linear unit）。先回顾两个激活函数：

- **GELU**（Gaussian Error Linear Unit）：$$\operatorname{GELU}(x) = x\,\Phi(x)$$，$$\Phi$$ 是标准正态分布的 CDF。和 ReLU 的硬截断不同，它平滑地压低较小或负的输入，保留较大的正输入。
- **GLU**（Gated Linear Unit）：一个线性层输出两半 $$[v, g]$$，结果是 $$v \odot \sigma(g)$$。$$\sigma(g) \in (0, 1)$$ 像一个软开关，决定 value 的每一维通过多少。

GEGLU 把 sigmoid 换成 GELU：

$$
[v, g] = W x + b,\qquad \operatorname{GEGLU}(x) = v \odot \operatorname{GELU}(g),\qquad
\operatorname{FF}(x) = W_2\,\operatorname{GEGLU}(x)
$$

value 是要传下去的内容，gate 根据当前输入决定每一维关掉、保留还是放大。例如 value $$= [2, -1, 3]$$、$$\operatorname{GELU}(g) \approx [0, 0, 1.95]$$，输出约为 $$[0, 0, 5.85]$$：前两维被关掉，第三维被放大。和 sigmoid 门不同，GELU 门不限于 $$(0, 1)$$，可以放大，也可以取很小的负值（最小约 $$-0.17$$）。

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/sd15_transformer_block.png"><img src="/deep-learning/assets/stable-diffusion/sd15_transformer_block.png" alt="BasicTransformerBlock 流程图"></a>
<figcaption>BasicTransformerBlock：第一个 attention 是 self-attention（图中 context 留空），第二个是 cross-attention，然后是前馈网络；三处都是 LayerNorm 在前、残差在后（pre-norm）。</figcaption>
</figure>

$$
x \leftarrow x + \operatorname{SelfAttn}(\operatorname{LN}_1(x)),\qquad
x \leftarrow x + \operatorname{CrossAttn}(\operatorname{LN}_2(x),\ c),\qquad
x \leftarrow x + \operatorname{FF}(\operatorname{LN}_3(x))
$$

```python
class GEGLU(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2)            # 一次算出 [v, g]

    def forward(self, x):                                   # [B, N, C] → [B, N, inner]
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.gelu(gate)                         # v ⊙ GELU(g)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(GEGLU(dim, dim * mult), nn.Dropout(0.0), nn.Linear(dim * mult, dim))

    def forward(self, x):                                   # [B, N, C] → [B, N, C]
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim=768, num_heads=8):
        super().__init__()
        self.norm1, self.attn1 = nn.LayerNorm(dim), Attention(dim, None, num_heads)          # self
        self.norm2, self.attn2 = nn.LayerNorm(dim), Attention(dim, context_dim, num_heads)   # cross
        self.norm3, self.ff = nn.LayerNorm(dim), FeedForward(dim)

    def forward(self, x, context):                          # x: [B, HW, C]，context: [B, 77, 768]
        x = x + self.attn1(self.norm1(x))                   # 图像 token 之间互相看
        x = x + self.attn2(self.norm2(x), context)          # 图像 token 去读文本
        x = x + self.ff(self.norm3(x))                      # 逐 token 的非线性变换
        return x
```

> **易错点**
>
> 1. **GEGLU 的线性层输出是 `2 × inner_dim`。** `inner_dim = 4C`，所以 `proj` 是 $$C \to 8C$$，再 `chunk` 成两半；第二个线性层是 $$4C \to C$$。按普通 MLP 写成 $$C \to 4C$$，参数量和权重 shape 都对不上。
> 2. **GELU 不是"大于等于"（greater than or equal to）的缩写**，是 Gaussian Error Linear Unit。
> 3. **前馈网络在每个 token 上独立计算**，token 之间的信息交换只发生在两个 attention 里。

### 6.4 SpatialTransformer：在特征图和 token 序列之间转换

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/sd15_spatial_transformer.png"><img src="/deep-learning/assets/stable-diffusion/sd15_spatial_transformer.png" alt="SpatialTransformer 流程图"></a>
<figcaption>SpatialTransformer：GroupNorm 和 1×1 投影后展平成 H·W 个 token，过一个 BasicTransformerBlock，再变回特征图、投影、加残差。图中的 Linear 在 SD 1.5 里实现为 1×1 卷积，两者等价。</figcaption>
</figure>

```python
class SpatialTransformer(nn.Module):
    def __init__(self, channels, context_dim=768, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, 1)      # 1×1 卷积 = 逐位置的线性层
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(channels, context_dim, num_heads)])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):                          # x: [B, C, H, W]，context: [B, 77, 768]
        B, C, H, W = x.shape
        h = self.proj_in(self.norm(x))                       # [B, C, H, W]
        h = h.flatten(2).transpose(1, 2)                     # [B, HW, C]：每个位置一个 token
        for block in self.transformer_blocks:
            h = block(h, context)                            # [B, HW, C]
        h = h.transpose(1, 2).reshape(B, C, H, W)            # 先换回 [B, C, HW] 再 reshape
        return x + self.proj_out(h)                          # 残差，[B, C, H, W]
```

> **易错点**
>
> 1. **SD 1.5 里每个 SpatialTransformer 只有 1 个 BasicTransformerBlock**（`transformer_layers_per_block=1`）。SDXL 在低分辨率上堆到 10 个，这是它参数量大的主要原因之一。
> 2. **`proj_in`/`proj_out` 在 SD 1.5 是 1×1 卷积，在 SD 2.x 是 `nn.Linear`**（`use_linear_projection=True`）。数学上等价，但权重 shape 不同（`[C, C, 1, 1]` 和 `[C, C]`），互相加载时要 reshape。
> 3. **这里的 GroupNorm 的 eps 是 `1e-6`，ResBlock 里是 `1e-5`。** 照抄权重时细节要一致，否则输出会有小偏差。

### 6.5 SD15DownBlock：2 层，存 2～3 条 skip

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/sd15_down_block.png"><img src="/deep-learning/assets/stable-diffusion/sd15_down_block.png" alt="SD 1.5 DownBlock 流程图"></a>
<figcaption>SD 1.5 的 DownBlock：(ResnetBlock2D → SpatialTransformer) 重复 2 次，每次的输出都存为 skip；再下采样，下采样的输出也存。最后一个 block 没有下采样，也就没有第 3 条 skip。图中通道写的是 C → 2C，实际只有 DownBlock 2、3 翻倍（320→640→1280），DownBlock 1、4 不变；DownBlock 4 没有 SpatialTransformer。</figcaption>
</figure>

```python
class SD15DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, has_attention=True, add_downsample=True,
                 num_layers=2):
        super().__init__()
        self.resnets = nn.ModuleList([                                # 只有第 1 个 ResBlock 改通道
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_layers)
        ])
        self.attentions = nn.ModuleList([
            SpatialTransformer(out_channels) for _ in range(num_layers)
        ]) if has_attention else None
        self.downsample = Downsample(out_channels, out_channels) if add_downsample else None

    def forward(self, x, temb, context):              # x: [B, C_in, H, W]
        skips = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)                                       # [B, C_out, H, W]
            if self.attentions is not None:
                x = self.attentions[i](x, context)
            skips.append(x)                                           # 每层的输出都存
        if self.downsample is not None:
            x = self.downsample(x)                                    # [B, C_out, H/2, W/2]
            skips.append(x)                                           # 下采样的输出也存
        return x, skips
```

加上 `conv_in` 的输出，编码器一共往栈里压 **1 + 3 + 3 + 3 + 2 = 12** 条 skip（$$512 \times 512$$ 输入）：

| 顺序 | 来源 | shape |
|---|---|---|
| 1 | `conv_in` | `[B, 320, 64, 64]` |
| 2、3 | DownBlock 1 的第 1、2 层 | `[B, 320, 64, 64]` |
| 4 | DownBlock 1 的下采样 | `[B, 320, 32, 32]` |
| 5、6 | DownBlock 2 的第 1、2 层 | `[B, 640, 32, 32]` |
| 7 | DownBlock 2 的下采样 | `[B, 640, 16, 16]` |
| 8、9 | DownBlock 3 的第 1、2 层 | `[B, 1280, 16, 16]` |
| 10 | DownBlock 3 的下采样 | `[B, 1280, 8, 8]` |
| 11、12 | DownBlock 4 的第 1、2 层（无 attention） | `[B, 1280, 8, 8]` |

换个角度看：**每个分辨率上都有 3 条 skip**，1 条是"进入这个分辨率时的特征"（`conv_in` 或上一个 block 的下采样输出），2 条是这个分辨率上两层的输出。这正好让每个 UpBlock 能在同一个分辨率上取 3 条。

### 6.6 SD15UpBlock：3 层，每层取 1 条 skip

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/sd15_up_block.png"><img src="/deep-learning/assets/stable-diffusion/sd15_up_block.png" alt="SD 1.5 UpBlock 流程图"></a>
<figcaption>SD 1.5 的 UpBlock：(拼接一条 skip → ResnetBlock2D → SpatialTransformer) 重复 3 次，最后上采样（最后一个 block 不上采样）。图中通道写的是 C → C/2，实际最深的两个 UpBlock 保持 1280，后两个是 1280→640、640→320；最深的 UpBlock 没有 SpatialTransformer。</figcaption>
</figure>

UpBlock 比 DownBlock 多一层：每层先 `pop()` 一条 skip、沿通道拼接，再过 ResBlock 和 SpatialTransformer。4 个 UpBlock × 3 层正好用完 12 条 skip。

```python
class SD15UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, has_attention=True, add_upsample=True):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i, c_skip in enumerate(skip_channels):                    # skip_channels：3 次 pop 的通道数
            c_x = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(c_x + c_skip, out_channels))   # 输入是拼接后的通道数
        self.attentions = nn.ModuleList([
            SpatialTransformer(out_channels) for _ in skip_channels
        ]) if has_attention else None
        self.upsample = Upsample(out_channels, out_channels) if add_upsample else None

    def forward(self, x, skips, temb, context):       # skips：编码器的栈，这里会被 pop 3 次
        for i, resnet in enumerate(self.resnets):
            skip = skips.pop()                                        # 后进先出
            x = torch.cat([x, skip], dim=1)                           # H、W 必须相同，通道相加
            x = resnet(x, temb)                                       # [B, C_out, H, W]
            if self.attentions is not None:
                x = self.attentions[i](x, context)
        if self.upsample is not None:
            x = self.upsample(x)                                      # 整个 block 最后才上采样
        return x
```

12 次 `pop()` 的完整过程（"拼接后"是送进 ResBlock 的通道数）：

| UpBlock（分辨率） | 依次取出的 skip | 拼接后 | ResBlock 后 |
|---|---|---|---|
| 1（8×8） | 12、11、10：`1280, 1280, 1280` | 2560, 2560, 2560 | 1280 |
| 2（16×16） | 9、8、7：`1280, 1280, 640` | 2560, 2560, 1920 | 1280 |
| 3（32×32） | 6、5、4：`640, 640, 320` | 1920, 1280, 960 | 640 |
| 4（64×64） | 3、2、1：`320, 320, 320` | 960, 640, 640 | 320 |

例如 UpBlock 3 的第 1 层：输入来自 UpBlock 2（1280 通道，已上采样到 32×32），拼上 640 通道的 skip 6，是 1920；ResBlock 把它压到 640。之后两层的输入都是 640。

### 6.7 SD15MidBlock

最低分辨率上的 ResBlock → SpatialTransformer → ResBlock，不改 shape：

```python
class SD15MidBlock(nn.Module):
    def __init__(self, channels=1280):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(channels, channels), ResnetBlock2D(channels, channels)])
        self.attentions = nn.ModuleList([SpatialTransformer(channels)])

    def forward(self, x, temb, context):             # [B, 1280, 8, 8] → [B, 1280, 8, 8]
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        return self.resnets[1](x, temb)
```

> **易错点**
>
> 1. **skip 用栈，后进先出。** 最先压进去的 `conv_in` 输出最后才被用到，给最后一个 UpBlock 的最后一层。
> 2. **每次拼接只要求 H、W 相同，通道可以不同**（例如 1280 + 640）。分辨率对不上通常是 latent 尺寸不能被 8 整除。
> 3. **DownBlock 每层后存 skip，UpBlock 每层前取 skip。** 只在 block 的边界存取（第 5 节教学版的做法）也能训练，但 skip 数量和权重 shape 都和 SD 不同。
> 4. **最深的一对 block 没有 attention**（DownBlock 4、UpBlock 1，Diffusers 里叫 `DownBlock2D`、`UpBlock2D`），带 attention 的叫 `CrossAttnDownBlock2D`、`CrossAttnUpBlock2D`。MidBlock 有 attention。

### 6.8 拼成完整的 SD15UNet

<figure class="sd-img">
<a href="/deep-learning/assets/stable-diffusion/sd15_unet.png"><img src="/deep-learning/assets/stable-diffusion/sd15_unet.png" alt="SD 1.5 完整 U-Net 流程图"></a>
<figcaption>完整的 SD 1.5 U-Net：红色虚线是 time embedding（图中 t_dim = 320，4·t_dim = 1280），灰色虚线是文本条件，横向箭头标出了 12 条 skip 的分组。图中 UpBlock 按分辨率编号（UpBlock4 最深、最先执行），和正文按执行顺序的编号相反。</figcaption>
</figure>

| block | 通道（输入 → 输出） | 分辨率 | attention | 结尾 |
|---|---|---|---|---|
| DownBlock 1 | 320 → 320 | 64×64 | 有 | 下采样 |
| DownBlock 2 | 320 → 640 | 32×32 | 有 | 下采样 |
| DownBlock 3 | 640 → 1280 | 16×16 | 有 | 下采样 |
| DownBlock 4 | 1280 → 1280 | 8×8 | 无 | — |
| MidBlock | 1280 → 1280 | 8×8 | 有 | — |
| UpBlock 1 | 1280 → 1280 | 8×8 | 无 | 上采样 |
| UpBlock 2 | 1280 → 1280 | 16×16 | 有 | 上采样 |
| UpBlock 3 | 1280 → 640 | 32×32 | 有 | 上采样 |
| UpBlock 4 | 640 → 320 | 64×64 | 有 | — |

```python
class SD15UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(                        # 正弦编码 320 → temb 1280
            nn.Linear(320, 1280), nn.SiLU(), nn.Linear(1280, 1280))
        self.conv_in = nn.Conv2d(4, 320, 3, padding=1)
        self.down_blocks = nn.ModuleList([
            SD15DownBlock(320, 320),
            SD15DownBlock(320, 640),
            SD15DownBlock(640, 1280),
            SD15DownBlock(1280, 1280, has_attention=False, add_downsample=False),
        ])
        self.mid_block = SD15MidBlock(1280)
        self.up_blocks = nn.ModuleList([              # skip_channels 按 pop 的顺序写
            SD15UpBlock(1280, 1280, [1280, 1280, 1280], has_attention=False),
            SD15UpBlock(1280, 1280, [1280, 1280, 640]),
            SD15UpBlock(1280, 640, [640, 640, 320]),
            SD15UpBlock(640, 320, [320, 320, 320], add_upsample=False),
        ])
        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-5)
        self.conv_out = nn.Conv2d(320, 4, 3, padding=1)

    def forward(self, sample, timesteps, encoder_hidden_states):
        # sample: [B, 4, h, w]，timesteps: [B] 或标量，encoder_hidden_states: [B, 77, 768]
        timesteps = torch.as_tensor(timesteps, device=sample.device).reshape(-1).expand(sample.shape[0])
        temb = self.time_mlp(sinusoidal_timestep_embedding(timesteps, 320).to(sample.dtype))   # [B, 1280]

        x = self.conv_in(sample)                                       # [B, 320, h, w]
        skips = [x]                                                    # 第 1 条 skip
        for block in self.down_blocks:
            x, new_skips = block(x, temb, encoder_hidden_states)
            skips.extend(new_skips)                                    # 3 + 3 + 3 + 2
        assert len(skips) == 12

        x = self.mid_block(x, temb, encoder_hidden_states)             # [B, 1280, h/8, w/8]
        for block in self.up_blocks:
            x = block(x, skips, temb, encoder_hidden_states)           # 每个 block pop 3 条
        assert not skips                                               # 12 条正好用完
        return self.conv_out(F.silu(self.conv_norm_out(x)))           # [B, 4, h, w]
```

在 $$64 \times 64$$ 的 latent 上完整跑一遍很吃内存（最高分辨率的 self-attention），检查结构时用小一点的 latent 就够了，拓扑和通道数不变：

```python
sd15 = SD15UNet().eval()
print(sum(p.numel() for p in sd15.parameters()))            # 859520964，和官方 SD 1.5 U-Net 相同
with torch.no_grad():
    out = sd15(torch.randn(1, 4, 16, 16), torch.tensor([500]), torch.randn(1, 77, 768))
print(out.shape)                                            # torch.Size([1, 4, 16, 16])
```

参数量能逐位对上，说明每个模块的层数、通道、bias 都和官方一致；只是参数的命名（state dict 的 key）没有完全照抄，不能直接加载官方权重。

> **易错点**
>
> 1. **参数主要在低分辨率。** 1280 通道的 block（DownBlock 3、4，MidBlock，UpBlock 1、2）占了大部分参数；计算量则相反，集中在 64×64 的高分辨率 attention 上。
> 2. **时间步要扩展到 batch 大小。** 推理时 scheduler 给的是一个标量 `t`，而 CFG 下 batch 是 $$2B$$，这里用 `reshape(-1).expand(B)` 统一成 `[B]`。
> 3. **正弦编码的细节要和权重一致。** SD 1.5 是 cos 在前、频率分母用 $$D/2$$（Diffusers 里 `flip_sin_to_cos=True`、`freq_shift=0`），5.2 节的函数正好如此。

## 7. 训练一步

训练只更新 U-Net，目标和 DDPM 完全相同，只是换成了 latent 和带文本条件的网络：

$$
\mathcal{L} = \mathbb{E}_{x,\,y,\,t,\,\epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t,\ c(y)\big)\big\rVert^2\Big],\qquad z_0 = 0.18215 \cdot E(x)
$$

其中 $$y$$ 是图像配的文字描述。为了推理时能用 CFG，训练时以 10% 的概率把描述换成空字符串，让同一个 U-Net 同时学会有条件和无条件的预测（DDPM 篇第 9 节）。

```python
def stable_diffusion_training_step(images, prompts, tokenizer, text_encoder, vae, unet,
                                   noise_scheduler, optimizer, device, condition_dropout_prob=0.1):
    # 1. 冻结的 VAE：[B, 3, 512, 512] → z0 [B, 4, 64, 64]（已乘 scaling factor）
    latents = encode_images(images, vae, sample_posterior=True)

    # 2. 每张图随机一个 t 和一份噪声，一步加噪：z_t = √ᾱ_t z0 + √(1−ᾱ_t) ε
    noise = torch.randn_like(latents)                                              # ε，[B, 4, 64, 64]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (latents.shape[0],), device=latents.device)          # [B]
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)           # z_t

    # 3. 10% 的描述换成空串（CFG 的无条件分支），再用冻结的 CLIP 编码
    drop = torch.rand(len(prompts)) < condition_dropout_prob
    prompts = ["" if d else p for d, p in zip(drop.tolist(), prompts)]
    text_embeddings = encode_prompts(prompts, tokenizer, text_encoder, device)      # c，[B, 77, 768]

    # 4. 预测噪声，MSE，只更新 U-Net
    pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample   # ε̂
    loss = F.mse_loss(pred.float(), noise.float())                                 # ‖ε − ε̂‖²
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)               # 梯度范数超过 1 就整体缩放
    optimizer.step()
    return loss.item()
```

| 组件 | 训练时的状态 |
|---|---|
| VAE 编码器 / 解码器 | 预训练，冻结（解码器不参与） |
| Tokenizer | 无参数 |
| CLIP 文本编码器 | 预训练，冻结 |
| U-Net | **训练** |
| Scheduler | 无参数 |

> **易错点**
>
> 1. **冻结的模块要 `eval()` 并 `requires_grad_(False)`，优化器只放 U-Net 的参数。** 否则显存会被 CLIP 和 VAE 的梯度占满，而且它们会被悄悄改掉。
> 2. **丢掉的描述换成空字符串 `""`，不是全零向量。** 推理时无条件分支用的也是 `encode_prompts([""])`，两边必须一致。空串编码后仍然是 `<BOS><EOS><PAD>...`，向量并不为零。
> 3. **loss 用 float32 算。** 混合精度训练时 U-Net 输出是 fp16，直接算 MSE 精度不够，先 `.float()`。
> 4. **训练只跑一次 U-Net，不跑反向链，也不跑 VAE 解码器。** 大规模训练还会事先把所有图像的 latent 算好存下来，训练时连 VAE 编码器都不用跑。
> 5. **`clip_grad_norm_` 不是必需的，但很便宜。** 偶尔一个 batch 的梯度特别大时，它把整体范数缩放到 `max_norm`，避免 loss 突然尖峰或者 NaN。`zero_grad(set_to_none=True)` 把梯度设成 `None` 而不是 0，省内存，也让没收到梯度的参数被优化器跳过。
> 6. **把"准备数据"和"更新一步"分开更好测。** 这里为了紧凑写在一个函数里；实际代码常拆成三层：`prepare_batch`（冻结的 VAE、CLIP 和描述置空，在 `no_grad` 下）、`diffusion_loss`（加噪、前向、MSE，没有副作用）、`train_step`（反向和优化器）。这样 loss 可以单独单元测试，梯度累积、混合精度也只改最外层。
> 7. **完整的训练还需要** 混合精度、梯度累积、EMA、多卡、按分辨率分桶等，这里只保留决定学习目标的核心路径。

## 8. 推理全流程与 CFG

### 8.1 CFG 的两个辅助函数

推理和 DDPM 篇第 9 节的 CFG 完全一样，只是"无条件"换成了**负向 prompt**（默认是空串） [[6]](#ref-6)：

$$
\hat\epsilon_{\text{cfg}} = \epsilon_\theta(z_t, t, c_{\text{neg}}) + w\,\big(\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, c_{\text{neg}})\big),\qquad w = 7.5
$$

实现时把两份条件拼成一个 `2B` 的 batch，一次前向算完：

```python
@torch.no_grad()
def encode_cfg_prompts(prompts, negative_prompts, tokenizer, text_encoder, device):
    uncond = encode_prompts(negative_prompts, tokenizer, text_encoder, device)   # c_neg，[B, 77, 768]
    cond = encode_prompts(prompts, tokenizer, text_encoder, device)              # c，[B, 77, 768]
    return torch.cat([uncond, cond], dim=0)                                      # 无条件在前，[2B, 77, 768]


@torch.no_grad()
def predict_noise_with_cfg(latents, t, cfg_embeddings, unet, scheduler, guidance_scale):
    model_input = torch.cat([latents, latents], dim=0)                 # 同一份 latent 复制两份，[2B, 4, h, w]
    model_input = scheduler.scale_model_input(model_input, t)
    noise = unet(model_input, t, encoder_hidden_states=cfg_embeddings).sample   # [2B, 4, h, w]
    eps_uncond, eps_cond = noise.chunk(2)                              # 顺序和 cfg_embeddings 的拼接顺序一致
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)       # ε_cfg，[B, 4, h, w]
```

### 8.2 完整的生成函数

```python
@torch.no_grad()
def generate_images(prompts, tokenizer, text_encoder, vae, unet, scheduler, device,
                    negative_prompts=None, height=512, width=512,
                    num_inference_steps=50, guidance_scale=7.5, seed=0):
    B = len(prompts)
    negative_prompts = negative_prompts or [""] * B

    # 1. 文本只编码一次：[2B, 77, 768]
    cfg_embeddings = encode_cfg_prompts(prompts, negative_prompts, tokenizer, text_encoder, device)

    # 2. 从 latent 空间的随机噪声开始：[B, 4, height/8, width/8]
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn(B, unet.config.in_channels, height // 8, width // 8,
                          generator=generator, device=device, dtype=cfg_embeddings.dtype)
    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    # 3. 循环：U-Net（2B）→ CFG 合并 → scheduler 走一步
    for t in scheduler.timesteps:
        eps = predict_noise_with_cfg(latents, t, cfg_embeddings, unet, scheduler, guidance_scale)
        latents = scheduler.step(eps, t, latents).prev_sample          # [B, 4, h, w]

    # 4. VAE 只解码一次：[B, 3, height, width]，范围 [0, 1]
    return decode_latents(latents, vae)
```

一次生成里各模块的调用次数（$$N$$ 为采样步数）：

| 模块 | 次数 |
|---|---|
| Tokenizer + CLIP | 1 次（正向、负向各一份） |
| U-Net | $$N$$ 次，每次 batch 是 $$2B$$ |
| Scheduler step | $$N$$ 次 |
| VAE 编码器 | 0 次 |
| VAE 解码器 | 1 次 |

绝大部分时间花在 U-Net 上，这就是为什么少步采样器（DDIM、DPM-Solver）和蒸馏方法能直接加速生成。

> **易错点**
>
> 1. **拼接顺序和 `chunk` 顺序必须一致。** 这里约定无条件在前、有条件在后；写反了相当于 $$w$$ 变成了负数方向，生成的东西会和 prompt 正好"相反"。
> 2. **负向 prompt 就是替换掉空串的无条件分支。** CFG 把结果推离 $$c_{\text{neg}}$$、推向 $$c$$，所以在负向 prompt 里写 "blurry, low quality" 能起作用。它没有额外的模型或训练。
> 3. **$$w$$ 太大会过饱和。** 7～8 是 SD 1.x 常用的值；15 以上颜色和对比度会明显失真。$$w = 1$$ 等于不用 CFG，可以省掉一半计算。
> 4. **dtype 要一致。** fp16 推理时 latent、文本条件、U-Net、VAE 必须是同一种精度；`torch.randn` 默认 float32，要指定 dtype。
> 5. **随机种子决定初始 latent。** 同一个种子、同一个 prompt、确定性的 scheduler（如 DDIM，$$\eta=0$$）会得到同一张图；换 scheduler 或步数，即使种子相同，图也会变化。

### 8.3 用 Diffusers 直接跑

想要的只是出图时，`StableDiffusionPipeline` 把上面 8.2 节的所有步骤都包好了：

```python
import torch
from diffusers import StableDiffusionPipeline

if torch.cuda.is_available():
    device, dtype = "cuda", torch.float16          # GPU 上用 fp16，显存减半
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float32           # Apple 芯片
else:
    device, dtype = "cpu", torch.float32           # 能跑，但很慢

# 模型 ID 是一个完整仓库：tokenizer、CLIP、VAE、U-Net、scheduler 的权重和配置都在里面
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=dtype, use_safetensors=True,
).to(device)

image = pipe(
    prompt="a photo of a corgi wearing sunglasses, high detail",
    negative_prompt="blurry, low quality, distorted",   # 推离这些描述，见 8.1 节
    height=512, width=512,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator("cpu").manual_seed(42),   # 在 CPU 上生成初始噪声，换设备也能复现
).images[0]                                             # PIL 图像
image.save("corgi.png")
```

环境：`pip install -U torch diffusers transformers accelerate safetensors pillow`。

拆开 pipeline 手动调用各组件时（就是 8.2 节的 `generate_images`），有 5 条约定必须守住：

1. `scheduler.set_timesteps(N)` 决定实际走哪些时间步；初始 latent 乘 `scheduler.init_noise_sigma`。
2. 每次调用 U-Net 前先 `scheduler.scale_model_input(latents, t)`。
3. 同一个 `t` 同时传给 U-Net 和 `scheduler.step`。
4. scheduler 的 `prediction_type` 和 U-Net 训练时的一致（SD 1.5 是 `epsilon`）。
5. 解码前除以 `vae.config.scaling_factor`。

常用参数：

| 参数 | 常用值 | 作用 |
|---|---|---|
| `num_inference_steps` | 20～50 | scheduler 走几步；更多更慢，不一定更好 |
| `guidance_scale` | 7～8 | CFG 强度 $$w$$ |
| `negative_prompt` | `""` 或 "blurry, low quality" | 替换无条件分支 |
| `height`、`width` | 512 | SD 1.5 的训练分辨率；必须是 8 的倍数，偏离 512 太多容易出现重复的主体 |
| `generator` | 固定种子 | 决定初始 latent，用于复现和对比 |

> **易错点**
>
> 1. **模型 ID 指的是整个仓库**，不是某个 `.ckpt` / `.safetensors` 文件。单文件权重要用 `from_single_file` 加载。
> 2. **pipeline 默认带 safety checker**，手动拼组件时没有它。
> 3. **访问令牌用 `huggingface-cli login` 管理**，不要写进脚本里。

## 9. 训练和推理的对照

| | 训练 | 文生图推理 |
|---|---|---|
| 起点 | 真实图像 $$x$$ | 随机 latent $$z_T$$ |
| VAE 编码器 | 用（或缓存 latent），编码后乘 0.18215 | 不用 |
| 加噪 | 一次 `add_noise`，随机 $$t$$ | 不用 |
| U-Net 调用 | 每张图一次 | 每步一次（CFG 时 batch 翻倍） |
| 文本条件 | 10% 换成空串 | 正向 + 负向两份 |
| 目标 | 已知的噪声 $$\epsilon$$ | 未知，靠 scheduler 迭代 |
| 损失和梯度 | MSE，反向传播只到 U-Net | 无 |
| VAE 解码器 | 不用 | 最后除以 0.18215 后用一次 |

最关键的一点：**训练时随机抽一个 $$t$$、一步造出 $$z_t$$，不需要走完整条链；推理时不知道干净的 latent，所以必须一步步反向走。** 这和 DDPM 完全一样，Stable Diffusion 只是把它搬到了 latent 空间，并加上了文本条件。

## 10. 自检

不需要下载任何权重，用随机的 latent 和"假"的文本条件检查第 5 节的 U-Net 和第 8 节的 CFG 逻辑（第 6 节的精确 U-Net 在 6.8 节末尾单独检查过）：

```python
torch.manual_seed(0)
B, L, D_text = 2, 77, 768

# 1. 时间编码：[B] → [B, D]，取值在 [-1, 1]
emb = sinusoidal_timestep_embedding(torch.tensor([0, 999]), 320)
assert emb.shape == (2, 320) and emb.abs().max() <= 1

# 2. cross-attention：shape 不变，注意力权重在 77 个词上和为 1
attn = SpatialCrossAttention(channels=64, context_dim=D_text, num_heads=8)
x, context = torch.randn(B, 64, 16, 16), torch.randn(B, L, D_text)
assert attn(x, context).shape == x.shape
q = attn.to_q(attn.norm(x).flatten(2).transpose(1, 2)).view(B, 256, 8, 8).transpose(1, 2)
k = attn.to_k(context).view(B, L, 8, 8).transpose(1, 2)
w = (q @ k.transpose(-2, -1) / math.sqrt(8)).softmax(-1)
assert w.shape == (B, 8, 256, L) and torch.allclose(w.sum(-1), torch.ones(B, 8, 256))

# 3. 完整 U-Net：输出和输入同 shape；4 个层级只有 3 次采样
unet = ConditionalUNet(base_channels=32, channel_multipliers=(1, 2, 4, 4))
z_t = torch.randn(B, 4, 32, 32)
eps = unet(z_t, torch.tensor([800, 250]), torch.randn(B, L, D_text))
assert eps.shape == z_t.shape
assert unet.down_blocks[-1].downsample is None and unet.up_blocks[-1].upsample is None

# 4. 尺寸不能被 8 整除时直接报错，而不是在 cat 时才出错
try:
    unet(torch.randn(1, 4, 36, 36), torch.tensor([10]), torch.randn(1, L, D_text))
    raise RuntimeError("should have failed")
except AssertionError:
    pass

# 5. CFG 的拼接顺序：w = 1 时结果等于有条件预测，w = 0 时等于无条件预测
class FakeUNet:                                    # 输出 = latent + 文本条件的均值，便于区分两个分支
    class Out:
        def __init__(self, sample): self.sample = sample
    def __call__(self, sample, t, encoder_hidden_states):
        return self.Out(sample + encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1))

class IdentityScheduler:
    def scale_model_input(self, x, t): return x

latents = torch.randn(B, 4, 8, 8)
c_neg, c = torch.zeros(B, L, D_text), torch.ones(B, L, D_text)
cfg_embeddings = torch.cat([c_neg, c])
for w, expected in [(1.0, latents + 1), (0.0, latents)]:
    out = predict_noise_with_cfg(latents, 0, cfg_embeddings, FakeUNet(), IdentityScheduler(), w)
    assert torch.allclose(out, expected)

# 6. SD 1.x 的 scaled_linear 调度：终点信号系数约 0.068，不是 0
betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
assert abs(torch.cumprod(1 - betas, 0)[-1].sqrt().item() - 0.068) < 1e-3
```

## 11. 要点回顾

- **Stable Diffusion 由哪些部分组成？** Tokenizer、CLIP 文本编码器、VAE、U-Net、scheduler。只有 U-Net 训练；tokenizer 和 scheduler 没有参数，CLIP 和 VAE 预训练后冻结。
- **和 DDPM 有什么不同？** 数学和训练目标相同。不同之处在于在 VAE 的 $$4 \times 64 \times 64$$ latent 里做扩散，用 CLIP 的逐 token 向量通过 cross-attention 做条件，推理时用 CFG。
- **文本怎么进入网络？** Prompt → 77 个 token → CLIP → `[B, 77, 768]`。在 U-Net 的 cross-attention 里，图像特征当 query，文本当 key/value，每个空间位置在 77 个词上做一次加权平均。保留全部 77 个向量，是为了让不同位置能关注不同的词。
- **时间怎么进入网络？** 正弦编码 → MLP → 在每个 ResBlock 里投影成每通道一个偏置，加到特征图上。
- **为什么要 scaling factor？** VAE latent 的标准差约 5.5，乘 0.18215 后接近 1，和扩散过程假设的数据尺度一致。编码后乘、解码前除。
- **U-Net 的结构？** $$N$$ 个 DownBlock（各存一个 skip，前 $$N-1$$ 个下采样）→ MidBlock → $$N$$ 个 UpBlock（倒序拼接 skip，前 $$N-1$$ 个上采样）。SD 1.5 是 4 层，通道 320/640/1280/1280，约 8.6 亿参数。
- **真实 SD 1.5 的 U-Net 比教学版多了什么？** ResBlock 后面的 attention 换成 Spatial Transformer（self-attention → cross-attention → GEGLU 前馈，都是 pre-norm 残差）；DownBlock 2 层、UpBlock 3 层；每层输出、每次下采样和 `conv_in` 都存 skip，共 12 条，UpBlock 每层 pop 一条。参数量 859,520,964。
- **一次训练做什么？** 编码图像 → 随机 $$t$$ 和噪声一步加噪 → 编码文本（10% 置空）→ U-Net 预测噪声 → MSE → 只更新 U-Net。
- **一次推理做什么？** 编码正向和负向 prompt 一次 → 从随机 latent 开始，每步 U-Net（batch 翻倍）+ CFG + scheduler → VAE 解码一次。

## 参考资料

- <a id="ref-1"></a>**[1] 主论文**：Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)。VAE latent、cross-attention 条件和整体结构都来自这篇，详见 [LDM 篇](/deep-learning/latent-diffusion/)。
- <a id="ref-2"></a>**[2] 官方仓库**：CompVis/stable-diffusion [github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)。SD 1.x 的原始代码和配置（`v1-inference.yaml` 里有 U-Net 的全部超参数）。
- <a id="ref-3"></a>**[3] Diffusers 文档**：Hugging Face Diffusers [huggingface.co/docs/diffusers/en/index](https://huggingface.co/docs/diffusers/en/index)。第 6～8 节的 `UNet2DConditionModel`、scheduler 和 pipeline 用法。
- <a id="ref-4"></a>**[4] CLIP**：Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)。第 2 节的文本编码器。
- <a id="ref-5"></a>**[5] 图解博客**：Jay Alammar. *The Illustrated Stable Diffusion*. [jalammar.github.io/illustrated-stable-diffusion/](https://jalammar.github.io/illustrated-stable-diffusion/)。把数据流画得很清楚，适合先看一遍建立整体印象。
- <a id="ref-6"></a>**[6] CFG**：Ho, Salimans. *Classifier-Free Diffusion Guidance*. 2022. [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)。第 8 节推理时的引导。
