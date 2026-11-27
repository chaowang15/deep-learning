---
date: 2026-11-27 08:00:00 -0800
keywords: >-
  IP-Adapter, image prompt, decoupled cross-attention, CLIP image embedding, OpenCLIP ViT-H/14, image projection, IP-Adapter Plus, IP-Adapter FaceID, Stable Diffusion 1.5, SDXL, ControlNet, T2I-Adapter, diffusers, PyTorch.
summary: >-
  IP-Adapter 怎么让 Stable Diffusion 接受"一张图"作为提示：用冻结的 CLIP 图像编码器提取参考图特征，投影成 4 个图像 token，再给 U-Net 的每个 cross-attention 加一对新的 K、V 投影，和文本分支解耦地各算一次注意力后相加。讲清楚为什么必须解耦、只训练 2200 万参数够在哪、训练数据和丢弃策略、推理时的 λ 和 CFG，给出 SD 1.5 上逐层精确并经过验证的实现，以及 Diffusers 的用法、和 ControlNet / T2I-Adapter 的组合、Plus / Face / FaceID 等变体。
layout: note
title: "IP-Adapter：让扩散模型用图片当提示词"
description: "How IP-Adapter lets Stable Diffusion take an image as a prompt: CLIP image features projected to tokens, decoupled cross-attention with new key/value projections, why decoupling matters, training data and dropout, inference scale and CFG, an exact verified SD 1.5 implementation, diffusers usage, combining with ControlNet / T2I-Adapter, and the Plus / Face / FaceID variants."
---

<style>
.ddpm-fig{margin:1.2em 0}
.ddpm-fig svg{width:100%;height:auto;display:block;max-width:760px;margin:0 auto}
.ddpm-fig svg text{font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;font-size:13px;fill:#202124}
.ddpm-fig svg .sm{font-size:11.5px;fill:#667085}
.ddpm-fig svg .lane{font-size:14px;font-weight:600}
.ddpm-fig figcaption{text-align:center;color:#667085;font-size:.9em;margin-top:.4em}
</style>

# IP-Adapter：让扩散模型用图片当提示词

很多东西用文字说不清：一种具体的画风、一个人的长相、一件衣服的花纹。"把这张图的感觉画出来"最直接的办法是**把图本身当成提示**（image prompt）。

[T2I-Adapter](/deep-learning/t2i-adapter/) 和 [ControlNet](/deep-learning/controlnet/) 给的是**空间**条件：条件图和输出像素对齐，管的是"画在哪、什么形状"。IP-Adapter 给的是**内容和风格**条件：参考图和输出不需要对齐，管的是"画成什么样子"。它的做法是：

1. 用一个冻结的 CLIP 图像编码器提取参考图的全局特征，投影成 **4 个图像 token**；
2. 在 U-Net 每个 cross-attention 里，**另加一对 K、V 投影**专门处理这 4 个 token，和原来的文本注意力**分开算、结果相加**；
3. 只训练投影层和新加的 K、V，共约 **2200 万参数**，U-Net 和两个编码器都冻结。

训练完的 IP-Adapter 可以挂在同一底模的各种微调版本上，也可以和文本提示、ControlNet、T2I-Adapter 同时使用。

## 参考资料

本文主要参考 IP-Adapter 原论文 [1] 和官方仓库 [2]，Diffusers 的用法参考 [3]。正文中用 [n] 标注具体出处。

- <a id="ref-1"></a>**[1] 主论文**：Ye et al. *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models*. 2023. [arxiv.org/abs/2308.06721](https://arxiv.org/abs/2308.06721)。解耦 cross-attention、图像投影、训练数据和丢弃比例都来自这篇（第 3、4 节）。
- <a id="ref-2"></a>**[2] 官方仓库**：tencent-ailab/IP-Adapter [github.com/tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)。训练脚本、Plus / FaceID 等变体的实现。
- <a id="ref-3"></a>**[3] Diffusers 文档**：Hugging Face Diffusers: IP-Adapter [huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter)。第 6 节的加载、调强度、缓存嵌入。
- <a id="ref-4"></a>**[4] 权重**：h94/IP-Adapter [huggingface.co/h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)。第 6.4 节列出的各个权重文件。

## 0. 全文路线

| 节 | 内容 |
|---|---|
| 1 | 图像提示要解决什么，和 img2img、DreamBooth / LoRA、unCLIP 有什么不同 |
| 2 | 结构和公式：图像 token、解耦 cross-attention、为什么必须解耦 |
| 3 | SD 1.5 上逐层精确的实现（2232 万参数），并验证 λ = 0 时等于原模型 |
| 4 | 训练：数据、冻结、丢弃策略、训练一步 |
| 5 | 推理：λ、CFG、和文本 / 空间控制组合 |
| 6 | 用 Diffusers：加载、调强度、和 ControlNet 组合、缓存图像嵌入 |
| 7 | 变体（Plus、Face、FaceID）、相关方法、要点回顾 |

## 1. 图像提示要解决什么

### 1.1 几种"参考一张图"的方法

| 方法 | 怎么用参考图 | 要训练吗 | 局限 |
|---|---|---|---|
| img2img | 参考图加噪后当作起点再去噪 | 不用 | 保留的是**布局和颜色**，不是内容；强度一大就和参考图无关 |
| DreamBooth / LoRA | 拿几张参考图微调模型 | **每个新概念都要训练** | 每换一个人物、风格就要训一次 |
| unCLIP（如 DALL·E 2） | 整个模型用 CLIP 图像嵌入作条件训练 | 要从头训练或全量微调 | 代价大，而且换了底模就要重来 |
| **IP-Adapter** | CLIP 图像特征 → 新的 cross-attention 分支 | **只训练一次，之后任何参考图都直接用** | 细节保真度不如逐个微调；空间布局不受控 |

IP-Adapter 的定位是：像 unCLIP 一样"给一张图就能用"，又像 ControlNet 一样只训练一个外挂模块、底模冻结。

### 1.2 为什么用 CLIP 图像特征

CLIP 的图像编码器和文本编码器是对齐训练的，它的图像嵌入天然编码了"这张图是什么、什么风格"这类语义，而不是像素细节。SD 1.5 版本的 IP-Adapter 用的是 OpenCLIP **ViT-H/14**，取的是它最后投影出来的全局图像嵌入（1024 维）。

这也决定了 IP-Adapter 的能力边界：**它看到的是 CLIP 看到的东西**。语义、风格、配色、大致构图能传过去；精确的人脸身份、文字、细小的花纹就容易丢，这正是后面 Plus、FaceID 等变体要补的地方（7.1 节）。

> **易错点**
>
> 1. **IP-Adapter 不是 img2img。** img2img 的参考图进入的是采样的**起点**，IP-Adapter 的参考图进入的是 **cross-attention 的条件**。前者保留布局、丢掉内容，后者保留内容和风格、不保留布局。
> 2. **图像编码器要和训练时一致。** SD 1.5 的 IP-Adapter 配 ViT-H/14；SDXL 的基础版配的是更大的 ViT-bigG/14。拿错图像编码器，嵌入维度可能刚好对得上（不报错），但语义完全错位。

## 2. 结构和公式

### 2.1 整体数据流

<figure class="ddpm-fig">
<svg viewBox="0 0 760 380" role="img" aria-label="IP-Adapter 整体结构">
  <defs><marker id="ia" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475467"/></marker>
  <marker id="io" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#f79009"/></marker></defs>
  <text x="20" y="40" class="lane">文本分支（原有）</text>
  <rect x="20" y="60" width="90" height="44" rx="7" fill="#fff" stroke="#475467"/><text x="65" y="80" text-anchor="middle">文本提示</text><text x="65" y="96" text-anchor="middle" class="sm">"a dog …"</text>
  <rect x="130" y="60" width="120" height="44" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="190" y="80" text-anchor="middle">CLIP 文本编码器</text><text x="190" y="96" text-anchor="middle" class="sm">冻结</text>
  <rect x="270" y="60" width="110" height="44" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="325" y="80" text-anchor="middle">文本 token</text><text x="325" y="96" text-anchor="middle" class="sm">77 × 768</text>
  <line x1="110" y1="82" x2="128" y2="82" stroke="#475467" marker-end="url(#ia)"/>
  <line x1="250" y1="82" x2="268" y2="82" stroke="#475467" marker-end="url(#ia)"/>
  <text x="20" y="230" class="lane" style="fill:#b54708">图像分支（新增）</text>
  <rect x="20" y="250" width="90" height="44" rx="7" fill="#fff" stroke="#475467"/><text x="65" y="270" text-anchor="middle">图像提示</text><text x="65" y="286" text-anchor="middle" class="sm">参考图</text>
  <rect x="130" y="250" width="120" height="44" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="190" y="270" text-anchor="middle">CLIP 图像编码器</text><text x="190" y="286" text-anchor="middle" class="sm">冻结 · 1024 维</text>
  <rect x="270" y="250" width="110" height="44" rx="7" fill="#fffbeb" stroke="#f79009"/><text x="325" y="270" text-anchor="middle">Linear + LN</text><text x="325" y="286" text-anchor="middle" class="sm">→ 4 × 768</text>
  <line x1="110" y1="272" x2="128" y2="272" stroke="#475467" marker-end="url(#ia)"/>
  <line x1="250" y1="272" x2="268" y2="272" stroke="#475467" marker-end="url(#ia)"/>
  <rect x="400" y="14" width="352" height="352" rx="10" fill="#eef4ff" stroke="#c7d7fe"/>
  <text x="412" y="40" class="lane">U-Net 里每个 cross-attention（16 个）</text>
  <rect x="420" y="60" width="110" height="44" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="475" y="80" text-anchor="middle">K, V 投影</text><text x="475" y="96" text-anchor="middle" class="sm">原有 · 冻结</text>
  <rect x="420" y="155" width="110" height="44" rx="7" fill="#fff" stroke="#2a6fdb"/><text x="475" y="175" text-anchor="middle">Q 投影</text><text x="475" y="191" text-anchor="middle" class="sm">U-Net 特征 · 共用</text>
  <rect x="420" y="250" width="110" height="44" rx="7" fill="#fffbeb" stroke="#f79009"/><text x="475" y="270" text-anchor="middle">K′, V′ 投影</text><text x="475" y="286" text-anchor="middle" class="sm">新增 · 可训练</text>
  <line x1="380" y1="82" x2="418" y2="82" stroke="#475467" marker-end="url(#ia)"/>
  <line x1="380" y1="272" x2="418" y2="272" stroke="#f79009" stroke-width="1.6" marker-end="url(#io)"/>
  <rect x="570" y="90" width="90" height="44" rx="7" fill="#fff" stroke="#475467"/><text x="615" y="117" text-anchor="middle">Attention</text>
  <rect x="570" y="220" width="90" height="44" rx="7" fill="#fff" stroke="#f79009"/><text x="615" y="247" text-anchor="middle">Attention</text>
  <line x1="530" y1="88" x2="568" y2="104" stroke="#475467" marker-end="url(#ia)"/>
  <line x1="530" y1="170" x2="568" y2="122" stroke="#475467" marker-end="url(#ia)"/>
  <line x1="530" y1="184" x2="568" y2="232" stroke="#475467" marker-end="url(#ia)"/>
  <line x1="530" y1="266" x2="568" y2="250" stroke="#f79009" stroke-width="1.6" marker-end="url(#io)"/>
  <circle cx="690" cy="177" r="10" fill="#fff" stroke="#475467"/><text x="690" y="182" text-anchor="middle">+</text>
  <path d="M660,112 L690,112 L690,165" fill="none" stroke="#475467" marker-end="url(#ia)"/>
  <path d="M660,242 L690,242 L690,189" fill="none" stroke="#f79009" stroke-width="1.6" marker-end="url(#io)"/>
  <text x="696" y="222" class="sm" style="fill:#b54708">× λ</text>
  <line x1="700" y1="177" x2="744" y2="177" stroke="#475467" marker-end="url(#ia)"/>
  <text x="722" y="170" text-anchor="middle">Z</text>
  <text x="412" y="325" class="sm">K′、V′ 用文本的 K、V 权重初始化</text>
  <text x="412" y="343" class="sm">两个 softmax 分开算，λ = 0 时就是原模型</text>
  <text x="20" y="330" class="sm">橙色：可训练（投影层 + 16 对 K′、V′，共约 2200 万参数）</text>
  <text x="20" y="348" class="sm">其余全部冻结；图像嵌入每次生成只算一次</text>
</svg>
<figcaption>IP-Adapter 的结构（按论文图 2 重画）。左边是两个冻结的 CLIP 编码器，图像特征经过可训练的投影层变成 4 个 token。右边是 U-Net 里任意一个 cross-attention：Q 共用，文本和图像各有自己的 K、V，两次注意力的结果按 λ 相加。</figcaption>
</figure>

### 2.2 图像 token

CLIP 图像编码器输出一个 1024 维的全局嵌入 $$e_{\text{img}}$$。一个 1024 维的向量直接拿去做 cross-attention 只有 1 个 token，表达力太弱，所以先用一个线性层把它展开成 $$N = 4$$ 个 token，再做 LayerNorm [[1]](#ref-1)：

$$
c_i = \operatorname{LN}\big(\operatorname{reshape}(W_p\, e_{\text{img}} + b_p)\big) \in \mathbb{R}^{N \times 768},\qquad W_p \in \mathbb{R}^{(N \cdot 768) \times 1024}
$$

768 是 SD 1.5 文本 token 的维度（`cross_attention_dim`），这样图像 token 和文本 token 形状一致。

### 2.3 解耦 cross-attention

原来的 cross-attention，$$Z$$ 是 U-Net 某一层的特征（作为 query），$$c_t$$ 是文本 token [[1]](#ref-1)：

$$
Z' = \operatorname{Attention}(Q, K, V) = \operatorname{Softmax}\Big(\frac{Q K^\top}{\sqrt d}\Big) V,\qquad
Q = Z W_q,\ K = c_t W_k,\ V = c_t W_v
$$

IP-Adapter 在**同一层**再加一组只给图像 token 用的 $$W_k', W_v'$$，**共用同一个 Q**，单独算一次注意力，然后相加：

$$
Z_{\text{new}} = \operatorname{Softmax}\Big(\frac{Q K^\top}{\sqrt d}\Big) V + \lambda \cdot \operatorname{Softmax}\Big(\frac{Q K'^\top}{\sqrt d}\Big) V',\qquad
K' = c_i W_k',\ V' = c_i W_v'
$$

| 符号 | 含义 | 是否训练 |
|---|---|---|
| $$W_q, W_k, W_v$$ | 原 cross-attention 的投影 | 冻结 |
| $$W_k', W_v'$$ | 新加的图像 K、V 投影，每个 cross-attention 一对 | 训练，**用 $$W_k, W_v$$ 初始化** |
| $$W_p, b_p$$ 和 LayerNorm | 图像嵌入 → 4 个 token | 训练 |
| $$\lambda$$ | 图像提示的强度 | 推理时设定，训练时为 1 |

SD 1.5 的 U-Net 里有 16 个 cross-attention（编码器 6 个、中间 1 个、解码器 9 个），每个都加一对 $$W_k', W_v'$$。

### 2.4 为什么必须"解耦"

最直接的想法是把 4 个图像 token 拼到 77 个文本 token 后面，变成 81 个 token，走原来的 cross-attention。论文对比过这种做法，效果明显差。原因有两个 [[1]](#ref-1)：

1. **投影不对。** 原来的 $$W_k, W_v$$ 是为文本 token 训练的，现在冻结了。图像 token 只能靠前面的小投影层去"模仿"文本 token 的分布，才能被 $$W_k, W_v$$ 正确理解，这个约束太紧。解耦以后，图像有自己可训练的 $$W_k', W_v'$$。
2. **两种 token 在同一个 softmax 里抢注意力。** 拼在一起时，注意力权重在 81 个 token 上归一化，图像 token 多拿一点，文本 token 就少拿一点，两者互相挤占，也没有一个干净的旋钮来调图像的强度。解耦以后两个 softmax 各自归一化，$$\lambda$$ 就是图像分支的强度。

T2I-Adapter 论文里的 style adapter 用的就是拼接的做法（见 [T2I-Adapter 篇](/deep-learning/t2i-adapter/) 4.1 节），IP-Adapter 可以看作对它的改进。

### 2.5 两个初始化细节

- **$$W_k', W_v'$$ 从 $$W_k, W_v$$ 复制。** 起点上图像分支和文本分支"说同一种语言"，训练只需要在这个基础上调整，收敛更快。
- **没有零初始化。** 和 ControlNet 的零卷积不同，图像分支在 $$\lambda = 1$$ 时从第一步就会改变输出（3.2 节的检查直接显示了这一点）。保证"能退回原模型"的是推理时的 $$\lambda = 0$$，而不是初始化。

> **易错点**
>
> 1. **残差连接在 Transformer block 里，不在 attention 里。** SD 的 `BasicTransformerBlock` 是 `x = attn2(norm2(x), context) + x`，cross-attention 本身只返回 $$Z_{\text{new}}$$。自己实现解耦 attention 时，如果在里面又加一次输入 $$x$$，残差就被加了两次，输出会整体漂移（$$\lambda = 0$$ 的检查也过不了）。
> 2. **λ 乘在注意力输出上，不是乘在图像 token 上。** 先把 $$c_i$$ 乘以 $$\lambda$$ 再算 $$K', V'$$，改变的是 softmax 之后注意力分布的形状，强度和 λ 不再是线性关系，调起来很难预测。

## 3. SD 1.5 上的精确实现

下面的代码直接挂在 [SD 篇](/deep-learning/stable-diffusion/)的 `SD15UNet` 上，结构和官方 `ip-adapter_sd15.bin` 一一对应。

### 3.1 投影层和解耦 cross-attention

```python
import math
import torch
import torch.nn as nn


class ImageProjModel(nn.Module):
    """CLIP 全局图像嵌入 [B, 1024] → 4 个图像 token [B, 4, 768]"""
    def __init__(self, image_embed_dim=1024, cross_attention_dim=768, num_tokens=4):
        super().__init__()
        self.num_tokens, self.dim = num_tokens, cross_attention_dim
        self.proj = nn.Linear(image_embed_dim, num_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        tokens = self.proj(image_embeds).view(-1, self.num_tokens, self.dim)   # [B, 4, 768]
        return self.norm(tokens)


class IPCrossAttention(nn.Module):
    """解耦 cross-attention：同一个 Q，文本和图像各用自己的 K、V，结果相加"""
    def __init__(self, base_attn, image_context_dim=768, num_tokens=4, scale=1.0):
        super().__init__()
        self.base = base_attn.requires_grad_(False)              # 原来的 attn2（冻结）
        dim = base_attn.to_q.out_features
        self.to_k_ip = nn.Linear(image_context_dim, dim, bias=False)   # W'_k（可训练）
        self.to_v_ip = nn.Linear(image_context_dim, dim, bias=False)   # W'_v（可训练）
        with torch.no_grad():                                    # 用文本的 W_k、W_v 初始化
            self.to_k_ip.weight.copy_(base_attn.to_k.weight)
            self.to_v_ip.weight.copy_(base_attn.to_v.weight)
        self.num_tokens, self.scale = num_tokens, scale

    def attend(self, q, k, v):
        b = self.base
        k, v = b.split_heads(k), b.split_heads(v)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(b.head_dim)).softmax(dim=-1)
        return weights @ v                                       # [B, h, N, d]

    def forward(self, x, context):
        # context = [文本 token (77) ; 图像 token (4)]，在最后 num_tokens 个位置切开
        text, image = context[:, :-self.num_tokens], context[:, -self.num_tokens:]
        b = self.base
        q = b.split_heads(b.to_q(x))                             # 共用的 Q，[B, h, N, d]
        out_text = self.attend(q, b.to_k(text), b.to_v(text))                # Attn(Q, K, V)
        out_image = self.attend(q, self.to_k_ip(image), self.to_v_ip(image))  # Attn(Q, K', V')
        out = out_text + self.scale * out_image                  # Z = Z_text + λ·Z_image
        return b.to_out(out.transpose(1, 2).reshape(x.shape))    # 不在这里加残差


def install_ip_adapter(unet, num_tokens=4, scale=1.0):
    """把 SD15UNet 里所有 BasicTransformerBlock 的 attn2 换成解耦版；返回新增的模块"""
    unet.requires_grad_(False)
    new = []
    for m in unet.modules():
        if isinstance(m, BasicTransformerBlock):
            m.attn2 = IPCrossAttention(m.attn2, num_tokens=num_tokens, scale=scale)
            new.append(m.attn2)
    return new


def set_ip_scale(unet, scale):
    for m in unet.modules():
        if isinstance(m, IPCrossAttention):
            m.scale = scale
```

U-Net 的调用方式不用改：图像 token 拼在文本 token 后面作为同一个 `context` 传进去，由每个 `IPCrossAttention` 自己在最后 4 个位置切开。官方实现（以及 Diffusers 的 `IPAdapterAttnProcessor`）也是这样传的。

### 3.2 验证：参数量和 λ = 0

```python
torch.manual_seed(0)
unet = SD15UNet().eval()
z, t = torch.randn(1, 4, 16, 16), torch.tensor([500])
text = torch.randn(1, 77, 768)
with torch.no_grad():
    ref = unet(z, t, text)                                       # 原模型，只有文本

proj = ImageProjModel()
layers = install_ip_adapter(unet)
n_train = sum(p.numel() for p in proj.parameters()) + \
          sum(p.numel() for m in layers for p in m.parameters() if p.requires_grad)
print(len(layers), n_train)                                      # 16 22319616（论文：约 22M）

image_tokens = proj(torch.randn(1, 1024))                        # CLIP 图像嵌入 → [1, 4, 768]
context = torch.cat([text, image_tokens], dim=1)                 # [1, 81, 768]
with torch.no_grad():
    set_ip_scale(unet, 0.0)
    assert torch.allclose(unet(z, t, context), ref, atol=1e-5)   # λ = 0：等于原模型
    set_ip_scale(unet, 1.0)
    print((unet(z, t, context) - ref).abs().mean() > 0)          # tensor(True)：λ = 1 就有影响
```

参数量拆开看：

| 部分 | 参数 |
|---|---|
| 投影层 `Linear(1024, 3072)` + `LayerNorm(768)` | 3,150,336 |
| 16 对 $$W_k', W_v'$$（`768 → 320/640/1280`，无偏置） | 19,169,280 |
| 合计 | **22,319,616** |

只占 U-Net（8.6 亿）的 2.6%，比 T2I-Adapter（7700 万）和 ControlNet（3.6 亿）都小得多。

> **易错点**
>
> 1. **切分 token 的位置要和拼接一致。** 这里约定图像 token 在最后 `num_tokens` 个；如果拼接时放在前面、或者 Plus 版本换成了 16 个 token 而这里还写 4，代码不会报错，但文本和图像的 K、V 会错位。
> 2. **CFG 的负分支也有图像 token。** 无条件分支用的是**全零图像嵌入**经过投影层得到的 token（和训练时"丢掉图像"的做法一致，见 4.2 节），不是直接去掉这 4 个 token。

## 4. 训练

### 4.1 数据和设置

- 数据：约 1000 万图文对（LAION-2B 和 COYO-700M 的子集）。**参考图就是目标图本身**，不需要成对的"参考图 → 结果图"数据； [[1]](#ref-1)
- 底模：SD 1.5，图像编码器 OpenCLIP ViT-H/14，两者和文本编码器都冻结；
- 分辨率 512，AdamW，学习率 $$10^{-4}$$，weight decay 0.01，8 张 V100 训练约 100 万步（每卡 batch 8）。

"参考图 = 目标图"看起来像在学复制，但 CLIP 的全局嵌入只保留了语义和风格，丢掉了大部分空间细节，所以模型学到的是"按这个语义和风格去生成"，而不是逐像素复制。

### 4.2 丢弃策略

为了推理时能用 CFG，训练时按小概率丢掉条件：

| 情况 | 概率 | 做法 |
|---|---|---|
| 只丢文本 | 0.05 | caption 换成空串 |
| 只丢图像 | 0.05 | CLIP 图像嵌入置零 |
| 同时丢 | 0.05 | 两者都丢 |
| 都保留 | 0.85 | |

### 4.3 训练一步

用 3.1 节的模块，`unet` 是已经 `install_ip_adapter` 过的 `SD15UNet`；`vae`、文本编码器、图像编码器都冻结。图像编码器可以用 `transformers` 的 `CLIPVisionModelWithProjection`，它的 `.image_embeds` 就是 1024 维的全局嵌入：

```python
import torch
import torch.nn.functional as F


def ip_adapter_training_step(batch, vae, text_encoder, image_encoder, unet, image_proj,
                             noise_scheduler, empty_text):
    # batch["pixel_values"]: 目标图像 [B, 3, 512, 512]，范围 [-1, 1]
    # batch["clip_images"]: 同一张图按 CLIP 的方式预处理 [B, 3, 224, 224]
    # batch["input_ids"]: [B, 77]；empty_text: 空串的文本 token [1, 77, 768]
    B = batch["pixel_values"].shape[0]
    with torch.no_grad():
        latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
        latents = latents * vae.config.scaling_factor            # 0.18215
        text = text_encoder(batch["input_ids"])[0]                # [B, 77, 768]
        image_embeds = image_encoder(batch["clip_images"]).image_embeds   # [B, 1024]

    # 4.2 节的丢弃：r < 0.05 丢图像，0.05～0.10 丢文本，0.10～0.15 都丢
    r = torch.rand(B, device=latents.device)
    drop_image = (r < 0.05) | ((r >= 0.10) & (r < 0.15))
    drop_text = (r >= 0.05) & (r < 0.15)
    image_embeds = torch.where(drop_image[:, None], torch.zeros_like(image_embeds), image_embeds)
    text = torch.where(drop_text[:, None, None], empty_text.expand_as(text), text)

    noise = torch.randn_like(latents)
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=latents.device)
    noisy = noise_scheduler.add_noise(latents, noise, t)

    context = torch.cat([text, image_proj(image_embeds)], dim=1)  # [B, 81, 768]
    pred = unet(noisy, t, context)                                # 冻结，但不能 no_grad
    return F.mse_loss(pred.float(), noise.float())
```

优化器里只放 `image_proj.parameters()` 和每个 `IPCrossAttention` 的 `to_k_ip`、`to_v_ip`。

> **易错点**
>
> 1. **"丢图像"是把嵌入置零，不是去掉 token。** 置零的嵌入经过投影层仍然是 4 个 token（等于 LayerNorm 的偏置），推理时 CFG 的负分支要用同样的方式得到，否则训练和推理对不上。
> 2. **CLIP 预处理和 VAE 预处理是两套。** 进 VAE 的是 $$[-1, 1]$$、512×512；进 CLIP 的是按 CLIP 的均值方差归一化、224×224。直接把 VAE 的输入喂给 CLIP，不会报错，但嵌入是错的。
> 3. **底模冻结不等于 U-Net 前向可以 no_grad。** 梯度要穿过冻结的 U-Net 才能回到 $$W_k', W_v'$$ 和投影层。

## 5. 推理

### 5.1 λ 和 CFG

推理时有两个强度：

- **λ**（`ip_adapter_scale`）：图像提示的强度，乘在图像分支的注意力输出上；
- **CFG 的 $$w$$**（`guidance_scale`）：正分支同时带文本和图像条件，负分支是空文本 + 全零图像嵌入：

$$
\hat\epsilon = \epsilon_\theta(z_t, \varnothing_t, \varnothing_i) + w\,\big(\epsilon_\theta(z_t, c_t, c_i) - \epsilon_\theta(z_t, \varnothing_t, \varnothing_i)\big)
$$

经验上：

| 用法 | λ | 文本 |
|---|---|---|
| 只用图像提示（生成参考图的"变体"） | 1.0 | 空串 |
| 图像 + 文本（参考图的风格，文本改内容） | 0.5 左右 | 正常写 |
| 只想借一点风格或配色 | 0.2～0.4 | 正常写 |

λ 太大时文本几乎不起作用（图像 token 的信息比文本具体得多），这是 IP-Adapter 最常见的"不听 prompt"问题，调小 λ 就好。

### 5.2 和空间控制一起用

IP-Adapter 改的是 cross-attention，ControlNet 和 T2I-Adapter 改的是卷积特征和 skip，**三者作用在 U-Net 的不同位置，可以直接同时挂上**，各自的强度独立调：

| 组合 | 效果 |
|---|---|
| IP-Adapter + 文本 | 参考图的风格，文本描述的内容 |
| IP-Adapter + ControlNet / T2I-Adapter（姿态、深度、边缘） | 参考图的外观，条件图的结构 |
| IP-Adapter + LoRA | 一般可以叠加；两者都改了风格时要降低各自的强度 |

而且因为底模冻结，同一个 IP-Adapter 一般可以直接用在同一底模的社区微调版本上（比如 SD 1.5 的各种写实、动漫微调模型）。

> **易错点**
>
> 1. **图像嵌入每次生成只需要算一次。** CLIP 编码器不依赖 $$z_t$$ 和 $$t$$，图像 token 在整个采样过程中不变。批量生成、或者同一张参考图配多个 prompt 时，把嵌入缓存下来（6.3 节）。
> 2. **参考图的背景也会被"学"过去。** CLIP 嵌入是整张图的语义，参考图里的背景、画框、文字水印都会影响结果。只想要主体时，先把参考图裁剪到主体。

## 6. 用 Diffusers

### 6.1 加载和生成

```python
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.6)                       # λ

ref = load_image("reference.png")
out = pipe(
    prompt="a cat sitting on a sofa, best quality",
    negative_prompt="blurry, low quality",
    ip_adapter_image=ref,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
```

`load_ip_adapter` 会同时加载图像编码器（同一个仓库的 `models/image_encoder`，即 ViT-H/14），并把 U-Net 的 cross-attention 换成解耦版，也就是第 3 节做的事。`pipe.unload_ip_adapter()` 可以卸载，恢复原模型。 [[3]](#ref-3)

### 6.2 和 ControlNet 一起用

```python
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
).to("cuda")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.7)

out = pipe(
    prompt="best quality",
    image=canny_image,                   # 结构：来自 ControlNet
    ip_adapter_image=ref,                # 外观：来自 IP-Adapter
    controlnet_conditioning_scale=1.0,
    num_inference_steps=30,
).images[0]
```

T2I-Adapter 的 pipeline 同样支持 `load_ip_adapter`，用法相同。

### 6.3 缓存图像嵌入

```python
image_embeds = pipe.prepare_ip_adapter_image_embeds(
    ip_adapter_image=ref, ip_adapter_image_embeds=None, device="cuda",
    num_images_per_prompt=1, do_classifier_free_guidance=True)    # 负分支的全零嵌入也在里面

for prompt in ["a cat on a sofa", "a cat in the snow", "a cat in a spaceship"]:
    out = pipe(prompt=prompt, ip_adapter_image_embeds=image_embeds,
               num_inference_steps=30).images[0]
```

嵌入可以存成文件，下次直接加载，省掉每次跑 CLIP 图像编码器。

### 6.4 常用权重

`h94/IP-Adapter` 仓库里 SD 1.5 的几个版本 [[4]](#ref-4)：

| 权重 | 图像特征 | token 数 | 适合 |
|---|---|---|---|
| `ip-adapter_sd15.bin` | CLIP 全局嵌入 | 4 | 通用：风格、语义、整体外观 |
| `ip-adapter_sd15_light.bin` | CLIP 全局嵌入 | 4 | 更"轻"的图像影响，文本更容易起作用 |
| `ip-adapter-plus_sd15.bin` | CLIP patch 特征 + resampler | 16 | 更贴近参考图的细节 |
| `ip-adapter-plus-face_sd15.bin` | 同上，用裁好的人脸训练 | 16 | 人脸参考 |

FaceID 系列在单独的仓库里，需要额外的人脸识别模型和 LoRA（7.1 节）。SDXL 的权重在同一仓库的 `sdxl_models` 目录。

> **易错点**
>
> 1. **Plus 版本和图像编码器的输出要对上。** Plus 用的是 CLIP 倒数第二层的 patch 特征，不是全局嵌入；Diffusers 会根据权重自动处理，但自己写训练或推理代码时要换输入。
> 2. **用了 IP-Adapter 以后，负向图像也可以给。** `ip_adapter_image_embeds` 里负分支默认是全零嵌入；也可以换成一张"不想要的风格"图的嵌入，作用类似负向提示词。

## 7. 变体、相关方法、要点回顾

### 7.1 变体

| 变体 | 改了什么 | 解决什么 |
|---|---|---|
| **IP-Adapter Plus** | 用 CLIP 的 patch 特征（257 个 token），经过一个 Perceiver 式的 resampler 压成 16 个 token | 全局嵌入丢细节；Plus 更贴近参考图 |
| **Plus-Face / Full-Face** | 同样结构，用裁好的人脸图训练 | 人脸参考 |
| **FaceID** | 图像特征换成人脸识别模型（InsightFace）的身份嵌入，再加一个 U-Net 上的 LoRA | CLIP 嵌入对"是不是同一个人"不敏感；身份嵌入专门编码身份 |
| **FaceID-Plus** | 身份嵌入 + CLIP 人脸特征 | 身份和脸部外观都保留 |
| **SDXL 版** | 底模 SDXL，`cross_attention_dim` 2048 | 更高分辨率；基础版用 ViT-bigG，Plus 版用 ViT-H |

结构的主干（解耦 cross-attention + 投影层）都不变，变的是"用什么特征表示参考图"和"变成几个 token"。 [[2]](#ref-2)

### 7.2 相关方法

| 方法 | 和 IP-Adapter 的关系 |
|---|---|
| T2I-Adapter 的 style adapter | CLIP 图像特征拼到文本 token 后面，共用 K、V；IP-Adapter 把它改成解耦 |
| GLIGEN | 在 Transformer block 里加一个**门控**的新 attention 层，输入是"框 + 文本/图像"的 grounding token，管的是"什么东西放在哪个框里" |
| DreamBooth / LoRA / Textual Inversion | 每个新概念训练一次；保真度更高，但不是"给图就能用" |
| Prompt-to-Prompt | 不训练，只操作 cross-attention 的注意力图来编辑已有生成结果 |

### 7.3 和 T2I-Adapter、ControlNet 放在一起

| | ControlNet | T2I-Adapter | IP-Adapter |
|---|---|---|---|
| 条件 | 空间图（对齐） | 空间图（对齐） | 参考图（不对齐） |
| 管什么 | 结构 | 结构 | 内容、风格 |
| 注入位置 | skip + 中间层 | 编码器 block 内部 | 每个 cross-attention |
| 可训练参数（SD 1.5） | 3.61 亿 | 0.77 亿 | 0.22 亿 |
| 每次生成的额外前向 | 每步一次 | 一次 | 一次（CLIP 编码器） |
| 退回原模型 | 强度 0 | 强度 0 | λ = 0 |

### 7.4 要点回顾

- IP-Adapter 把参考图当提示：冻结的 CLIP 图像编码器 → 全局嵌入 → Linear + LayerNorm → 4 个图像 token。
- 每个 cross-attention 加一对新的 $$W_k', W_v'$$（用文本的 $$W_k, W_v$$ 初始化），共用 Q，和文本注意力分开算再相加：$$Z = \operatorname{Attn}(Q, K, V) + \lambda \operatorname{Attn}(Q, K', V')$$。
- 解耦的原因：冻结的文本 K、V 不适合图像 token；两种 token 在同一个 softmax 里会互相挤占。
- SD 1.5 上只训练 2232 万参数；$$\lambda = 0$$ 精确退回原模型，但初始化不是零，$$\lambda = 1$$ 从第一步就有影响。
- 训练用参考图 = 目标图，文本、图像、两者各以 0.05 的概率丢弃，丢图像就是把嵌入置零，推理时 CFG 的负分支同样用全零嵌入。
- 推理时 λ 越大越像参考图、越不听文本；和 ControlNet / T2I-Adapter 作用在不同位置，可以直接组合成"参考图的外观 + 条件图的结构"。
