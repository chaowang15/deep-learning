---
layout: note
title: "Deep Learning 学习笔记"
description: "Deep learning technical notes and study blog."
---

# Deep Learning 学习笔记

<p class="note-index-intro">
  这里整理我在学习和复盘 deep learning 过程中的技术笔记，重点放在数学基础、生成模型、Transformer 架构和代码实现细节。内容更偏向“把一件事从直觉、公式到实现串起来”，方便之后继续补充、修订和查阅。
</p>

<div class="note-index-list">
  <a class="note-index-card" href="/deep-learning/muse-realtime-avatar/">
    <span class="note-index-card-head">
      <strong>Meta Muse Realtime Avatar 技术解析</strong>
      <time datetime="2026-11">2026.11</time>
    </span>
    <span class="note-index-keywords">Audio-driven causal DiT, speech VQ tokens, KV cache, Self-Forcing, DMD distillation, real-time 25 fps inference.</span>
    <span class="note-index-summary">拆解 Meta Muse 实时数字人的网络架构：语音 token 如何同时驱动声音和画面，块因果 DiT 每个模块的输入输出与张量形状，参考图锚点和滚动 KV cache，以及如何用 Self-Forcing + DMD 把 40 步 teacher 蒸馏成 2 步实时 student。配论文风格架构图，区分博客原文与推断。</span>
  </a>
  <a class="note-index-card" href="/deep-learning/diffusion-study/">
    <span class="note-index-card-head">
      <strong>Diffusion Models 扩散模型学习</strong>
      <time datetime="2026-06">2026.06</time>
    </span>
    <span class="note-index-keywords">DDPM, forward process, reverse process, ELBO, beta schedule, training loss, and sampling.</span>
    <span class="note-index-summary">这篇笔记整理 Diffusion Models 和 DDPM 的核心流程，从 forward process 如何逐步加噪，到 reverse process 如何学习去噪分布，再展开 ELBO 推导、training loss、beta schedule 与 sampling 过程。内容更偏公式推导和直觉解释结合，适合作为生成模型笔记中 diffusion 部分的专题页。</span>
  </a>
  <a class="note-index-card" href="/deep-learning/transformer-study/">
    <span class="note-index-card-head">
      <strong>Transformer 学习 / Transformer Study</strong>
      <time datetime="2026-05">2026.05</time>
    </span>
    <span class="note-index-keywords">Seq2Seq and RNN review, attention, Transformer encoder-decoder flow, multi-head attention, loss design, and inference.</span>
    <span class="note-index-summary">这篇是对 Transformer 的系统学习记录，从 RNN/Seq2Seq 回顾开始，引出 Attention 和 Self-Attention，再展开 Encoder、Decoder、Multi-Head Attention、mask、loss design 和 inference。它会同时保留公式、图解和 PyTorch 风格代码，适合按流程复盘整个架构。</span>
  </a>
  <a class="note-index-card" href="/deep-learning/generative-models-2-gan/">
    <span class="note-index-card-head">
      <strong>Generative Models 2: GAN</strong>
      <time datetime="2026-01">2026.01</time>
    </span>
    <span class="note-index-keywords">GAN intuition, adversarial objectives, alternating training, generator/discriminator structure, and implementation flow.</span>
    <span class="note-index-summary">这篇关注 GAN 的核心思想：Generator 和 Discriminator 如何通过 adversarial training 相互推动。内容包括目标函数直觉、训练不稳定的原因、Generator/Discriminator 的代码结构，以及 alternating training 的实现细节。</span>
  </a>
  <a class="note-index-card" href="/deep-learning/generative-models-1-vae/">
    <span class="note-index-card-head">
      <strong>Generative Models 1: VAE</strong>
      <time datetime="2025-09">2025.09</time>
    </span>
    <span class="note-index-keywords">AutoEncoders, VAE derivation, reparameterization trick, Conv-VAE, and VQ-VAE references.</span>
    <span class="note-index-summary">这篇从 AutoEncoder 的直觉出发，解释为什么需要 latent distribution，并推导 VAE 的 ELBO、reconstruction term 和 KL regularization。后半部分连接到 PyTorch 实现、reparameterization trick、Conv-VAE，以及 VQ-VAE 相关工作。</span>
  </a>
  <a class="note-index-card" href="/deep-learning/math-foundation/">
    <span class="note-index-card-head">
      <strong>深度学习中的数学基础 / Deep Learning Math Foundation</strong>
      <time datetime="2025-01">2025.01</time>
    </span>
    <span class="note-index-keywords">Bayes rule, Gaussian distributions, expected value, variance, KL divergence, BCE, transposed convolution, and trace.</span>
    <span class="note-index-summary">这篇主要整理 deep learning 中反复会用到的数学工具，从贝叶斯公式、期望方差、高斯分布推导，到 KL Divergence、BCE Loss 和转置卷积。它更像后续生成模型笔记的基础索引，遇到公式推导时可以回头查。</span>
  </a>
  <a class="note-index-card" href="/deep-learning/fast-video-diffusion/">
    <span class="note-index-card-head">
      <strong>实时视频扩散的关键技术：DMD、Self-Forcing 与推理加速</strong>
      <time datetime="2026-11">2026.11</time>
    </span>
    <span class="note-index-keywords">CFG distillation, Distribution Matching Distillation (DMD/DMD2), CausVid, Self-Forcing, 4-bit QAT, CUDA Graph, DiT.</span>
    <span class="note-index-summary">Muse 解析的配套笔记，把实时视频扩散用到的技术逐个展开：从 CFG 讲到 DMD 如何用两个 score 网络之差训练少步生成器，DMD2 的四个改进，CausVid 的双向到因果蒸馏，Self-Forcing 如何消除自回归的训练推理不一致，以及 4-bit 量化和 CUDA Graph。附论文、代码和建议的学习顺序。</span>
  </a>
</div>
