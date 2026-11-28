# code/：学习代码

博客文章对应的学习代码。Jekyll 构建时排除整个 `code/` 目录，它不会出现在网站上，只在 GitHub 上可见。

| 目录 | 内容 |
|---|---|
| [`diffusion/`](diffusion/) | DDPM 从零实现，一个阶段一个 notebook |
| [`vae/`](vae/) | VAE / VQ-VAE 实验 |
| [`gan/`](gan/) | GAN / DCGAN / WGAN-GP / VQGAN 实验 |
| [`flow/`](flow/) | Flow Matching / Rectified Flow 实验 |
| [`_template/`](_template/) | 新论文 / 新主题学习项目的骨架，复制后改名即可 |

约定：

- 数据放 `code/data/`，生成结果放各项目的 `outputs/`，都不进 git。
- 这里只跑小样本例子；真正的大规模训练另开仓库。
- 开新主题：`cp -r code/_template code/<topic>`，然后按 `_template/README.md` 填。
