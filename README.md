# Deep Learning 学习笔记

Chao Wang 的深度学习技术笔记，发布在
[https://chaowang15.github.io/deep-learning/](https://chaowang15.github.io/deep-learning/)。

使用 Jekyll 构建，由 GitHub Pages 部署。每篇笔记是一个目录下的 `index.md`，图片放在 `assets/<笔记名>/`。

本地预览：

```bash
jekyll serve --baseurl /deep-learning
```

然后打开 `http://localhost:4000/deep-learning/`。

## 新增笔记

首页卡片由 `index.md` 自动生成，按 `date` 从新到旧排序，不需要手动编辑首页。新笔记在 front matter 里写好这几项即可：

```yaml
---
layout: note
title: "标题"
description: "一句话描述"
date: 2026-11-26 14:00:00 -0800   # 用完整时间，同一天的多篇也能排序
tags: [Diffusion, Sampling]           # 1-3 个，必须用 _data/tags.yml 里的名字
keywords: >-
  English keywords (only used for the page's <meta>, not shown on the card).
summary: >-
  卡片上的中文摘要。
---
```

没有 `summary` 的页面不会出现在首页列表里。

## 标签

首页右侧的 Topics 面板和卡片上的标签来自每篇笔记 front matter 里的 `tags`。可用的标签、分组和显示顺序在 [`_data/tags.yml`](_data/tags.yml)：

- 每篇选 1–3 个，名字要和 `tags.yml` 里的 `name` 完全一致（含空格和大小写；带 `&` 的要加引号，如 `"Video & Avatar"`）。
- 现有标签不够用时，先在 `tags.yml` 里加一项（选好 `group`），再在笔记里用。不在 `tags.yml` 里的标签不会出现在筛选面板。
- 可以用 `#tag=<slug>` 直接链接到筛选结果，比如 `/deep-learning/#tag=diffusion`。

## 系列（上一篇 / 下一篇）

文章底部的「系列」导航来自 [`_data/series.yml`](_data/series.yml)：每个系列按阅读顺序列出页面 URL（如 `/ddim/`）。新文章把自己的 URL 插到合适系列的合适位置即可；不在任何系列里的文章底部不显示导航。

## 其他自动生成的内容

- 首页卡片和文章标题下方的日期显示到「日」（`date` 字段），阅读时长按正文字数估算（去掉代码块和公式，约 500 字/分钟），不需要手写。
- 首页侧栏的 Archive 按月份统计文章数，点击可按月筛选（`#month=2026-11`）。
- 深色模式：右上角按钮切换，默认跟随系统，选择记在浏览器里。文章里的图（`figure`、`img`）在深色模式下放在浅色卡片上，所以画 SVG 时按浅色背景设计即可。
