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
keywords: >-
  English keywords shown on the card.
summary: >-
  卡片上的中文摘要。
---
```

没有 `summary` 的页面不会出现在首页列表里。
