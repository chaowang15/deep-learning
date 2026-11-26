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
  {%- assign notes = site.pages | where_exp: "p", "p.summary" | sort: "date" | reverse -%}
  {%- for note in notes %}
  <a class="note-index-card" href="{{ note.url | relative_url }}">
    <span class="note-index-card-head">
      <strong>{{ note.title }}</strong>
      <time datetime="{{ note.date | date: '%Y-%m-%d' }}">{{ note.date | date: '%Y.%m' }}</time>
    </span>
    <span class="note-index-keywords">{{ note.keywords }}</span>
    <span class="note-index-summary">{{ note.summary }}</span>
  </a>
  {%- endfor %}
</div>
