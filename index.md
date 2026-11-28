---
layout: note
title: "Deep Learning 学习笔记"
description: "Deep learning technical notes and study blog."
tag_panel: true
---

# Deep Learning 学习笔记

<p class="note-index-intro">
  这里整理我在学习和复盘 deep learning 过程中的技术笔记，重点放在数学基础、生成模型、Transformer 架构和代码实现细节。内容更偏向“把一件事从直觉、公式到实现串起来”，方便之后继续补充、修订和查阅。
</p>

<div class="tag-bar">
  {% include tag-filters.html %}
</div>

<div class="tag-status" data-tag-status hidden>
  <span>标签 <b class="tag-status-name" data-tag-status-name></b> · <span data-tag-status-count></span> 篇</span>
  <button type="button" class="tag-status-clear" data-tag-filter="">清除筛选 ×</button>
</div>

<div class="note-index-list" data-note-list>
  {%- assign notes = site.pages | where_exp: "p", "p.summary" | sort: "date" | reverse -%}
  {%- for note in notes %}
  <article class="note-index-card" data-note-card data-tags="{% for t in note.tags %}{{ t | slugify }} {% endfor %}">
    <a class="note-index-card-head" href="{{ note.url | relative_url }}">
      <strong>{{ note.title }}</strong>
      <time datetime="{{ note.date | date: '%Y-%m-%d' }}">{{ note.date | date: '%Y.%m' }}</time>
    </a>
    <span class="note-index-summary">{{ note.summary }}</span>
    {%- if note.tags %}
    <span class="note-index-tags">
      {%- for t in note.tags %}
      {%- assign td = site.data.tags.tags | where: "name", t | first %}
      <button type="button" class="tag-chip tag-chip--sm" data-group="{{ td.group | default: 'other' }}" data-tag-filter="{{ t | slugify }}" data-tag-label="{{ t }}" aria-pressed="false">{{ t }}</button>
      {%- endfor %}
    </span>
    {%- endif %}
  </article>
  {%- endfor %}
</div>

<p class="tag-empty" data-tag-empty hidden>这个标签下暂时还没有文章。</p>
