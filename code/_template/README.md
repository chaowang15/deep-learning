# <Topic>：<论文 / 方法名> 学习代码

> 复制方式：`cp -r code/_template code/<topic>`，然后替换本文件里的 `<...>`。

## 设计：一个阶段一个 notebook

把方法拆成几个可以单独理解、单独检查的阶段。每个阶段：

1. 一个 notebook `notebooks/NN_<stage>.ipynb`：讲解 + 可视化 + shape 检查 + 小样本实验。
2. 对应的 `src/*.py`：实现，第一行 docstring 写 `Stage NN (notebooks/NN_<stage>.ipynb): ...`。
3. `tests/test_<module>.py`：shape、数值、公式的快速自检。

| 阶段 | Notebook | 涉及的 `src/` | 状态 |
|---|---|---|---|
| 00 数据 | `00_check_dataset.ipynb` | `datasets.py` | ⏳ |
| 01 <核心步骤 1> | `01_<...>.ipynb` | `<...>.py` | ⏳ |
| 02 <核心步骤 2> | `02_<...>.ipynb` | `<...>.py` | ⏳ |

经验：先做"不需要训练就能验证"的部分（数据、公式、前向计算），再做模型，最后训练和推理。面试准备时，每个阶段都应该能脱离其他阶段讲清楚输入、输出和 tensor shape。

## 目录

```text
notebooks/   学习入口
src/         实现（from src.xxx import ...）
tests/       pytest 自检
scripts/     命令行入口（check_env.py / train.py / sample.py）
configs/     yaml 配置
outputs/     生成结果（git 忽略）
assets/      图示
```

## 环境

```bash
cd code/<topic>
python -m pip install -r requirements.txt
python scripts/check_env.py
python -m pytest -q
```
