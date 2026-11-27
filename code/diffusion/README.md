# Diffusion Learning（DDPM 从零实现）

这个目录用来从零学习 diffusion，重点是理解 DDPM 的核心公式和代码结构，而不是追求复杂模型或高质量生成结果。

代码解读文章发布在博客：[DDPM 从零实现：从公式到代码](https://chaowang15.github.io/deep-learning/ddpm-from-scratch/)

## 设计：一个阶段一个 notebook

整个 pipeline 故意拆成几个阶段，每个阶段对应一个 notebook 和若干 `src/` 模块：

- **notebook 是学习入口**：讲解、可视化、shape 检查、小样本实验。
- **`src/` 放可复用的实现**：公式写在代码旁边，notebook 只调用它。
- **`tests/` 做快速自检**：改完 `src/` 后跑一下，确认 shape 和公式没写错。

| 阶段 | Notebook | 涉及的 `src/` | 状态 |
|---|---|---|---|
| 00 数据 | `00_check_dataset.ipynb` | `datasets.py` | ✅ 完成 |
| 01 前向过程 | `01_forward_process.ipynb` | `schedule.py`, `forward_process.py` | ✅ 完成 |
| 02 噪声预测网络 | `02_noise_prediction.ipynb` | `time_embedding.py`, `unet.py` | 🟡 `src/` 已实现，notebook 待写 |
| 03 DDPM 训练 | `03_ddpm_training.ipynb` | `diffusion.py`, `train.py` | 🟡 `src/` 已实现，notebook 待写 |
| 04 采样 | `04_sampling.ipynb` | `sample.py` | 🟡 `src/` 已实现，notebook 待写 |
| 05 DDIM / CFG | `05_ddim_and_cfg.ipynb` | `sample.py`（`sample_ddim`）、`unet.py`（`ClassConditionedTinyUNet`） | 🟡 `src/` 已实现，notebook 待写 |

每个 `src/*.py` 第一行的 docstring 都写了它属于哪个阶段、对应哪个 notebook。

## 目录

```text
code/diffusion/
├── notebooks/      # 学习入口，按阶段编号 00–05
├── src/            # 实现（import 方式：from src.schedule import ...）
├── tests/          # pytest 自检
├── scripts/        # 命令行入口（check_env.py；以后 train.py / sample.py）
├── configs/        # 每个数据集一个 yaml
├── outputs/        # 生成结果，已被 .gitignore 忽略（只保留空目录）
├── assets/         # 图示
├── pyproject.toml  # 依赖 + pytest 配置
├── requirements.txt          # 默认（Mac / CPU）
├── requirements-cuda121.txt  # Windows + NVIDIA GPU
└── AGENTS.md       # 给 coding agent 看的项目说明
```

数据集默认下载到 `code/data/`（从本目录看是 `../data`），已被 git 忽略。这里只跑 notebook 里的小样本例子；需要真正大规模训练时另开代码仓库。

## 环境

### Mac（Apple Silicon，主力机）

```bash
cd code/diffusion
conda create -n diffusion python=3.11 -y
conda activate diffusion
python -m pip install -r requirements.txt
python -m ipykernel install --user --name diffusion --display-name "Python (conda diffusion)"

python scripts/check_env.py     # 期望 device: mps
python -m pytest -q             # 期望全部 passed
```

PyPI 上的 macOS 版 torch 自带 MPS（Apple GPU）支持，不需要额外的 index。

### Windows + RTX 3080

```powershell
cd code\diffusion
conda create -n diffusion python=3.10 -y
conda activate diffusion
python -m pip install -r requirements-cuda121.txt
python scripts\check_env.py     # 期望 device: cuda
```

## 学习顺序

0. `00_check_dataset.ipynb`：确认 MNIST / CIFAR-10 能加载，shape 和归一化正确。
1. `schedule.py` + `forward_process.py` + `01_forward_process.ipynb`：理解 beta、alpha、alpha_bar，以及 `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise`。
2. `time_embedding.py` + `unet.py` + `02_noise_prediction.ipynb`：模型输入 noisy image 和 timestep，输出预测噪声；先在一个小 batch 上确认 loss 能下降。
3. `diffusion.py` + `train.py` + `03_ddpm_training.ipynb`：把 schedule、forward process、U-Net、loss 连接起来。
4. `sample.py` + `04_sampling.ipynb`：从纯噪声开始一步步反向去噪。
5. `05_ddim_and_cfg.ipynb`：DDPM 主线跑通后再看 DDIM 和 classifier-free guidance。
