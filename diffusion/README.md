# Diffusion Learning

这个目录用来从零学习 diffusion，重点是理解 DDPM 的核心公式和代码结构，而不是一上来追求复杂模型或高质量生成结果。

当前阶段先搭好项目骨架。数据集和配置相关代码已经准备好，核心 diffusion 模块会在后续一步步手写。

## Directory

```text
diffusion/
├── notebooks/
│   ├── 00_check_dataset.ipynb
│   ├── 01_forward_process.ipynb
│   ├── 02_noise_prediction.ipynb
│   ├── 03_ddpm_training.ipynb
│   ├── 04_sampling.ipynb
│   └── 05_ddim_and_cfg.ipynb
├── src/
│   ├── datasets.py
│   ├── schedule.py
│   ├── forward_process.py
│   ├── time_embedding.py
│   ├── unet.py
│   ├── diffusion.py
│   ├── train.py
│   ├── sample.py
│   └── utils.py
├── configs/
│   ├── mnist_ddpm.yaml
│   ├── fashion_mnist_ddpm.yaml
│   └── cifar10_ddpm.yaml
├── outputs/
│   ├── samples/
│   ├── checkpoints/
│   └── logs/
├── assets/
│   └── diagrams/
├── AGENTS.md
├── requirements.txt
└── README.md
```

## Learning Path

推荐按这个顺序来：

0. `00_check_dataset.ipynb`
   确认 MNIST/CIFAR-10 能正常加载，图片 shape 和 normalization 都正确。

1. `schedule.py` + `01_forward_process.ipynb`
   理解 beta、alpha、alpha_bar，以及图片如何从 `x_0` 变成 `x_t`。

2. `forward_process.py`
   手写加噪公式：
   `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise`

3. `time_embedding.py`
   理解为什么模型需要知道当前 timestep。

4. `unet.py`
   写一个小型 U-Net，让它输入 noisy image 和 timestep，输出预测噪声。

5. `diffusion.py` + `03_ddpm_training.ipynb`
   把 schedule、forward process、U-Net、loss 连接起来。

6. `sample.py` + `04_sampling.ipynb`
   从纯噪声开始一步步反向去噪生成图片。

7. `05_ddim_and_cfg.ipynb`
   等 DDPM 主线跑通后，再看 DDIM 和 classifier-free guidance。

## Environment

This project uses a Windows Conda environment named `diffusion`.

Run these commands in Windows PowerShell:

```powershell
cd C:\dev\deep_learning_lab\diffusion

conda create -n diffusion python=3.10 -y
conda activate diffusion

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m ipykernel install --user --name diffusion --display-name "Python (conda diffusion)"
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected result on the desktop machine:

```text
2.4.1+cu121
True
NVIDIA GeForce RTX 3080
```

如果想打开 Notebook：

```powershell
conda activate diffusion
jupyter notebook
```

In VS Code Notebook, choose the kernel:

```text
Python (conda diffusion)
```

## Dataset

一开始推荐使用 `fashion_mnist`，它是 28x28 灰度图，训练快，也比 MNIST 更容易看出生成结果是否有结构。

数据集默认放在大项目根目录的 `data/` 目录下。也就是说，从当前 `diffusion/` 目录看，默认路径是 `../data`。

数据集加载代码在 `src/datasets.py`，配置在 `configs/*.yaml`。这部分先作为工具代码使用，后面学习重点放在 diffusion 本身。

## Codex Context

`AGENTS.md` records project-specific Codex notes: environment choice, current progress, coding/comment style, and recommended next steps. Keep it updated when the learning path changes.
