# Diffusion Learning

这个目录用来从零学习 diffusion，重点是理解 DDPM 的核心公式和代码结构，而不是一上来追求复杂模型或高质量生成结果。

当前阶段先搭好项目骨架。数据集和配置相关代码已经准备好，核心 diffusion 模块会在后续一步步手写。

## Directory

```text
diffusion/
├── notebooks/
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
├── requirements.txt
└── README.md
```

## Learning Path

推荐按这个顺序来：

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

```bash
cd /mnt/c/dev/deep_learning_lab/diffusion

python3 -m venv .venv-diffusion
source .venv-diffusion/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果想打开 Notebook：

```bash
source .venv-diffusion/bin/activate
jupyter notebook
```

## Dataset

一开始推荐使用 `fashion_mnist`，它是 28x28 灰度图，训练快，也比 MNIST 更容易看出生成结果是否有结构。

数据集默认放在大项目根目录的 `data/` 目录下。也就是说，从当前 `diffusion/` 目录看，默认路径是 `../data`。

数据集加载代码在 `src/datasets.py`，配置在 `configs/*.yaml`。这部分先作为工具代码使用，后面学习重点放在 diffusion 本身。
