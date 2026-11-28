"""Shared helpers for the VAE / VQ-VAE blog experiments (FashionMNIST, CPU)."""
import numpy as np, torch
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "FashionMNIST" / "raw"
OUT = Path(__file__).resolve().parents[2] / "assets" / "generative-models-1-vae"
OUT.mkdir(parents=True, exist_ok=True)


def load_fmnist(split="train"):
    """Returns images in [0, 1] as float tensor [N, 1, 28, 28] and labels [N]."""
    pre = "train" if split == "train" else "t10k"
    with open(DATA / f"{pre}-images-idx3-ubyte", "rb") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    with open(DATA / f"{pre}-labels-idx1-ubyte", "rb") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    return torch.from_numpy(x.astype(np.float32) / 255.0), torch.from_numpy(y.astype(np.int64))
