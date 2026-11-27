from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DatasetName = Literal["mnist", "fashion_mnist", "cifar10"]
PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetConfig:
    name: DatasetName = "fashion_mnist"
    data_dir: str = "../data"
    image_size: int = 28
    channels: int = 1
    batch_size: int = 128
    num_workers: int = 2
    download: bool = True


def image_settings_for_dataset(name: DatasetName) -> tuple[int, int]:
    """Return the natural image size and channel count for a supported dataset."""
    if name == "cifar10":
        return 32, 3
    if name in {"mnist", "fashion_mnist"}:
        return 28, 1
    raise ValueError(f"Unsupported dataset: {name}")


def build_transform(image_size: int) -> transforms.Compose:
    """Convert images to tensors normalized from [0, 1] into [-1, 1]."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )


def get_dataset(config: DatasetConfig, train: bool = True):
    root = Path(config.data_dir)
    if not root.is_absolute():
        root = PROJECT_DIR / root
    transform = build_transform(config.image_size)

    if config.name == "mnist":
        return datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=config.download,
        )

    if config.name == "fashion_mnist":
        return datasets.FashionMNIST(
            root=root,
            train=train,
            transform=transform,
            download=config.download,
        )

    if config.name == "cifar10":
        return datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=config.download,
        )

    raise ValueError(f"Unsupported dataset: {config.name}")


def make_dataloader(config: DatasetConfig, train: bool = True) -> DataLoader:
    dataset = get_dataset(config, train=train)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Convert image tensors from [-1, 1] back to [0, 1] for display."""
    return (images.clamp(-1.0, 1.0) + 1.0) / 2.0
