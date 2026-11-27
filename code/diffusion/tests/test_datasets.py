"""Sanity checks for src/datasets.py (Stage 00). No download needed."""

import pickle

import torch

from src.datasets import build_transform, denormalize


def test_transform_is_picklable():
    # DataLoader workers on macOS use "spawn" and pickle the transform.
    pickle.dumps(build_transform(28))


def test_denormalize_roundtrip():
    x = torch.rand(2, 1, 4, 4)
    assert torch.allclose(denormalize(x * 2.0 - 1.0), x)
