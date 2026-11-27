"""Replace with real checks: shapes, value ranges, and formulas from the paper."""

import torch


def test_torch_works():
    assert torch.ones(2, 3).sum().item() == 6
