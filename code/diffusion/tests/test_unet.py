"""Sanity checks for Stage 02: timestep embedding and TinyUNet."""

import pytest
import torch

from src.time_embedding import sinusoidal_timestep_embedding
from src.unet import SpatialSelfAttention, TimeResBlock, TinyUNet


def test_embedding_shape_and_range():
    t = torch.tensor([0, 1, 150, 299])
    emb = sinusoidal_timestep_embedding(t, 128)
    assert emb.shape == (4, 128)
    assert emb.abs().max() <= 1.0
    # t = 0: sin part is 0, cos part is 1
    assert torch.allclose(emb[0, :64], torch.zeros(64)) and torch.allclose(emb[0, 64:], torch.ones(64))
    # different t give different embeddings
    assert not torch.allclose(emb[1], emb[2])


def test_embedding_rejects_odd_dim():
    with pytest.raises(ValueError):
        sinusoidal_timestep_embedding(torch.tensor([1]), 7)


def test_resblock_and_attention_shapes():
    x = torch.randn(2, 8, 14, 14)
    emb = torch.randn(2, 32)
    assert TimeResBlock(8, 16, 32)(x, emb).shape == (2, 16, 14, 14)
    assert SpatialSelfAttention(8, num_heads=4)(x).shape == x.shape


@pytest.mark.parametrize("shape", [(2, 1, 28, 28), (2, 3, 32, 32)])
@pytest.mark.parametrize("attn", [False, True])
def test_tiny_unet_output_matches_input(shape, attn):
    model = TinyUNet(in_channels=shape[1], base_channels=16, time_dim=32, use_attention=attn)
    x_t = torch.randn(shape)
    t = torch.randint(0, 300, (shape[0],))
    assert model(x_t, t).shape == x_t.shape
