import os
import sys

import pytest
import torch
import torch.nn as nn

# Ensure repo root on sys.path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ulsd_model.models.discriminator import Discriminator


def compute_spatial_dims(height, width, kernels, strides, paddings):
    """Return spatial dims after sequential convs with integer hyperparams."""
    h, w = height, width
    for kernel, stride, padding in zip(kernels, strides, paddings):
        h = (h + 2 * padding - kernel) // stride + 1
        w = (w + 2 * padding - kernel) // stride + 1
    return h, w


def test_discriminator_forward_shape_matches_conv_config():
    model = Discriminator()
    x = torch.randn(2, model.im_channels, 64, 64)

    out = model(x)
    expected_h, expected_w = compute_spatial_dims(
        64, 64, model.kernels, model.strides, model.paddings
    )

    assert out.shape == (2, 1, expected_h, expected_w)


def test_discriminator_blocks_configured_with_bias_and_norm_rules():
    cfg = {
        "conv_channels": [32, 64],
        "kernels": [4, 4, 3],
        "strides": [2, 2, 1],
        "paddings": [1, 1, 1],
    }
    model = Discriminator(**cfg)

    first_conv, first_norm, first_act = model.conv_blocks[0]
    middle_conv, middle_norm, middle_act = model.conv_blocks[1]
    final_conv, final_norm, final_act = model.conv_blocks[2]

    # Bias present only on first and last conv layers
    assert first_conv.bias is not None
    assert middle_conv.bias is None
    assert final_conv.bias is not None

    # BatchNorm skipped for first and final layers, used in the middle
    assert isinstance(first_norm, nn.Identity)
    assert isinstance(final_norm, nn.Identity)
    assert isinstance(middle_norm, nn.BatchNorm2d)

    # Final activation should be identity (no LeakyReLU applied)
    assert isinstance(final_act, nn.Identity)


def test_discriminator_raises_when_hyperparameter_lengths_mismatch():
    with pytest.raises(AssertionError):
        Discriminator(
            conv_channels=[64, 128],
            kernels=[4, 4],
            strides=[2, 2, 2],
            paddings=[1, 1, 1],
        )
