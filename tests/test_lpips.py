import math
import types

import torch
import pytest

import os
import sys

# Ensure repo root is on path for local package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ulsd_model.models.lpips as lpips


class DummyVGG(torch.nn.Module):
    """Deterministic VGG stub that returns 5 feature maps
    with channel sizes [64, 128, 256, 512, 512], keeping spatial size.
    The features are simple channel-tilings of the input to ensure
    LPIPS(x, x) == 0 when using squared differences.
    """

    def __init__(self, requires_grad: bool = False, pretrained: bool = False):
        super().__init__()
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, X):
        # X: [N, 3, H, W]
        N, C, H, W = X.shape

        def expand_channels(x, out_c):
            repeat = (out_c + C - 1) // C
            x_rep = x.repeat(1, repeat, 1, 1)
            return x_rep[:, :out_c, :, :]

        h1 = expand_channels(X, 64)
        h2 = expand_channels(X, 128)
        h3 = expand_channels(X, 256)
        h4 = expand_channels(X, 512)
        h5 = expand_channels(X, 512)
        VggOutputs = lpips.namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        return VggOutputs(h1, h2, h3, h4, h5)


def test_spatial_average_basic():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # [1,1,2,2]
    m = lpips.spatial_average(x, keepdim=True)
    assert m.shape == (1, 1, 1, 1)
    assert torch.allclose(m, torch.tensor([[[[2.5]]]]))


def test_scaling_layer_normalization():
    layer = lpips.ScalingLayer()
    x = torch.zeros(1, 3, 2, 2)
    y = layer(x)
    # y = (0 - shift) / scale
    expected = (-layer.shift) / layer.scale
    assert torch.allclose(y, expected.expand_as(y))


def test_netlinlayer_shape():
    layer = lpips.NetLinLayer(chn_in=64, chn_out=1, use_dropout=False)
    x = torch.randn(2, 64, 8, 8)
    y = layer(x)
    assert y.shape == (2, 1, 8, 8)


def test_lpips_forward_monkeypatched(monkeypatch):
    # Avoid external weights and heavy backbones by stubbing load_state_dict and vgg16
    monkeypatch.setattr(lpips, "vgg16", DummyVGG, raising=True)
    monkeypatch.setattr(
        lpips.LPIPS,
        "load_state_dict",
        lambda self, sd, strict=False: None,
        raising=True,
    )
    # Prevent file access for weights by stubbing torch.load used inside the module
    monkeypatch.setattr(lpips.torch, "load", lambda *a, **k: {}, raising=True)

    torch.manual_seed(0)
    model = lpips.LPIPS()

    x = torch.rand(1, 3, 32, 32)
    y = torch.rand(1, 3, 32, 32)

    d_xy = model(x, y, normalize=True)
    d_yx = model(y, x, normalize=True)
    d_xx = model(x, x, normalize=True)

    # Shapes and basic properties
    assert d_xy.shape == (1, 1, 1, 1)
    assert d_xx.shape == (1, 1, 1, 1)
    assert torch.isfinite(d_xy).all()
    assert torch.isfinite(d_xx).all()

    # Symmetry
    assert torch.allclose(d_xy, d_yx, atol=1e-6)

    # Identity distance is ~0 with our deterministic dummy backbone
    assert torch.allclose(d_xx, torch.zeros_like(d_xx), atol=1e-6)

    # Different images should not match exactly (magnitude non-zero)
    assert (d_xy.abs() > 1e-8).all()
