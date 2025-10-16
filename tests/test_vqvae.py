import os
import sys
import types

import pytest
import torch


# Ensure repo root is on path for local package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def install_unetblocks_stub():
    """Install lightweight stubs for ddpm_model.models.UNetBlocks before importing vqvae.

    The real repo does not include ddpm_model, so we provide minimal blocks that
    match the constructor signatures and perform simple (down/up) sampling.
    """
    # If user's implementation is available, prefer it
    try:
        __import__("ddpm_model.models.UNetBlocks")
        return  # real blocks available, no stub needed
    except Exception:
        pass

    # Create package hierarchy: ddpm_model -> ddpm_model.models -> UNetBlocks
    ddpm_model = types.ModuleType("ddpm_model")
    models = types.ModuleType("ddpm_model.models")
    unetblocks = types.ModuleType("ddpm_model.models.UNetBlocks")

    import torch.nn as nn
    import torch.nn.functional as F

    class DownSamplingBlock(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            time_emb_dim=None,
            down_sample=False,
            num_heads=1,
            num_layers=1,
            use_attn=False,
            grp_norm_chanels=8,
        ):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.down = bool(down_sample)

        def forward(self, x, time_emb=None):
            x = self.conv(x)
            if self.down:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
            return x

    class BottleNeck(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            time_emb_dim=None,
            num_heads=1,
            num_layers=1,
            grp_norm_chanels=8,
        ):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        def forward(self, x, time_emb=None):
            return self.conv(x)

    class UpSamplingBlock(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            time_emb_dim=None,
            up_sample=False,
            num_heads=1,
            num_layers=1,
            use_attn=False,
            grp_norm_chanels=8,
        ):
            super().__init__()
            self.up = bool(up_sample)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        def forward(self, x, time_emb=None, out_down=None):
            if self.up:
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            return self.conv(x)

    # Attach classes to stub module
    unetblocks.DownSamplingBlock = DownSamplingBlock
    unetblocks.BottleNeck = BottleNeck
    unetblocks.UpSamplingBlock = UpSamplingBlock

    # Register modules
    sys.modules["ddpm_model"] = ddpm_model
    sys.modules["ddpm_model.models"] = models
    sys.modules["ddpm_model.models.UNetBlocks"] = unetblocks


def make_min_config():
    """Return a small, consistent VQVAE config for tests."""
    return {
        "down_channels": [16, 32, 64],  # 2 transitions
        "mid_channels": [64, 64],  # in==out==last down channel
        "down_sample": [
            True,
            True,
            False,
        ],  # len == len(down_channels) per model's assert
        "num_down_layers": 1,
        "num_mid_layers": 1,
        "num_up_layers": 1,
        "attn_down": [False, False, False],  # len == len(down_channels)
        "z_channels": 8,
        "codebook_size": 16,  # small to keep cdist light
        "norm_channels": 8,  # divides 64 and 16
        "num_heads": 1,
    }


def test_vqvae_forward_shapes_and_losses():
    install_unetblocks_stub()
    from ulsd_model.models.vqvae import VectorQuantizedVAE

    cfg = make_min_config()
    model = VectorQuantizedVAE(input_channels=3, VQVAE=cfg)

    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    recon, quant, losses = model(x)

    # Reconstruction matches input shape
    assert recon.shape == x.shape

    # Quantized latent has configured channels and expected spatial size
    assert quant.shape[0] == x.shape[0]
    assert quant.shape[1] == cfg["z_channels"]

    # Effective downsamples used by encoder are first len(down_channels)-1 flags
    used_down_flags = cfg["down_sample"][: len(cfg["down_channels"]) - 1]
    factor = 1
    for f in used_down_flags:
        if f:
            factor *= 2
    assert quant.shape[2] == x.shape[2] // factor
    assert quant.shape[3] == x.shape[3] // factor

    # Losses contain expected keys and are finite tensors
    assert "codebook" in losses and "commitment" in losses
    assert isinstance(losses["codebook"], torch.Tensor)
    assert isinstance(losses["commitment"], torch.Tensor)
    assert torch.isfinite(losses["codebook"]).all()
    assert torch.isfinite(losses["commitment"]).all()


def test_vqvae_backward_pass():
    install_unetblocks_stub()
    from ulsd_model.models.vqvae import VectorQuantizedVAE

    cfg = make_min_config()
    model = VectorQuantizedVAE(input_channels=3, VQVAE=cfg)

    x = torch.randn(1, 3, 32, 32)
    recon, _, _ = model(x)
    loss = torch.nn.functional.mse_loss(recon, x)
    loss.backward()

    # Ensure some gradients flowed
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert has_grad


def test_validate_architecture_asserts_on_mismatch():
    install_unetblocks_stub()
    from ulsd_model.models.vqvae import VectorQuantizedVAE

    bad_cfg = make_min_config()
    # Break bottleneck/channel consistency
    bad_cfg["mid_channels"] = [999, 999]

    with pytest.raises(AssertionError):
        VectorQuantizedVAE(input_channels=3, VQVAE=bad_cfg)
