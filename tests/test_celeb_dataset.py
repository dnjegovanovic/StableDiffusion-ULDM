"""Tests for the CelebDataset data pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ulsd_model.datasets.CelebDataset import CelebDataset


def _build_minimal_celeb_root(base_dir: Path) -> tuple[Path, list[str]]:
    """Create a synthetic CelebA-HQ-like directory tree with one sample."""

    img_dir = base_dir / "CelebA-HQ-img"
    cap_dir = base_dir / "celeba-caption"
    mask_dir = base_dir / "CelebAMask-HQ-mask"
    img_dir.mkdir(parents=True)
    cap_dir.mkdir()
    mask_dir.mkdir()

    image_path = img_dir / "00001.jpg"
    image_array = np.full((8, 8, 3), 127, dtype=np.uint8)
    Image.fromarray(image_array).save(image_path)

    captions = ["person with hat", "smiling portrait"]
    (cap_dir / "00001.txt").write_text("\n".join(captions), encoding="utf-8")

    mask_array = np.ones((4, 4), dtype=np.uint8)
    Image.fromarray(mask_array, mode="L").save(mask_dir / "1.png")

    return base_dir, captions


def test_celeb_dataset_returns_image_and_conditioning(tmp_path: Path) -> None:
    root, captions = _build_minimal_celeb_root(tmp_path / "celeb_root")

    dataset = CelebDataset(
        split="train",
        im_path=root,
        im_size=4,
        im_channels=3,
        condition_config={
            "condition_types": ["text", "image"],
            "image_condition_config": {
                "image_condition_input_channels": 18,
                "image_condition_h": 4,
                "image_condition_w": 4,
            },
        },
    )

    assert len(dataset) == 1

    image_tensor, conditioning = dataset[0]

    assert torch.is_tensor(image_tensor)
    assert image_tensor.shape == (3, 4, 4)
    assert image_tensor.min().item() >= -1.0001
    assert image_tensor.max().item() <= 1.0001

    assert "text" in conditioning
    assert conditioning["text"] in captions

    assert "image" in conditioning
    mask_tensor = conditioning["image"]
    assert mask_tensor.shape == (18, 4, 4)
    assert mask_tensor.dtype == torch.float32
    assert torch.all((mask_tensor == 0) | (mask_tensor == 1))


def test_celeb_dataset_uses_latents_when_available(tmp_path: Path) -> None:
    root, _ = _build_minimal_celeb_root(tmp_path / "celeb_root_latent")

    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()
    latent_sample = torch.randn(3, 2, 2)
    with (latent_dir / "sample.pkl").open("wb") as handle:
        pickle.dump({"00001.jpg": [latent_sample]}, handle)

    dataset = CelebDataset(
        split="train",
        im_path=root,
        im_size=4,
        im_channels=3,
        use_latents=True,
        latent_path=latent_dir,
    )

    item = dataset[0]

    assert torch.is_tensor(item)
    assert item.shape == latent_sample.shape
    assert torch.allclose(item, latent_sample)
