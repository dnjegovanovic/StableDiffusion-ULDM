#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_image_paths(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for p in inputs:
        pth = Path(p)
        if pth.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                paths.extend(sorted(pth.glob(ext)))
        elif any(ch in p for ch in "*?["):
            paths.extend([Path(s) for s in sorted(Path().glob(p))])
        else:
            paths.append(pth)
    paths = [p for p in paths if p.exists()]
    return paths


def pil_to_tensor(img: Image.Image, max_side: int = 512) -> torch.Tensor:
    # Resize keeping aspect ratio if needed
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
    arr = np.asarray(img.convert("RGB")) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).float()
    return ten


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0, 1)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def make_grid_visual(
    original: torch.Tensor, recon: torch.Tensor, indices: torch.Tensor
) -> Image.Image:
    # original/recon: [3,H,W] in [0,1]; indices: [H',W']
    H, W = original.shape[-2:]
    Hq, Wq = indices.shape
    # Upsample indices to image size for display
    idx_img = torch.from_numpy(indices.cpu().numpy()).float().unsqueeze(0).unsqueeze(0)
    idx_up = F.interpolate(idx_img, size=(H, W), mode="nearest").squeeze()
    idx_up_np = idx_up.numpy()
    # Normalize for colormap
    vmax = max(idx_up_np.max(), 1.0)
    vmin = idx_up_np.min()
    normed = (idx_up_np - vmin) / (vmax - vmin + 1e-8)
    cmap = plt.get_cmap("viridis")
    colored = (cmap(normed)[..., :3] * 255).astype(np.uint8)
    idx_pil = Image.fromarray(colored)

    o_pil = tensor_to_pil(original)
    r_pil = tensor_to_pil(recon)

    # Concatenate side by side
    pad = 8
    total_w = o_pil.width + r_pil.width + idx_pil.width + 2 * pad
    total_h = max(o_pil.height, r_pil.height, idx_pil.height)
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    x = 0
    for im in (o_pil, r_pil, idx_pil):
        y = (total_h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + pad
    return canvas


def plot_codebook_hist(indices: torch.Tensor, codebook_size: int, out_path: Path):
    flat = indices.flatten().cpu().numpy().astype(np.int64)
    hist, _ = np.histogram(flat, bins=np.arange(codebook_size + 1))
    p = hist / max(hist.sum(), 1)
    # entropy/perplexity
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz + 1e-12)).sum())
    perplexity = float(np.exp(entropy))

    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(codebook_size), hist, width=1.0)
    plt.title(f"Codebook usage (perplexity≈{perplexity:.2f})")
    plt.xlabel("Code index")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_model_from_config(config_path: Path, device: str):
    from ulsd_model.models.vqvae import VectorQuantizedVAE

    with open(config_path, "r") as f:
        cfg_all = yaml.safe_load(f)
    if "VQVAE" not in cfg_all:
        raise ValueError("Config YAML missing 'VQVAE' section")
    vq_cfg = cfg_all["VQVAE"]

    model = VectorQuantizedVAE(input_channels=3, VQVAE=vq_cfg)
    model.eval().to(device)
    return model, vq_cfg


def visualize_batch(
    model, vq_cfg, images: List[Path], out_dir: Path, device: str, max_side: int
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in images:
        img = Image.open(p).convert("RGB")
        x = pil_to_tensor(img, max_side=max_side).unsqueeze(0).to(device)
        with torch.no_grad():
            q, losses, indices = model.encode_with_indices(x)
            recon = model.decode(q)

        # Compose visualization
        vis = make_grid_visual(x[0].cpu(), recon[0].cpu(), indices[0].cpu())
        vis_path = out_dir / f"{p.stem}_recon_indices.png"
        vis.save(vis_path)

        # Save histogram
        hist_path = out_dir / f"{p.stem}_code_hist.png"
        plot_codebook_hist(
            indices[0].cpu(), int(vq_cfg.get("codebook_size", 0)), hist_path
        )

        # Optionally save raw reconstruction
        recon_path = out_dir / f"{p.stem}_recon.png"
        tensor_to_pil(recon[0].cpu()).save(recon_path)

        # Write a brief txt with losses
        with open(out_dir / f"{p.stem}_loss.txt", "w") as f:
            f.write(f"codebook_loss: {float(losses['codebook']):.6f}\n")
            f.write(f"commitment_loss: {float(losses['commitment']):.6f}\n")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize VQ-VAE reconstructions and codebook usage"
    )
    ap.add_argument(
        "inputs", nargs="+", help="Image files, directories, or globs to visualize"
    )
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "ulsd_model" / "config.yml"),
        help="Path to YAML config containing VQVAE section",
    )
    ap.add_argument(
        "--out", default="viz_out", help="Output directory for visualizations"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--max-side", type=int, default=512, help="Resize images so max side <= this"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    image_paths = load_image_paths(args.inputs)
    if not image_paths:
        raise SystemExit("No input images found. Provide files/dirs/globs.")

    model, vq_cfg = build_model_from_config(Path(args.config), args.device)
    visualize_batch(
        model, vq_cfg, image_paths, Path(args.out), args.device, args.max_side
    )
    print(f"Saved visualizations to {args.out}")


if __name__ == "__main__":
    main()
