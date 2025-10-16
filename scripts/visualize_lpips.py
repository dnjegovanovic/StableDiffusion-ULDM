#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# Ensure repo root is on sys.path so `ulsd_model` can be imported without installation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ulsd_model.models.lpips as lpips


class DummyVGG(torch.nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = False):
        super().__init__()

    def forward(self, X):
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


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    t = torch.from_numpy(
        (
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            .view(img.size[1], img.size[0], 3)
            .permute(2, 0, 1)
            .float()
        )
        / 255.0
    )
    return t.unsqueeze(0)  # [1,3,H,W]


def save_grayscale_map(tensor_map, out_path):
    # tensor_map: [H, W] in [0,1]
    arr = (tensor_map.clamp(0, 1) * 255.0).byte().cpu().numpy()
    Image.fromarray(arr, mode="L").save(out_path)


def compute_lpips_map(model: lpips.LPIPS, x0: torch.Tensor, x1: torch.Tensor):
    # Replicate internal logic to get a per-pixel map
    x0_in, x1_in = model.scaling_layer(x0), model.scaling_layer(x1)
    outs0, outs1 = model.net.forward(x0_in), model.net.forward(x1_in)

    diffs = []
    for kk in range(model.L):
        f0 = torch.nn.functional.normalize(outs0[kk], dim=1)
        f1 = torch.nn.functional.normalize(outs1[kk], dim=1)
        diff2 = (f0 - f1) ** 2
        w = model.lins[kk](diff2)  # [N,1,h,w]
        diffs.append(w)

    # Sum layers and remove channel dim, keep spatial
    val_map = sum(diffs)  # [N,1,h,w]
    return val_map.squeeze(0).squeeze(0)  # [h,w]


def main():
    p = argparse.ArgumentParser(description="Visualize LPIPS distance and heatmap.")
    p.add_argument("--img0", type=Path, required=False, help="Path to first image")
    p.add_argument("--img1", type=Path, required=False, help="Path to second image")
    p.add_argument(
        "--size", type=int, default=256, help="Resize shorter side to this size"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("lpips_heatmap.png"),
        help="Output heatmap path",
    )
    p.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy backbone and skip weight loading",
    )
    p.add_argument(
        "--auto-download",
        action="store_true",
        help="Auto-download LPIPS weights if missing",
    )
    p.add_argument(
        "--weights-url",
        type=str,
        default=None,
        help="Override URL for LPIPS weights download",
    )
    args = p.parse_args()

    if args.img0 is None or args.img1 is None:
        # Generate two random images for demo
        H = W = args.size
        x0 = torch.rand(1, 3, H, W)
        x1 = torch.rand(1, 3, H, W)
    else:
        # Keep both images identical size
        size = (args.size, args.size)
        x0 = load_image(args.img0, size=size)
        x1 = load_image(args.img1, size=size)

    if args.dummy:
        # Avoid downloading weights; use deterministic dummy and skip LPIPS weights
        setattr(lpips, "vgg16", DummyVGG)
        setattr(lpips.LPIPS, "load_state_dict", lambda self, sd, strict=False: None)

    model = lpips.LPIPS(auto_download=args.auto_download, weights_url=args.weights_url)
    with torch.no_grad():
        d = model(x0, x1, normalize=True)
        print(f"LPIPS distance: {float(d.item()):.6f}")

        heat = compute_lpips_map(model, x0, x1)
        # Normalize heat to [0,1]
        heat = heat - heat.min()
        if heat.max() > 0:
            heat = heat / heat.max()
        save_grayscale_map(heat, args.out)
        print(f"Heatmap saved to: {args.out}")


if __name__ == "__main__":
    main()
