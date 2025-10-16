#!/usr/bin/env python3
"""
Generate a simple architecture diagram for the VQ-VAE from the YAML config.

This script does not instantiate the model or require ddpm_model. It reads the
VQVAE section and renders a left-to-right diagram showing encoder blocks,
quantizer, and decoder blocks with key annotations (channels, attention, (down/up)sampling).
"""

import argparse
from pathlib import Path
from typing import List

import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_args():
    ap = argparse.ArgumentParser(
        description="Render VQ-VAE architecture diagram from config"
    )
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "ulsd_model" / "config.yml"),
        help="Path to YAML config containing VQVAE section",
    )
    ap.add_argument(
        "--out", default="vqvae_architecture.png", help="Output image path (PNG)"
    )
    ap.add_argument(
        "--input-channels",
        type=int,
        default=3,
        help="Input image channels (for labeling)",
    )
    return ap.parse_args()


def load_vq_cfg(cfg_path: Path):
    with open(cfg_path, "r") as f:
        cfg_all = yaml.safe_load(f)
    if "VQVAE" not in cfg_all:
        raise ValueError("Config YAML missing 'VQVAE' section")
    return cfg_all["VQVAE"]


class Canvas:
    def __init__(self, total_nodes: int):
        # Compute width based on number of nodes to ensure everything is visible
        step = 1.6
        margin = 1.0
        width = margin * 2 + total_nodes * step
        figsize = (max(8, width), 4)

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.axis("off")
        self.step = step
        self.margin = margin
        self.x = margin  # current x position (left margin)
        self.y = 0.5

        # Set limits so artists outside [0,1] are shown
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, 1)

    def box(self, label: str, color="#89CFF0"):
        w, h = 1.2, 0.8
        left = self.x - w / 2
        right = self.x + w / 2
        rect = Rectangle(
            (left, self.y - h / 2),
            w,
            h,
            linewidth=1.0,
            edgecolor="#222",
            facecolor=color,
        )
        self.ax.add_patch(rect)
        self.ax.text(self.x, self.y, label, ha="center", va="center", fontsize=9)
        return left, right, self.y

    def arrow_to_next(self, x0, y0, x1, y1):
        self.ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#222"),
        )

    def advance(self):
        self.x += self.step


def render_arch(vq_cfg: dict, input_channels: int, out_path: Path):
    down_ch: List[int] = list(vq_cfg["down_channels"])  # length N
    mid_ch: List[int] = list(vq_cfg["mid_channels"])  # length M
    down_flags: List[bool] = list(vq_cfg["down_sample"])  # length N
    attn_down: List[bool] = list(vq_cfg["attn_down"])  # length N
    zc = int(vq_cfg["z_channels"])  # latent channels
    codebook = int(vq_cfg["codebook_size"])  # K
    n_down_layers = int(vq_cfg["num_down_layers"]) if "num_down_layers" in vq_cfg else 1
    n_mid_layers = int(vq_cfg["num_mid_layers"]) if "num_mid_layers" in vq_cfg else 1
    n_up_layers = int(vq_cfg["num_up_layers"]) if "num_up_layers" in vq_cfg else 1

    # Count nodes to size the canvas
    n_nodes = (
        1  # Input
        + 1  # initial conv
        + (len(down_ch) - 1)  # down blocks
        + (len(mid_ch) - 1)  # bottleneck enc
        + 1  # GN+SiLU+Conv3x3 to z
        + 1  # PreQuant 1x1
        + 1  # Quantize
        + 1  # PostQuant 1x1
        + 1  # Conv3x3 z->mid[-1]
        + (len(mid_ch) - 1)  # bottleneck dec
        + (len(down_ch) - 1)  # up blocks
        + 1  # output head
    )
    C = Canvas(total_nodes=n_nodes)

    # Input
    prev_left, prev_right, prev_y = C.box(f"Input\nC={input_channels}")
    C.advance()
    cur_left, cur_right, cur_y = C.box(f"Conv3x3\n{input_channels}->{down_ch[0]}")
    C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)

    # Encoder down blocks
    prev_right, prev_y = cur_right, cur_y
    for i in range(len(down_ch) - 1):
        C.advance()
        label = [
            f"DownBlock {i}",
            f"{down_ch[i]}->{down_ch[i+1]}",
            f"L={n_down_layers}",
        ]
        if down_flags[i]:
            label.append("downsample")
        if attn_down[i]:
            label.append("attn")
        cur_left, cur_right, cur_y = C.box("\n".join(label), color="#B0E0E6")
        C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)
        prev_right, prev_y = cur_right, cur_y

    # Bottleneck blocks (encoder side)
    for j in range(len(mid_ch) - 1):
        C.advance()
        cur_left, cur_right, cur_y = C.box(
            "\n".join(
                [f"Bottleneck {j}", f"{mid_ch[j]}->{mid_ch[j+1]}", f"L={n_mid_layers}"]
            ),
            color="#AFE1AF",
        )
        C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)
        prev_right, prev_y = cur_right, cur_y

    # Encoder head to latents
    C.advance()
    cur_left, cur_right, cur_y = C.box(f"GN+SiLU+Conv3x3\n{down_ch[-1]}->{zc}")
    C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)
    prev_right, prev_y = cur_right, cur_y

    C.advance()
    cur_left, cur_right, cur_y = C.box("PreQuant Conv1x1\n{zc}->{zc}".format(zc=zc))
    C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)
    prev_right, prev_y = cur_right, cur_y

    # Quantizer
    C.advance()
    cur_left, cur_right, cur_y = C.box(
        f"Quantize\nK={codebook}\nC={zc}", color="#F9D5E5"
    )
    C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)
    prev_right, prev_y = cur_right, cur_y

    # Decoder head
    C.advance()
    cur_left, cur_right, cur_y = C.box("PostQuant Conv1x1\n{zc}->{zc}".format(zc=zc))
    C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)
    prev_right, prev_y = cur_right, cur_y

    C.advance()
    cur_left, cur_right, cur_y = C.box(f"Conv3x3\n{zc}->{mid_ch[-1]}")
    C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)
    prev_right, prev_y = cur_right, cur_y

    # Bottleneck blocks (decoder side, reversed)
    for j in range(len(mid_ch) - 1, 0, -1):
        C.advance()
        cur_left, cur_right, cur_y = C.box(
            "\n".join(
                [
                    f"Bottleneck {j-1}",
                    f"{mid_ch[j]}->{mid_ch[j-1]}",
                    f"L={n_mid_layers}",
                ]
            ),
            color="#AFE1AF",
        )
        C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)
        prev_right, prev_y = cur_right, cur_y

    # Up blocks (decoder)
    for i in range(len(down_ch) - 1, 0, -1):
        C.advance()
        label = [
            f"UpBlock {i-1}",
            f"{down_ch[i]}->{down_ch[i-1]}",
            f"L={n_up_layers}",
        ]
        # upsample flag mirrors encoder down flag at i-1
        if down_flags[i - 1]:
            label.append("upsample")
        if attn_down[i - 1]:
            label.append("attn")
        cur_left, cur_right, cur_y = C.box("\n".join(label), color="#B0E0E6")
        C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)
        prev_right, prev_y = cur_right, cur_y

    # Output head
    C.advance()
    cur_left, cur_right, cur_y = C.box(
        f"GN+SiLU+Conv3x3\n{down_ch[0]}->{input_channels}"
    )
    C.arrow_to_next(prev_right, prev_y, cur_left, cur_y)

    C.fig.tight_layout()
    C.fig.savefig(out_path, dpi=160)
    plt.close(C.fig)


def main():
    args = parse_args()
    vq_cfg = load_vq_cfg(Path(args.config))
    render_arch(vq_cfg, args.input_channels, Path(args.out))
    print(f"Saved architecture diagram to {args.out}")


if __name__ == "__main__":
    main()
