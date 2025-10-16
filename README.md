# StableDiffusion-ULDM
Implementation of Uncoditional Latent Stable Diffusion model

**Overview**
- Repository includes a self-contained LPIPS implementation under `ulsd_model/models/lpips.py` used for perceptual distance.
- Tests and a small visualization script are provided to validate and explore LPIPS without needing to download weights.
- PatchGAN-style discriminator under `ulsd_model/models/discriminator.py` for adversarial training experiments, complete with unit tests for configuration and forward behaviour.

**Requirements**
- Python 3.9+
- PyTorch and torchvision compatible with your CUDA/CPU environment
- pytest for running tests

Install dependencies in your environment (example):
- `pip install torch torchvision pytest`

Note: If you plan to use the full VGG16 backbone and LPIPS weights (not required for tests/visualization in dummy mode), ensure you have:
- Torchvision VGG16 weights available locally (or internet access for the first run of `torchvision.models.vgg16(pretrained=True)`).
- LPIPS weights at `ulsd_model/models/weights/v0.1/vgg.pth`.

**Project Layout**
- `ulsd_model/models/lpips.py`: LPIPS model and helpers
- `ulsd_model/models/discriminator.py`: PatchGAN discriminator building block
- `tests/test_lpips.py`: unit tests for LPIPS internals and forward
- `tests/test_discriminator.py`: discriminator shape/configuration tests
- `scripts/visualize_lpips.py`: compute LPIPS between two images and save a heatmap

**Run Tests**
- From the repository root:
- `pytest -q`

What the tests do:
- Avoid any network/filesystem downloads by monkeypatching the VGG backbone with a deterministic dummy and skipping weight loading.
- Validate `spatial_average`, `ScalingLayer`, `NetLinLayer`, and an end-to-end LPIPS forward call.

**LPIPS Usage**
- Basic usage (may download torchvision VGG16 weights and requires local LPIPS weights file):
-
  ```python
  import torch
  from ulsd_model.models.lpips import LPIPS

  model = LPIPS()  # uses VGG16 backbone, expects LPIPS weights present
  x0 = torch.rand(1, 3, 256, 256)
  x1 = torch.rand(1, 3, 256, 256)
  d = model(x0, x1, normalize=True)  # [1,1,1,1]
  print(float(d))
  ```

If you only want to experiment locally without weights/network, use the visualization script in dummy mode (see below).

**Visualization Script**
- Compute an LPIPS distance and save a grayscale heatmap:
- `python scripts/visualize_lpips.py --dummy`

Options:
- `--img0 / --img1`: input image paths (if omitted, uses random images)
- `--size`: resize size for both images (default 256)
- `--out`: output path for the heatmap image (default `lpips_heatmap.png`)
- `--dummy`: use a dummy backbone and skip weight loading (no downloads)
 - `--auto-download`: auto-download LPIPS weights if missing
 - `--weights-url`: override the default download URL for LPIPS weights

Examples:
- No inputs (random images, no weights needed):
- `python scripts/visualize_lpips.py --dummy`
- Two images without downloads (dummy):
- `python scripts/visualize_lpips.py --img0 path/to/a.jpg --img1 path/to/b.jpg --dummy`
- With real backbone and weights available locally:
- `python scripts/visualize_lpips.py --img0 path/to/a.jpg --img1 path/to/b.jpg`

With auto-download (internet required):
- `python scripts/visualize_lpips.py --img0 a.jpg --img1 b.jpg --auto-download`

**Notes on Weights**
- The LPIPS constructor loads weights from `ulsd_model/models/weights/v0.1/vgg.pth` and a VGG16 backbone from torchvision.
- If these weights are not present and your environment is offline, use the `--dummy` flag in the script or rely on the provided tests which already stub out weight loading.
 - If online, you can let the script or the `LPIPS` class download weights automatically by passing `auto_download=True` (or the `--auto-download` flag). The default URL targets the official LPIPS repository.

**Import Path**
- Tests add the repo root to `sys.path` to import `ulsd_model` without installing.
- If you want to import from other projects without modifying `PYTHONPATH`, consider installing this repo as a package once `setup.py` is finalized.

## VQ-VAE

The repository includes a Vector-Quantized VAE implementation at `ulsd_model/models/vqvae.py` (`VectorQuantizedVAE`). It encodes images into a discrete latent grid using a learnable codebook, then decodes back to pixel space.

Key configuration (in `ulsd_model/config.yml` under `VQVAE`):
- `down_channels`: encoder channels per level (length N)
- `mid_channels`: bottleneck channels (first/last must equal `down_channels[-1]`)
- `down_sample`: encoder downsample flags (length N)
- `num_down_layers` / `num_mid_layers` / `num_up_layers`: residual/attention layers per block
- `attn_down`: attention flags for encoder levels (length N)
- `z_channels`: latent channels prior to quantization
- `codebook_size`: number of codebook entries
- `norm_channels`, `num_heads`: GroupNorm groups and attention heads

The class exposes:
- `forward(x) -> (recon, quantized, losses)`
- `encode(x) -> (quantized, losses)`
- `decode(z) -> recon`
- `encode_with_indices(x) -> (quantized, losses, indices)` for analysis/visualization

### VQ-VAE Visualization

Use `scripts/vqvae_visualize.py` to generate side-by-side original vs reconstruction and a heatmap of code indices, plus a codebook usage histogram and quantization loss report per image.

Examples:
- `python scripts/vqvae_visualize.py path/to/image.jpg --out viz_out`
- `python scripts/vqvae_visualize.py ./samples --out viz_out`
- `python scripts/vqvae_visualize.py ./samples --config ulsd_model/config.yml --device cpu`

Outputs per input image:
- `<stem>_recon.png`: reconstructed image
- `<stem>_recon_indices.png`: original | reconstruction | code index heatmap
- `<stem>_code_hist.png`: code usage histogram with approximate perplexity
- `<stem>_loss.txt`: quantizer losses (`codebook`, `commitment`)

Notes:
- The VQ-VAE depends on UNet blocks from `ddpm_model.models.UNetBlocks`. Ensure your `ddpm_model` package is installed and importable when running the visualization.
- The heatmap scales the discrete code indices to the image size using nearest-neighbor for readability.
- Extra deps for the script: `matplotlib`, `Pillow`, `PyYAML`.

### VQ-VAE Architecture Diagram

Generate a simple left-to-right architecture diagram directly from the config (no model instantiation required):

```
python scripts/vqvae_architecture_diagram.py --config ulsd_model/config.yml --out vqvae_architecture.png
```

This renders encoder down blocks, bottleneck, quantizer, and decoder up blocks, annotated with channels, attention flags, and sampling operations. The script only depends on `matplotlib` and `PyYAML`.

## Latent DDPM Training

Once the VQ-VAE has been trained, its discrete latents become the workhorse for diffusion. Rather than teaching the DDPM to denoise full-resolution pixels, we train it on the much smaller latent grid produced by the VQ-VAE encoder.

**Why go latent-first?**
- The encoder compresses MNIST frames (or any dataset configured) into a low-dimensional tensor, so the diffusion UNet sees fewer spatial locations and channels. This slashes memory and compute without sacrificing global structure.
- Quantized latents are perceptually aligned with the original images because the decoder has already learned to reconstruct faithfully. When the DDPM denoises latents well, the decoder maps them back to sharp pixels.
- Conditioning or guidance can be attached at the latent stage, keeping the pixel-space decoder frozen during diffusion training.

**Pipeline overview (all scripts live in this repo):**

```
           ┌───────────────┐        encode         ┌────────────────┐
images ───►│ train_vqvae.py├─────── latent z ─────►│train_ddpm_vqvae│
           └────┬──────────┘                       └──────┬─────────┘
                │ save reconstructions/ckpt                │ predicts noise in latent space
                ▼                                           ▼
         VQ-VAE checkpoint                          Diffusion checkpoint
                │                                           │
      decode samples during                           generate latent paths → decode
        inference/sampling                                 with frozen VQ-VAE
```

1. **Train the VQ-VAE** (`python train_vqvae.py --config ulsd_model/config.yml`). This learns the encoder/decoder pair and saves reconstructions plus the `vqvae_autoencoder_ckpt.pth` checkpoint.
2. **(Optional) Cache latents.** When `train_ddpm_vqvae.py` runs it can either encode batches on the fly or load pickled latents produced with the provided utilities.
3. **Train the DDPM in latent space** (`python train_ddpm_vqvae.py --config ulsd_model/config.yml`). The script:
   - loads the frozen VQ-VAE checkpoint,
   - encodes MNIST images into latents,
   - feeds those latents into a UNet that predicts the diffusion noise schedule,
   - stores DDPM checkpoints alongside the VQ-VAE artefacts.
4. **Sample** by running the DDPM to generate clean latents and decoding them with the same VQ-VAE decoder, recreating pixel images.

The combination mirrors latent diffusion models such as Stable Diffusion: the autoencoder captures high-frequency fidelity, while the diffusion model learns the global generative prior in a compact latent space.

## Discriminator

`ulsd_model/models/discriminator.py` provides a PatchGAN-style discriminator intended to pair with latent decoders. It stacks configurable convolutional blocks that progressively downsample inputs and outputs a grid of logits describing the real/fake likelihood of each patch.

Key traits:
- Bias terms appear only on the input and output convolutions, matching common PatchGAN practice.
- Intermediate blocks use `BatchNorm2d` and `LeakyReLU(0.2)` while the final block stays linear for flexible loss integration.
- List-based hyperparameters (`kernels`, `strides`, `paddings`) are validated up front to catch mismatched lengths.

Tests in `tests/test_discriminator.py` assert the expected output spatial dimensions, enforce the bias/normalization layout, and ensure configuration mismatches raise helpful assertions.

**Default Flow (64×64 input, defaults)**

```
Input: (B, 3, 64, 64)
 │
 ├─ Conv2d(3→64, k=4, s=2, p=1) + bias
 │   └─ LeakyReLU(0.2)
 ├─ Conv2d(64→128, k=4, s=2, p=1)
 │   └─ BatchNorm2d + LeakyReLU(0.2)
 ├─ Conv2d(128→256, k=4, s=2, p=1)
 │   └─ BatchNorm2d + LeakyReLU(0.2)
 └─ Conv2d(256→1, k=4, s=1, p=1) + bias
     └─ Identity (logits map)

Output: (B, 1, 7, 7)  # one logit per image patch
```

**PatchGAN Intuition**

- Convolution-only discriminator keeps spatial structure so every output pixel sees only a local receptive field (here ~70×70 when accounting for overlap).
- During training, each cell in the logits map judges its respective patch as real/fake, encouraging the generator to match high-frequency details everywhere instead of only global statistics.

```
Input image                Patch decisions (7×7 logits)
┌───────────────┐          ┌───────┬───────┬───────┬─┐
│ ██████░░░░░░  │  ───►    │ 0.87 │ 0.91 │ 0.63 │ │
│ ██████░░░░░░  │          ├───────┼───────┼───────┤ │
│ ██ detail ██  │          │ 0.72 │ 0.95 │ 0.58 │ │
│ ░░ smooth ░░  │          ├───────┼───────┼───────┤ │
│ ░░ smooth ░░  │          │ 0.44 │ 0.51 │ 0.39 │ │
└───────────────┘          └───────┴───────┴───────┴─┘
           ↑ overlapping receptive field slides across the image
```

High scores imply the local texture looks real; low scores highlight where the generator needs improvement.
