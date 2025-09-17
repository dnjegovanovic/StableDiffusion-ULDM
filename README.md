# StableDiffusion-ULDM
Implementation of Uncoditional Latent Stable Diffusion model

**Overview**
- Repository includes a self-contained LPIPS implementation under `ulsd_model/models/lpips.py` used for perceptual distance.
- Tests and a small visualization script are provided to validate and explore LPIPS without needing to download weights.

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
- `tests/test_lpips.py`: unit tests for LPIPS internals and forward
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

Examples:
- No inputs (random images, no weights needed):
- `python scripts/visualize_lpips.py --dummy`
- Two images without downloads (dummy):
- `python scripts/visualize_lpips.py --img0 path/to/a.jpg --img1 path/to/b.jpg --dummy`
- With real backbone and weights available locally:
- `python scripts/visualize_lpips.py --img0 path/to/a.jpg --img1 path/to/b.jpg`

**Notes on Weights**
- The LPIPS constructor loads weights from `ulsd_model/models/weights/v0.1/vgg.pth` and a VGG16 backbone from torchvision.
- If these weights are not present and your environment is offline, use the `--dummy` flag in the script or rely on the provided tests which already stub out weight loading.

**Import Path**
- Tests add the repo root to `sys.path` to import `ulsd_model` without installing.
- If you want to import from other projects without modifying `PYTHONPATH`, consider installing this repo as a package once `setup.py` is finalized.
