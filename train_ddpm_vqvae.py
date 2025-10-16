from __future__ import annotations  # Allow postponed evaluation of type hints

import argparse  # Parse command-line arguments for the training script
import random  # Provide RNG seeding for reproducibility
from pathlib import Path  # Handle filesystem paths in a platform-agnostic way
from typing import Any, Dict, Iterable, Optional, Tuple  # Expose common typing primitives

import numpy as np  # Supply numerical utilities and RNG seeding
import torch  # Core tensor and autograd library
import yaml  # Parse configuration files stored as YAML
from torch import Tensor  # Explicit tensor alias for type annotations
from torch.optim import Adam  # Optimizer used to train the UNet
from torch.utils.data import DataLoader  # Mini-batch loader for datasets
import torchvision.utils as vutils  # Utilities for saving image grids
from tqdm import tqdm  # Progress bar for iterative training loops

from ddpm_model.models.LinearNoiseScheduler import (
    LinearNoiseScheduler,
)  # Noise schedule for DDPM
from ulsd_model.models.unet import UNet  # Denoising network architecture
from ulsd_model.datasets.MnistDatasets import (
    MNISTDataset,
)  # MNIST dataset wrapper with latent support
from ulsd_model.models.vqvae import (
    VectorQuantizedVAE as VQVAE,
)  # Pretrained VQ-VAE used to encode latents


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Select CUDA when available


def load_config(path: Path) -> Dict[str, Any]:  # Load structured config data from disk
    """Load YAML configuration from disk."""  # Document helper purpose

    if not path.exists():  # Sanity-check config path
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )  # Raise informative error when missing
    with path.open(
        "r", encoding="utf-8"
    ) as handle:  # Open file safely with UTF-8 encoding
        return yaml.safe_load(handle)  # Parse YAML into Python dictionary


def set_seed(seed: int) -> None:  # Make training deterministic when possible
    """Seed Python and torch RNGs for reproducibility."""  # Describe purpose of function

    random.seed(seed)  # Seed Python's built-in RNG
    np.random.seed(seed)  # Seed NumPy RNG for consistency
    torch.manual_seed(seed)  # Seed PyTorch CPU RNG
    if DEVICE.type == "cuda":  # Apply additional seeding when using GPU
        torch.cuda.manual_seed_all(seed)  # Seed all CUDA devices
    torch.backends.cudnn.deterministic = True  # Enable deterministic CuDNN operations
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for reproducibility


def build_dataset(
    name: str, **dataset_kwargs: Any
) -> Iterable[Tensor]:  # Factory for dataset construction
    """Instantiate a dataset by name."""  # Clarify helper usage

    datasets = {"mnist": MNISTDataset}  # Map supported dataset names to classes
    if name not in datasets:  # Guard against unsupported names
        raise KeyError(f"Unsupported dataset '{name}'")  # Provide clear error
    return datasets[name](**dataset_kwargs)  # Instantiate and return the dataset


def ensure_dir(path: Path) -> None:  # Create directories as needed
    """Create a directory if it does not already exist."""  # Explain helper

    path.mkdir(
        parents=True, exist_ok=True
    )  # Recursively create directories without errors


def save_image_grid(
    images: torch.Tensor,  # Images expected in [0, 1]
    directory: Path,
    step: int,
    nrow: int,
    prefix: str,
) -> None:
    """Persist a tiled grid of images to disk."""

    ensure_dir(directory)
    grid = vutils.make_grid(
        images.clamp(0.0, 1.0),
        nrow=max(1, min(nrow, images.size(0))),
        padding=2,
    )
    vutils.save_image(grid, directory / f"{prefix}_{step:06d}.png")


def decode_latents_to_images(
    latents: torch.Tensor,
    vqvae: VQVAE,
) -> torch.Tensor:
    """Decode latent tensors to image space and map to [0, 1]."""

    with torch.no_grad():
        decoded = vqvae.decode(latents)
    decoded = decoded.clamp(-1.0, 1.0)
    return ((decoded + 1.0) / 2.0).detach().cpu()


def sample_diffusion_latents(
    model: UNet,
    scheduler: LinearNoiseScheduler,
    num_samples: int,
    latent_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Run the DDPM reverse process to produce latent samples."""

    was_training = model.training
    model.eval()

    with torch.no_grad():
        latents = torch.randn((num_samples, *latent_shape), device=DEVICE)
        predicted_x0: Optional[torch.Tensor] = None
        for t in range(scheduler.num_timesteps - 1, -1, -1):
            t_tensor = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)
            noise_pred = model(latents, t_tensor)
            latents, predicted_x0 = scheduler.sample_prev_timestep(latents, noise_pred, t)
        samples = predicted_x0 if predicted_x0 is not None else latents

    if was_training:
        model.train()

    return samples.detach()


def train(config: Dict[str, Any]) -> None:  # Main entry point for DDPM training
    """Train a DDPM on VQ-VAE latents for MNIST."""  # Summarize high-level behaviour

    dataset_cfg = config["dataset_params"]  # Extract dataset-related configuration
    diffusion_cfg = config[
        "diffusion_params"
    ]  # Extract diffusion schedule configuration
    unet_cfg = config["UnetParams"]  # Extract UNet architecture parameters
    train_cfg = config["train_params"]  # Extract training loop hyperparameters
    vqvae_cfg = config["VQVAE"]  # Extract VQ-VAE architecture definition

    set_seed(train_cfg.get("seed", 0))  # Seed RNGs using configured seed
    l_path = Path(train_cfg["task_name"]) / train_cfg["vqvae_latent_dir_name"]
    use_cached_latents = l_path.exists()
    dataset = build_dataset(  # Construct dataset (latents preferred when available)
        name=dataset_cfg["name"],  # Use dataset identifier from config
        dataset_split="train",  # Operate on the training split
        use_latents=use_cached_latents,  # Request latents only when cache exists
        data_root=dataset_cfg["im_path"],  # Point to dataset root directory
        latent_path=l_path if use_cached_latents else None,  # Optional latent directory
    )
    if use_cached_latents:
        print(f"Loaded cached latents from {l_path}")
    else:
        print("Latent cache not found; encoding batches on the fly.")

    data_loader = DataLoader(  # Prepare mini-batch loader
        dataset,  # Data source (latents or images)
        batch_size=train_cfg["ldm_batch_size"],  # Batch size for diffusion training
        shuffle=True,  # Shuffle batches each epoch
        num_workers=train_cfg.get("num_workers", 0),  # Optional dataloader workers
        pin_memory=DEVICE.type == "cuda",  # Pin memory when training on GPU
    )

    output_dir = Path(
        train_cfg.get("task_name_ddpm_vqvae", train_cfg["task_name"])
    )  # Directory for checkpoints and logs
    vqvae_output_dir = Path(
        train_cfg["task_name"]
    )
    ensure_dir(output_dir)  # Ensure output directory exists

    scheduler = LinearNoiseScheduler(  # Initialize noise scheduler for DDPM
        device=DEVICE,  # Ensure scheduler tensors are created on the active device
        num_timesteps=diffusion_cfg["num_timesteps"],  # Total diffusion steps
        beta_start=diffusion_cfg["beta_start"],  # Starting beta value
        beta_end=diffusion_cfg["beta_end"],  # Ending beta value
    )

    unet_params = dict(unet_cfg)  # Clone UNet configuration to avoid mutating config in-place
    unet_params["im_channels"] = vqvae_cfg["z_channels"]  # Ensure UNet sees latent channel count as input channels

    model = UNet(  # Instantiate denoising model using keyword configuration
        UnetParams=unet_params,  # Provide architecture parameters via expected keyword
    ).to(
        DEVICE
    )  # Move model to selected device
    model.train()  # Switch network to training mode

    vqvae = VQVAE(  # Instantiate the VQ-VAE for latent encoding
        input_channels=dataset_cfg["im_channels"],  # MNIST channel count (usually 1)
        VQVAE=vqvae_cfg,  # Pass architecture configuration
    ).to(
        DEVICE
    )  # Move autoencoder to chosen device
    vqvae.eval()  # Set VQ-VAE to evaluation mode to freeze behaviour
    for param in vqvae.parameters():  # Iterate over all VQ-VAE parameters
        param.requires_grad = False  # Disable gradients for the autoencoder

    vqvae_ckpt = (
        vqvae_output_dir / train_cfg["vqvae_autoencoder_ckpt_name"]
    )  # Determine checkpoint path
    if not vqvae_ckpt.exists():  # Ensure checkpoint is available
        raise FileNotFoundError(  # Inform user to train autoencoder first
            "VQ-VAE checkpoint not found. Train the autoencoder first to generate latents."
        )
    vqvae.load_state_dict(
        torch.load(vqvae_ckpt, map_location=DEVICE)
    )  # Restore VQ-VAE weights

    optimizer = Adam(
        model.parameters(), lr=train_cfg["ldm_lr"]
    )  # Configure optimizer for UNet
    criterion = torch.nn.MSELoss()  # Use mean squared error for noise prediction

    num_epochs = train_cfg["ldm_epochs"]  # Total diffusion training epochs

    dataset_returns_latents = bool(getattr(dataset, "use_latents", False))
    samples_root = output_dir / "ddpm_samples"
    recon_dir = samples_root / "recon"
    samples_dir = samples_root / "samples"
    save_every = max(
        1,
        int(
            train_cfg.get(
                "ldm_img_save_steps",
                train_cfg.get("autoencoder_img_save_steps", 100),
            )
        ),
    )
    viz_samples = max(1, int(train_cfg.get("ldm_viz_samples", 8)))
    default_rows = int(np.ceil(np.sqrt(viz_samples)))
    viz_rows = max(1, int(train_cfg.get("ldm_viz_rows", default_rows)))
    viz_rows = min(viz_rows, viz_samples)

    ensure_dir(recon_dir)
    ensure_dir(samples_dir)

    global_step = 0
    sample_index = 0
    latent_shape: Optional[Tuple[int, ...]] = None

    for epoch in range(num_epochs):  # Iterate through epochs
        epoch_losses: list[float] = []  # Track per-batch losses for logging
        for batch in tqdm(
            data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):  # Loop over batches with progress bar
            optimizer.zero_grad(set_to_none=True)  # Clear gradients efficiently

            data_tensor = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(data_tensor, dict):
                data_tensor = data_tensor.get("image", next(iter(data_tensor.values())))

            data_tensor = data_tensor.to(DEVICE, non_blocking=True).float()

            if dataset_returns_latents:
                latents = data_tensor.detach()
            else:
                with torch.no_grad():
                    latents, _ = vqvae.encode(data_tensor)
                latents = latents.detach()

            if latent_shape is None:
                latent_shape = tuple(latents.shape[1:])

            noise = torch.randn_like(
                latents
            )  # Sample target noise for training objective
            timesteps = torch.randint(  # Sample random diffusion steps per example
                low=0,
                high=diffusion_cfg["num_timesteps"],
                size=(latents.shape[0],),
                device=latents.device,
            )

            noisy_latents = scheduler.add_noise(
                latents, noise, timesteps
            )  # Diffuse latents to chosen timesteps
            noise_pred = model(noisy_latents, timesteps)  # Predict noise using the UNet

            loss = criterion(
                noise_pred, noise
            )  # Compute MSE loss between predicted and true noise
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update UNet parameters

            epoch_losses.append(loss.item())  # Record scalar loss for logging
            global_step += 1

            if global_step % save_every == 0 and latent_shape is not None:
                real_count = min(viz_samples, latents.shape[0])
                real_latents = latents[:real_count].detach()
                real_images = decode_latents_to_images(real_latents, vqvae)

                sampled_latents = sample_diffusion_latents(
                    model=model,
                    scheduler=scheduler,
                    num_samples=viz_samples,
                    latent_shape=latent_shape,
                )
                sampled_images = decode_latents_to_images(sampled_latents, vqvae)

                save_image_grid(real_images, recon_dir, sample_index, viz_rows, "recon")
                save_image_grid(sampled_images, samples_dir, sample_index, viz_rows, "sample")
                tqdm.write(
                    f"Saved diffusion samples at step {global_step} (index {sample_index})."
                )
                sample_index += 1

        mean_loss = (
            float(np.mean(epoch_losses)) if epoch_losses else 0.0
        )  # Average loss across epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Loss {mean_loss:.4f}"
        )  # Emit progress update

        torch.save(  # Persist UNet checkpoint after each epoch
            model.state_dict(),
            output_dir / train_cfg["ldm_ckpt_name"],
        )

    print("Done Training...")  # Signal completion of training loop


def parse_args() -> argparse.Namespace:  # Build CLI argument parser
    parser = argparse.ArgumentParser(
        description="Arguments for DDPM training"
    )  # Describe command-line usage
    parser.add_argument(  # Define configuration file flag
        "--config",
        dest="config_path",
        default="./ulsd_model/config.yml",
        type=str,
    )
    return parser.parse_args()  # Parse and return arguments


def main() -> None:  # Script entry point
    args = parse_args()  # Parse CLI arguments
    config = load_config(Path(args.config_path))  # Load YAML configuration
    train(config)  # Kick off training using parsed configuration


if __name__ == "__main__":  # Ensure module executed as script
    main()  # Invoke main entry point
