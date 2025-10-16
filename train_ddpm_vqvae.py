from __future__ import annotations  # Allow postponed evaluation of type hints

import argparse  # Parse command-line arguments for the training script
import random  # Provide RNG seeding for reproducibility
from pathlib import Path  # Handle filesystem paths in a platform-agnostic way
from typing import Any, Dict, Iterable  # Expose common typing primitives

import numpy as np  # Supply numerical utilities and RNG seeding
import torch  # Core tensor and autograd library
import yaml  # Parse configuration files stored as YAML
from torch import Tensor  # Explicit tensor alias for type annotations
from torch.optim import Adam  # Optimizer used to train the UNet
from torch.utils.data import DataLoader  # Mini-batch loader for datasets
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
    l_path = (
        Path(train_cfg["task_name"]) / train_cfg["vqvae_autoencoder_ckpt_name"]
    )  # Build path to stored latents
    dataset = build_dataset(  # Construct dataset (latents preferred when available)
        name=dataset_cfg["name"],  # Use dataset identifier from config
        dataset_split="train",  # Operate on the training split
        use_latents=True,  # Request latents instead of raw images when present
        data_root=dataset_cfg["im_path"],  # Point to dataset root directory
        latent_path=l_path,  # Provide path where latent pickles reside
    )

    data_loader = DataLoader(  # Prepare mini-batch loader
        dataset,  # Data source (latents or images)
        batch_size=train_cfg["ldm_batch_size"],  # Batch size for diffusion training
        shuffle=True,  # Shuffle batches each epoch
        num_workers=train_cfg.get("num_workers", 0),  # Optional dataloader workers
        pin_memory=DEVICE.type == "cuda",  # Pin memory when training on GPU
    )

    output_dir = Path(
        train_cfg["task_name_ddpm_vqvae"]
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

    for epoch in range(num_epochs):  # Iterate through epochs
        epoch_losses: list[float] = []  # Track per-batch losses for logging
        for batch in tqdm(
            data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):  # Loop over batches with progress bar
            optimizer.zero_grad(set_to_none=True)  # Clear gradients efficiently

            images = batch.float().to(
                DEVICE, non_blocking=True
            )  # Move latent batch to device
            with torch.no_grad():  # Stop gradients through VQ-VAE
                latents, _ = vqvae.encode(
                    images
                )  # Encode MNIST batch into quantized latents
                latents = latents.detach()  # Detach latents from graph for safety

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
