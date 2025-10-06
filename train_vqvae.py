"""Training script for the Vector-Quantized VAE."""  # Module docstring for high-level context

from __future__ import annotations  # Allow forward references in type hints

import argparse  # Parse CLI arguments
import random  # Seed Python's RNG for reproducibility
from pathlib import Path  # Work with filesystem paths
from typing import Any, Dict, Iterable  # Type annotations for clarity

import numpy as np  # Deterministic seeding and logging helpers
import torch  # Core deep-learning library
import torchvision.utils as vutils  # Utilities for saving image grids
import yaml  # Configuration loader
from torch import Tensor  # Tensor type alias
from torch.optim import Adam  # Optimizer class
from torch.utils.data import DataLoader  # Data loading utility
from tqdm import tqdm  # Progress bar for training loop

from ulsd_model.datasets.MnistDatasets import MNISTDataset  # MNIST dataset wrapper
from ulsd_model.models.discriminator import Discriminator  # PatchGAN discriminator
from ulsd_model.models.lpips import LPIPS  # Perceptual loss model
from ulsd_model.models.vqvae import VectorQuantizedVAE as VQVAE  # VQ-VAE model

# Select the default computation device (CUDA if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Global device handle


def load_config(path: Path) -> Dict[str, Any]:  # Load YAML configuration into a dictionary
    """Load and validate the training configuration from disk."""  # Docstring describing the helper

    if not path.exists():  # Ensure the configuration file is present
        raise FileNotFoundError(f"Config file not found: {path}")  # Provide actionable error message

    with path.open("r", encoding="utf-8") as handle:  # Open the YAML file safely
        try:  # Guard against malformed YAML
            return yaml.safe_load(handle)  # Parse and return the configuration dictionary
        except yaml.YAMLError as error:  # Catch YAML parsing errors
            raise ValueError(f"Failed to parse config: {error}") from error  # Re-raise with context


def set_seed(seed: int) -> None:  # Ensure reproducibility across libraries
    """Seed Python, NumPy, and PyTorch for deterministic behaviour where possible."""  # Describe the helper

    random.seed(seed)  # Seed Python's built-in RNG
    np.random.seed(seed)  # Seed NumPy's global RNG
    torch.manual_seed(seed)  # Seed CPU torch RNG
    if DEVICE.type == "cuda":  # Additional seeding when CUDA is available
        torch.cuda.manual_seed_all(seed)  # Seed all CUDA devices for determinism
    torch.backends.cudnn.deterministic = True  # Encourage deterministic convolutions where possible
    torch.backends.cudnn.benchmark = False  # Disable autotuner for determinism


def build_dataset(name: str, **dataset_kwargs: Any) -> Iterable[Tensor]:  # Factory for datasets
    """Instantiate the dataset indicated by name and configuration."""  # Helper description

    dataset_map = {"mnist": MNISTDataset}  # Map string identifiers to dataset classes
    if name not in dataset_map:  # Validate dataset availability
        raise KeyError(f"Unsupported dataset '{name}'. Available: {list(dataset_map)}")  # Inform user about options
    dataset_cls = dataset_map[name]  # Retrieve matching dataset class
    return dataset_cls(**dataset_kwargs)  # Instantiate and return the dataset


def harmonise_vqvae_config(cfg: Dict[str, Any]) -> Dict[str, Any]:  # Ensure config lists match model expectations
    """Pad boolean configuration lists so they match the length of channel definitions."""  # Helper explanation

    cfg = dict(cfg)  # Shallow copy to avoid mutating caller's dictionary
    down_channels = cfg.get("down_channels", [])  # Fetch encoder channel schedule
    target_length = len(down_channels)  # Expected length for associated boolean lists

    for key in ("down_sample", "attn_down"):  # Iterate over list-valued keys needing harmonisation
        values = list(cfg.get(key, []))  # Copy existing list (or empty if missing)
        if len(values) == target_length - 1:  # Allow legacy configs that omit final level flag
            values.append(False)  # Default to no-op (no downsample / no attention) on final level
        if len(values) != target_length:  # Validate final length after adjustment
            raise ValueError(
                f"Config key '{key}' must have length {target_length}, received {len(values)}"
            )
        cfg[key] = values  # Store harmonised list back into config copy

    return cfg  # Return padded/validated configuration


def ensure_dir(path: Path) -> None:  # Lightweight mkdir helper
    """Create directory if it does not already exist."""  # Docstring for helper

    path.mkdir(parents=True, exist_ok=True)  # Create the directory tree safely


def save_image_grid(inputs: Tensor, recons: Tensor, out_dir: Path, step: int) -> None:  # Save input/recon grids
    """Persist a visualization comparing inputs and reconstructions."""  # Helper description

    ensure_dir(out_dir)  # Guarantee target directory exists
    combined = torch.cat([inputs, recons], dim=0)  # Stack original and reconstructed batches
    grid = vutils.make_grid(combined, nrow=inputs.size(0))  # Arrange images into a single grid tensor
    output_path = out_dir / f"current_autoencoder_sample_{step}.png"  # Build deterministic filename
    vutils.save_image(grid, output_path)  # Write the grid to disk


def accumulate(loss: Tensor, factor: int) -> Tensor:  # Utility for gradient accumulation scaling
    """Scale losses for gradient accumulation without modifying the original tensor."""  # Helper docstring

    return loss / factor  # Return scaled tensor


def train(config: Dict[str, Any]) -> None:  # Main training routine
    """Train the VQ-VAE (with optional adversarial regularisation) using the provided config."""  # Docstring summarising behaviour

    dataset_cfg = config["dataset_params"]  # Extract dataset configuration block
    autoencoder_cfg = harmonise_vqvae_config(config["VQVAE"])  # Extract and normalise autoencoder params
    train_cfg = config["train_params"]  # Extract optimisation and logging parameters

    set_seed(train_cfg.get("seed", 0))  # Seed RNGs for reproducibility

    model = VQVAE(input_channels=dataset_cfg["im_channels"], VQVAE=autoencoder_cfg)  # Instantiate VQ-VAE model
    model.to(DEVICE)  # Move model parameters to the selected device
    model.train()  # Switch to training mode (enables dropout/batchnorm behaviour)

    discriminator = Discriminator(im_channels=dataset_cfg["im_channels"])  # Instantiate PatchGAN discriminator
    discriminator.to(DEVICE)  # Move discriminator to device
    discriminator.train()  # Enable training behaviour for discriminator

    lpips_model = LPIPS().to(DEVICE)  # Load LPIPS network for perceptual loss
    lpips_model.eval()  # Keep LPIPS frozen as perceptual feature extractor

    dataset = build_dataset(  # Build the dataset according to configuration
        name=dataset_cfg["name"],  # Dataset identifier (e.g. mnist)
        dataset_split="train",  # Use training split
        data_root=dataset_cfg["im_path"],  # Path to the images
        #im_size=dataset_cfg["im_size"],  # Resize target
        #im_channels=dataset_cfg["im_channels"],  # Image channel count
    )

    data_loader = DataLoader(  # Prepare PyTorch DataLoader
        dataset,
        batch_size=train_cfg["autoencoder_batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=DEVICE.type == "cuda",
    )

    output_dir = Path(train_cfg["task_name"])  # Directory where checkpoints/visuals are saved
    ensure_dir(output_dir)  # Ensure output directory exists

    sample_dir = output_dir / "vqvae_autoencoder_samples"  # Subdirectory for reconstructions

    recon_loss_fn = torch.nn.MSELoss()  # Reconstruction loss (L2)
    adv_loss_fn = torch.nn.MSELoss()  # Least-squares GAN loss

    optimizer_g = Adam(model.parameters(), lr=train_cfg["autoencoder_lr"], betas=(0.5, 0.999))  # Generator optimizer
    optimizer_d = Adam(discriminator.parameters(), lr=train_cfg["autoencoder_lr"], betas=(0.5, 0.999))  # Discriminator optimizer

    accumulation_steps = max(1, train_cfg.get("autoencoder_acc_steps", 1))  # Gradient accumulation factor
    save_every = max(1, train_cfg.get("autoencoder_img_save_steps", 100))  # Frequency for saving images
    disc_start = train_cfg.get("disc_start", 0)  # Delay before adversarial training kicks in

    codebook_weight = train_cfg.get("codebook_weight", 1.0)  # Weighting for codebook loss
    commitment_beta = train_cfg.get("commitment_beta", 0.25)  # Weighting for commitment loss
    perceptual_weight = train_cfg.get("perceptual_weight", 1.0)  # Weight for LPIPS perceptual loss
    disc_weight = train_cfg.get("disc_weight", 1.0)  # Weight for adversarial loss

    total_epochs = train_cfg["autoencoder_epochs"]  # Number of training epochs

    global_step = 0  # Track total optimisation steps for scheduling/logging
    sample_index = 0  # Counter for saved reconstruction grids

    for epoch in range(total_epochs):  # Epoch-wise training loop
        epoch_recon, epoch_codebook, epoch_perceptual = [], [], []  # Accumulate generator metrics
        epoch_adv_fake, epoch_disc = [], []  # Accumulate adversarial generator/discriminator losses

        optimizer_g.zero_grad(set_to_none=True)  # Reset generator gradients before epoch
        optimizer_d.zero_grad(set_to_none=True)  # Reset discriminator gradients before epoch

        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{total_epochs}"):  # Iterate over batches with progress bar
            global_step += 1  # Increment global step counter
            images = batch.float().to(DEVICE, non_blocking=True)  # Move current batch to device

            recon, quantized, quant_losses = model(images)  # Forward pass through VQ-VAE (returns recon + losses)

            if global_step == 1 or global_step % save_every == 0:  # Conditional visualization
                with torch.no_grad():  # Disable grad computations for logging
                    sample_count = min(8, images.size(0))  # Limit number of images shown
                    inputs_vis = (images[:sample_count].detach().cpu() + 1) / 2  # Rescale inputs to [0,1]
                    recons_vis = (recon[:sample_count].detach().cpu().clamp(-1, 1) + 1) / 2  # Clamp and rescale reconstructions
                    save_image_grid(inputs_vis, recons_vis, sample_dir, sample_index)  # Persist visualization grid
                    sample_index += 1  # Increment saved grid counter

            recon_loss = recon_loss_fn(recon, images)  # Compute reconstruction MSE
            codebook_loss = codebook_weight * quant_losses["codebook"]  # Weighted codebook loss
            commitment_loss = commitment_beta * quant_losses["commitment"]  # Weighted commitment loss

            gen_loss = accumulate(recon_loss, accumulation_steps)  # Start generator loss with scaled reconstruction term
            gen_loss = gen_loss + accumulate(codebook_loss, accumulation_steps)  # Add codebook component
            gen_loss = gen_loss + accumulate(commitment_loss, accumulation_steps)  # Add commitment component

            epoch_recon.append(recon_loss.item())  # Track raw reconstruction loss
            epoch_codebook.append(codebook_loss.item())  # Track raw codebook contribution

            with torch.no_grad():  # Freeze LPIPS model gradients
                lpips_value = lpips_model(recon, images).mean()  # Compute mean perceptual distance
            perceptual_term = perceptual_weight * lpips_value  # Apply weighting
            gen_loss = gen_loss + accumulate(perceptual_term, accumulation_steps)  # Add perceptual loss to generator objective
            epoch_perceptual.append(perceptual_term.item())  # Log perceptual contribution

            if global_step > disc_start:  # Adversarial loss only after specified warm-up
                fake_pred = discriminator(recon)  # Predict realism of reconstructed images
                adv_target = torch.ones_like(fake_pred, device=DEVICE)  # Target label (real) for generator
                adv_loss = adv_loss_fn(fake_pred, adv_target)  # Least-squares adversarial loss
                epoch_adv_fake.append((disc_weight * adv_loss).item())  # Log weighted adversarial contribution
                gen_loss = gen_loss + accumulate(disc_weight * adv_loss, accumulation_steps)  # Include adversarial term

            gen_loss.backward()  # Backprop generator objective

            if global_step % accumulation_steps == 0:  # Apply optimizer step when accumulation window completes
                optimizer_g.step()  # Update generator parameters
                optimizer_g.zero_grad(set_to_none=True)  # Reset gradients for next accumulation window

            if global_step > disc_start:  # Update discriminator only after warm-up
                real_pred = discriminator(images.detach())  # Discriminator prediction for real images
                fake_pred_detached = discriminator(recon.detach())  # Prediction for detached reconstructions

                real_labels = torch.ones_like(real_pred, device=DEVICE)  # Real target tensor
                fake_labels = torch.zeros_like(fake_pred_detached, device=DEVICE)  # Fake target tensor

                real_loss = adv_loss_fn(real_pred, real_labels)  # Discriminator loss on real samples
                fake_loss = adv_loss_fn(fake_pred_detached, fake_labels)  # Discriminator loss on fake samples

                disc_loss = disc_weight * 0.5 * (real_loss + fake_loss)  # Combine losses with weighting and average
                epoch_disc.append(disc_loss.item())  # Log discriminator loss

                accumulate(disc_loss, accumulation_steps).backward()  # Backprop scaled discriminator loss

                if global_step % accumulation_steps == 0:  # Apply discriminator update on accumulation boundary
                    optimizer_d.step()  # Update discriminator parameters
                    optimizer_d.zero_grad(set_to_none=True)  # Reset discriminator gradients

        if global_step % accumulation_steps != 0:  # Flush pending generator gradients at epoch end
            optimizer_g.step()  # Update parameters for incomplete accumulation window
            optimizer_g.zero_grad(set_to_none=True)  # Reset gradients

        if global_step % accumulation_steps != 0 and epoch_disc:  # Flush discriminator grads if needed and applicable
            optimizer_d.step()  # Update discriminator parameters
            optimizer_d.zero_grad(set_to_none=True)  # Reset gradients

        recon_mean = float(np.mean(epoch_recon)) if epoch_recon else 0.0  # Compute epoch-level reconstruction loss
        perceptual_mean = float(np.mean(epoch_perceptual)) if epoch_perceptual else 0.0  # Epoch perceptual loss
        codebook_mean = float(np.mean(epoch_codebook)) if epoch_codebook else 0.0  # Epoch codebook loss
        adv_mean = float(np.mean(epoch_adv_fake)) if epoch_adv_fake else 0.0  # Epoch adversarial generator loss
        disc_mean = float(np.mean(epoch_disc)) if epoch_disc else 0.0  # Epoch discriminator loss

        if epoch_disc:  # Detailed logging when adversarial branch active
            print(
                f"Epoch {epoch + 1}/{total_epochs} | Recon {recon_mean:.4f} | Perceptual {perceptual_mean:.4f} "
                f"| Codebook {codebook_mean:.4f} | G-Adv {adv_mean:.4f} | D {disc_mean:.4f}"
            )
        else:  # Log only generator-specific metrics during warm-up
            print(
                f"Epoch {epoch + 1}/{total_epochs} | Recon {recon_mean:.4f} | Perceptual {perceptual_mean:.4f} "
                f"| Codebook {codebook_mean:.4f}"
            )

        torch.save(model.state_dict(), output_dir / train_cfg["vqvae_autoencoder_ckpt_name"])  # Persist VQ-VAE weights
        torch.save(discriminator.state_dict(), output_dir / train_cfg["vqvae_discriminator_ckpt_name"])  # Persist discriminator weights

    print("Done Training...")  # Indicate completion to the user


def parse_args() -> argparse.Namespace:  # Command-line interface builder
    """Parse command-line flags for the training script."""  # Docstring for CLI helper

    parser = argparse.ArgumentParser(description="Arguments for VQ-VAE training")  # Create parser with description
    parser.add_argument("--config", dest="config_path", default="./ulsd_model/config.yml", type=str)  # Add config flag
    return parser.parse_args()  # Parse and return namespace


def main() -> None:  # Entrypoint wrapper
    """CLI entrypoint for training the VQ-VAE."""  # Docstring for main function

    args = parse_args()  # Parse CLI arguments
    config = load_config(Path(args.config_path))  # Load configuration from disk
    train(config)  # Begin training using loaded configuration


if __name__ == "__main__":  # Script entry guard
    main()  # Invoke main when executed directly
