"""Utility helpers for CelebA-style conditioning and latent caching."""  # Provide module intent

from __future__ import annotations  # Enable future-compatible type annotations

import pickle  # Handle serialization of latent dictionaries
from pathlib import Path  # Offer path-handling helpers
from typing import Dict, MutableMapping, Union  # Type annotations for clarity

import torch  # Torch tensors used for masks and conditioning operations


def load_latents(
    latent_path: Union[str, Path]
) -> Dict[str, torch.Tensor]:  # Define latent loader with typed signature
    """Load cached latent tensors from disk for faster training starts."""  # Describe the helper

    path = Path(latent_path)  # Normalize the incoming path argument
    if not path.exists():  # Guard against missing directories
        raise FileNotFoundError(
            f"Latent directory does not exist: {path}"
        )  # Provide actionable error message

    latents: Dict[str, torch.Tensor] = {}  # Prepare container for aggregated latents
    for file_path in sorted(
        path.glob("*.pkl")
    ):  # Iterate deterministically over latent files
        with file_path.open("rb") as handle:  # Safely open the pickle file
            snapshot: MutableMapping[str, torch.Tensor] = pickle.load(
                handle
            )  # Deserialize latent dictionary
        for key, value in snapshot.items():  # Traverse each latent entry in the file
            latents[key] = value[
                0
            ]  # Store the first view for each latent key (expected tensor batch)
    return latents  # Expose combined latent map to callers


def drop_text_condition(
    text_embed: torch.Tensor,  # Batch of text embeddings to perturb
    images: torch.Tensor,  # Reference batch for device placement and size
    empty_text_embed: torch.Tensor,  # Placeholder embedding representing no conditioning
    text_drop_prob: float,  # Probability threshold for dropping entries
) -> torch.Tensor:  # Return tensor mirrors the input embeddings
    """Randomly drop text conditioning vectors with a specified probability."""  # Explain drop strategy

    if text_drop_prob <= 0.0:  # Skip work when no dropping is requested
        return text_embed  # Return conditioning unchanged

    if empty_text_embed is None:  # Ensure a fallback embedding is available
        raise ValueError(
            "empty_text_embed must be provided when dropping text conditioning"
        )  # Fail loudly with context

    drop_mask = (
        torch.rand(images.shape[0], device=images.device) < text_drop_prob
    )  # Sample Bernoulli mask per example
    text_embed[drop_mask] = empty_text_embed[
        0
    ]  # Replace dropped positions with the empty text embedding
    return text_embed  # Return possibly modified conditioning tensor


def drop_image_condition(
    image_condition: torch.Tensor,  # Image-based conditioning tensor (e.g., concatenated frames)
    images: torch.Tensor,  # Reference batch for determining mask shape
    image_drop_prob: float,  # Probability of removing conditioning information
) -> torch.Tensor:  # Returns condition tensor with random elements zeroed
    """Randomly zero out image conditioning tensors with a specified probability."""  # Document image dropout

    if image_drop_prob <= 0.0:  # Fast path when dropout disabled
        return image_condition  # Return original conditioning tensor

    drop_mask = (
        torch.rand(images.shape[0], 1, 1, 1, device=images.device) > image_drop_prob
    )  # Broadcastable mask
    return image_condition * drop_mask  # Apply mask to condition tensor


def drop_class_condition(
    class_condition: torch.Tensor,  # Class embedding tensor subject to dropout
    class_drop_prob: float,  # Drop probability for each batch example
    images: torch.Tensor,  # Reference tensor for batch sizing and device selection
) -> torch.Tensor:  # Returns class conditioning with selected rows masked out
    """Randomly drop class-conditioning vectors across the batch."""  # Describe class dropout utility

    if class_drop_prob <= 0.0:  # Do nothing when probability is zero or negative
        return class_condition  # Return input unchanged

    drop_mask = (
        torch.rand(images.shape[0], 1, device=images.device) > class_drop_prob
    )  # Sample mask per example
    return class_condition * drop_mask  # Zero-out dropped class embeddings
