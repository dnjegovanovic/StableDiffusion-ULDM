from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Dict, Iterable

from ulsd_model.utils.celeb_datautils import load_latents


class MNISTDataset(Dataset):
    """Dataset class for MNIST-style folders with optional latent caching and class conditioning."""

    def __init__(
        self,
        dataset_split: str,
        data_root: Union[str, Path],
        image_extension: str = "png",
        use_latents: bool = False,
        latent_path: Optional[Union[str, Path]] = None,
        condition_config: Optional[Dict[str, Iterable[str]]] = None,
    ) -> None:

        self.dataset_split = dataset_split
        self.image_extension = image_extension.lower()
        self.data_root = Path(data_root)

        condition_config = condition_config or {}
        self.condition_types = list(condition_config.get("condition_types", []))

        self.image_paths, self.labels = self._load_image_metadata(self.data_root)

        self.use_latents = False
        self.latent_maps: Optional[
            Union[List[torch.Tensor], Dict[Union[str, int], torch.Tensor]]
        ] = None

        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if hasattr(latent_maps, "__len__") and len(latent_maps) == len(
                self.image_paths
            ):
                self.use_latents = True
                self.latent_maps = latent_maps
                print(f"Found {len(self.latent_maps)} latents for MNIST")
            elif isinstance(latent_maps, dict) and latent_maps:
                self.use_latents = True
                self.latent_maps = latent_maps
                print(f"Found {len(self.latent_maps)} latent entries (dict) for MNIST")
            else:
                print("MNIST latents not aligned; falling back to images")

        print(f"Found {len(self.image_paths)} images for {self.dataset_split} split")

    def _load_image_metadata(self, data_root: Path) -> Tuple[List[Path], List[int]]:
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset root directory {data_root} not found")

        image_paths: List[Path] = []
        labels: List[int] = []

        extensions = {self.image_extension, "png", "jpg", "jpeg"}

        for digit_dir in tqdm(data_root.iterdir(), desc="Scanning digits"):
            if not digit_dir.is_dir():
                continue
            try:
                label = int(digit_dir.name)
            except ValueError:
                continue

            for ext in extensions:
                for image_path in digit_dir.glob(f"*.{ext}"):
                    image_paths.append(image_path)
                    labels.append(label)

        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def _resolve_latent(self, index: int) -> torch.Tensor:
        assert self.latent_maps is not None, "Latents not loaded"
        latents = self.latent_maps

        if isinstance(latents, (list, tuple)):
            return latents[index]

        if isinstance(latents, np.ndarray):
            return latents[index]

        if isinstance(latents, dict):
            img_path = self.image_paths[index]
            candidates = [
                str(img_path),
                img_path.as_posix(),
                img_path.name,
                img_path.stem,
            ]
            try:
                candidates.append(int(img_path.stem))
            except ValueError:
                pass

            for key in candidates:
                if key in latents:
                    return latents[key]

        raise KeyError(
            f"Could not find latent for MNIST image: {self.image_paths[index]}"
        )

    def __getitem__(self, index: int):
        cond_inputs: Dict[str, int] = {}
        if "class" in self.condition_types:
            cond_inputs["class"] = self.labels[index]

        if self.use_latents:
            latent = self._resolve_latent(index)
            if isinstance(latent, np.ndarray):
                latent = torch.from_numpy(latent)
            elif not torch.is_tensor(latent):
                latent = torch.tensor(latent)
            if cond_inputs:
                return latent, cond_inputs
            return latent

        image = Image.open(self.image_paths[index])
        tensor = torchvision.transforms.ToTensor()(image)
        tensor = tensor.mul(2.0).sub(1.0)

        if cond_inputs:
            return tensor, cond_inputs
        return tensor
