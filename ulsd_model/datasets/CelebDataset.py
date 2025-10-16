from __future__ import annotations  # enable forward type hints on older Python
import random  # sampling captions
from pathlib import Path  # robust path handling
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union  # typing help

import numpy as np  # array ops for mask one-hot
import torch  # tensors
import torchvision  # transforms
from PIL import Image  # image I/O
from torch.utils.data import Dataset  # PyTorch dataset base
from tqdm import tqdm  # progress bar

from ulsd_model.utils.celeb_datautils import (
    load_latents,
)  # latent loader within package


class CelebDataset(Dataset):
    r"""
    CelebA-HQ dataset wrapper with optional text/image conditioning and optional latent loading.

    - Images are center-cropped + resized to `im_size` and normalized to [-1, 1].
    - Optional 'text' conditioning expects caption files in: {im_path}/celeba-caption/{stem}.txt
    - Optional 'image' conditioning expects masks in:       {im_path}/CelebAMask-HQ-mask/{id}.png
      where {id} matches the integer file stem of the image.
    - Optional latents: `load_latents(latent_path)` can return a list/array (positional) or dict (keyed).
      Dict keys may be absolute/relative paths or stems; we try several keys to resolve.

    Parameters
    ----------
    split : str
        Dataset split name (e.g., "train", "val", "test"). Not enforced here; provided for future use.
    im_path : str | Path
        Root path containing:
          - img_align_celeba/*.{png,jpg,jpeg}
          - optional celeba-caption/*.txt
          - optional img_align_celeba-mask/*.png
    im_size : int
        Square output size after resize+center-crop (default 256).
    im_channels : int
        Expected number of image channels (usually 3).
    im_ext : str | Sequence[str]
        Image extension(s) to search for. Default 'jpg'. We also always include png/jpg/jpeg for safety.
    use_latents : bool
        Whether to return latents instead of raw images (if available).
    latent_path : Optional[str | Path]
        Path passed to `load_latents` if `use_latents=True`.
    condition_config : Optional[dict]
        Configuration like:
          {
            "condition_types": ["text", "image"],
            "image_condition_config": {
               "image_condition_input_channels": 18,
               "image_condition_h": 256,
               "image_condition_w": 256
            }
          }
    """

    def __init__(
        self,
        split: str,
        im_path: Union[str, Path],
        im_size: int = 256,
        im_channels: int = 3,
        im_ext: Union[str, Sequence[str]] = "jpg",
        use_latents: bool = False,
        latent_path: Optional[Union[str, Path]] = None,
        condition_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Save constructor args
        self.split = split  # store split (not used internally)
        self.im_size = int(im_size)  # enforce int
        self.im_channels = int(im_channels)  # enforce int
        self.im_path = Path(im_path)  # normalize path type

        # Normalize extensions into a list of lowercase strings
        if isinstance(im_ext, str):  # if single string passed
            self.im_exts = [im_ext.lower()]  # convert to list
        else:
            self.im_exts = [str(e).lower() for e in im_ext]  # ensure list of str

        # Always include common image extensions unless explicitly restricted by user
        common_exts = {"png", "jpg", "jpeg"}  # default extensions
        self.search_exts = sorted(set(self.im_exts) | common_exts)  # union -> list

        # Conditioning configuration parsing
        condition_config = condition_config or {}  # default empty dict
        self.condition_types: List[str] = list(  # list of enabled condition types
            condition_config.get("condition_types", [])
        )

        # Image-mask configuration defaults
        image_cond_cfg = condition_config.get(
            "image_condition_config", {}
        )  # nested config or {}
        self.mask_channels: int = int(
            image_cond_cfg.get(  # number of classes/channels
                "image_condition_input_channels", 18
            )
        )
        self.mask_h: int = int(
            image_cond_cfg.get("image_condition_h", self.im_size)
        )  # mask H
        self.mask_w: int = int(
            image_cond_cfg.get("image_condition_w", self.im_size)
        )  # mask W

        # Label maps for masks (CelebAMask-HQ classes)
        self.idx_to_cls_map: Dict[int, str] = {}  # class index -> name
        self.cls_to_idx_map: Dict[str, int] = {}  # class name  -> index

        # Latent loading flags/holders
        self.use_latents = False  # only set True when verified
        self.latent_maps: Optional[Union[List[Any], Dict[Any, Any]]] = None  # storage

        # Build transforms once (avoid re-creating every __getitem__)
        # NOTE: Resize then CenterCrop to get a square of im_size, then ToTensor.
        # Normalize to [-1, 1] is applied later by a simple affine map for clarity & speed.
        self.image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    self.im_size
                ),  # resize shortest side to im_size
                torchvision.transforms.CenterCrop(
                    self.im_size
                ),  # center crop to (im_size, im_size)
                torchvision.transforms.ToTensor(),  # [0,1]
            ]
        )

        # Load file lists for images, captions, masks
        self.images, self.texts, self.masks = self._load_image_lists(self.im_path)

        # If requested, try to load latents and align them with images
        if use_latents and (latent_path is not None):
            # Load latents using user-provided utility
            latent_maps = load_latents(latent_path)  # may return list/ndarray/dict
            # Heuristic: if lengths match and it's indexable, assume positional alignment
            if hasattr(latent_maps, "__len__") and len(latent_maps) == len(self.images):
                self.use_latents = True  # enable latent usage
                self.latent_maps = latent_maps  # store as-is
                print(
                    f"[CelebDataset] Found {len(self.latent_maps)} latents (positional)."
                )
            else:
                # Otherwise, try dict-like access by keys; keep and resolve later in __getitem__
                if isinstance(latent_maps, dict) and len(latent_maps) > 0:
                    self.use_latents = True  # enable latent usage
                    self.latent_maps = latent_maps  # store dict
                    print(
                        f"[CelebDataset] Found {len(self.latent_maps)} latents (dict)."
                    )
                else:
                    # Could not match latents to images
                    self.use_latents = False
                    self.latent_maps = None
                    print(
                        "[CelebDataset] Latents not aligned or not found; falling back to images."
                    )

    def _load_image_lists(
        self, root: Path
    ) -> Tuple[List[Path], List[List[str]], List[Path]]:
        """
        Collect image paths (img_align_celeba/*.ext), optional caption lists, and optional mask paths.

        Returns
        -------
        images : List[Path]
            Absolute paths to image files.
        texts : List[List[str]]
            For each image, a list of caption strings (may be empty if text not enabled or missing).
        masks : List[Path]
            For each image, a corresponding mask path (empty list if image conditioning disabled).
        """
        # Validate root directory exists
        assert root.exists(), f"[CelebDataset] images path does not exist: {root}"

        # Build candidate glob patterns under img_align_celeba
        img_dir = root / "img_align_celeba"  # standard celeb-hq folder
        assert img_dir.exists(), f"[CelebDataset] missing folder: {img_dir}"

        # Gather images across allowed extensions
        images: List[Path] = []  # collected image paths
        for ext in self.search_exts:  # iterate over extensions
            images.extend(sorted(img_dir.glob(f"*.{ext}")))  # append matches

        # Fail fast if no images
        assert (
            len(images) > 0
        ), f"[CelebDataset] No images found in {img_dir} with {self.search_exts}"

        # Prepare caption/mask containers
        texts: List[List[str]] = []  # list of per-image caption lists
        masks: List[Path] = []  # list of per-image mask paths

        # If image conditioning is enabled, set up label maps for CelebAMask-HQ
        if "image" in self.condition_types:
            label_list = [
                "skin",
                "nose",
                "eye_g",
                "l_eye",
                "r_eye",
                "l_brow",
                "r_brow",
                "l_ear",
                "r_ear",
                "mouth",
                "u_lip",
                "l_lip",
                "hair",
                "hat",
                "ear_r",
                "neck_l",
                "neck",
                "cloth",
            ]
            self.idx_to_cls_map = {
                idx: name for idx, name in enumerate(label_list)
            }  # map idx->name
            self.cls_to_idx_map = {
                name: idx for idx, name in enumerate(label_list)
            }  # map name->idx
            # Validate configured channel count matches labels length
            assert self.mask_channels == len(
                label_list
            ), f"[CelebDataset] mask_channels={self.mask_channels} but label_list has {len(label_list)}"

        # Iterate over images and collect optional captions/masks
        for img_path in tqdm(images, desc="[CelebDataset] Scanning images"):
            # Always store the image path
            # (We keep it as a Path for robust handling in __getitem__)
            # Append immediately
            # (Texts and masks are appended in lockstep to retain same length as images.)
            # --- Images ---
            # img_path is already correct
            # Append to list
            # (We append texts/masks after this as needed)
            # ----------------------------------------------
            # NOTE: Align lengths across images/texts/masks
            # ----------------------------------------------
            # Prepare default entries (empty) to fill later
            caption_list: List[str] = []  # default empty captions
            mask_path: Optional[Path] = None  # default no mask

            # --- Text conditioning: read caption file if present ---
            if "text" in self.condition_types:
                # Caption file is expected at celeba-caption/{stem}.txt
                cap_file = root / "celeba-caption" / f"{img_path.stem}.txt"
                if cap_file.exists():  # guard missing captions
                    with cap_file.open("r", encoding="utf-8") as f:
                        # Collect non-empty stripped lines
                        caption_list = [line.strip() for line in f if line.strip()]
                else:
                    # Keep empty; we will assert length alignment after the loop
                    caption_list = []

            # --- Image conditioning: build mask path if expected ---
            if "image" in self.condition_types:
                # CelebAMask-HQ uses integer stems (e.g., "00001.png")
                # Convert stem to int if possible; masks are named {id}.png
                try:
                    mask_stem_int = int(img_path.stem)  # e.g., "12345" -> 12345
                    mask_path = root / "CelebAMask-HQ-mask" / f"{mask_stem_int}.png"
                except ValueError:
                    # If stem is not an integer, try direct name match (rare)
                    mask_path = root / "CelebAMask-HQ-mask" / f"{img_path.stem}.png"

            # Append aligned entries
            texts.append(caption_list)  # list of captions (possibly empty)
            masks.append(
                mask_path if mask_path is not None else Path()
            )  # empty Path if None
            # Finally append image path (after texts/masks to keep them in same order)
            # (The order doesn't matter as long as all three are appended once per loop.)
            # But we follow original request: images first is fine.
        # At this point we used "images" local list; assign to self via return

        # Post-checks: when conditioning is enabled, verify files exist/alignment holds
        if "text" in self.condition_types:
            # Ensure same number of caption entries as images
            assert len(texts) == len(
                images
            ), "[CelebDataset] Text conditioning misaligned with images."
            # Warn if some images have no captions
            missing_caps = sum(1 for caps in texts if len(caps) == 0)
            if missing_caps > 0:
                print(
                    f"[CelebDataset] WARNING: {missing_caps} images have no captions."
                )

        if "image" in self.condition_types:
            # Ensure same number of mask entries as images
            assert len(masks) == len(
                images
            ), "[CelebDataset] Image conditioning misaligned with images."
            # Count missing masks
            missing_masks = sum(1 for m in masks if (not m or not m.exists()))
            if missing_masks > 0:
                print(
                    f"[CelebDataset] WARNING: {missing_masks} masks not found in CelebAMask-HQ-mask/."
                )
            # Informative counts
        print(f"[CelebDataset] Found {len(images)} images.")
        if "image" in self.condition_types:
            print(f"[CelebDataset] Masks entries: {len(masks)}")
        if "text" in self.condition_types:
            print(f"[CelebDataset] Caption entries: {len(texts)}")

        # Return lists; keep images as Path objects
        return images, texts, masks

    def _load_mask(self, index: int) -> torch.Tensor:
        """
        Load a single mask PNG, resize to (mask_h, mask_w) with NEAREST, and one-hot encode.

        Returns
        -------
        torch.FloatTensor of shape [C, H, W] with 0/1 (float32).
        """
        # Resolve mask path for the given sample
        mask_path = self.masks[index]  # Path or empty Path
        # Defensive checks
        assert isinstance(
            mask_path, Path
        ), "[CelebDataset] Internal: mask path not a Path."
        assert mask_path.exists(), f"[CelebDataset] Mask file missing: {mask_path}"

        # Open mask image (label IDs encoded as pixel values 0..N)
        with Image.open(mask_path) as m_im:
            # Convert to single channel (L) so values are integers (0..255)
            m_im = m_im.convert("L")
            # Resize with NEAREST to preserve integer label IDs (no interpolation)
            m_im = m_im.resize((self.mask_w, self.mask_h), resample=Image.NEAREST)
            # Convert to numpy array of shape [H, W], dtype uint8
            mask_np = np.array(m_im, dtype=np.uint8)

        # Initialize one-hot base [H, W, C]
        one_hot = np.zeros(
            (self.mask_h, self.mask_w, self.mask_channels), dtype=np.float32
        )

        # In CelebAMask-HQ, class IDs are often 1..K; background = 0
        # We fill channel c at positions where mask == (c+1); background ignored
        for c in range(self.mask_channels):
            one_hot[mask_np == (c + 1), c] = 1.0

        # Convert to torch tensor [C, H, W]
        mask_tensor = torch.from_numpy(one_hot).permute(2, 0, 1).contiguous()
        return mask_tensor

    def __len__(self) -> int:
        # Dataset length equals number of discovered images
        return len(self.images)

    def _resolve_latent(self, index: int) -> Any:
        """
        Retrieve latent for given index from either positional list/array
        or dict keyed by various possible identifiers.
        """
        # Must have latents loaded
        assert self.latent_maps is not None, "[CelebDataset] Latents not loaded."
        lm = self.latent_maps

        # If it's list/array-like, assume positional mapping
        if isinstance(lm, (list, tuple)):
            return lm[index]

        try:
            import numpy as _np  # type: ignore

            if isinstance(lm, _np.ndarray):
                return lm[index]
        except Exception:
            pass  # numpy might not be importable in some environments

        # If dict-like, try resolving by several keys
        if isinstance(lm, dict):
            img_path: Path = self.images[index]
            candidates = [
                str(img_path),  # absolute/relative full path as string
                img_path.name,  # filename with extension
                img_path.stem,  # filename without extension
            ]
            # For CelebA-HQ masks often use integer stems; also try int(img.stem)
            try:
                candidates.append(int(img_path.stem))  # int key
            except ValueError:
                pass

            # Return first hit
            for key in candidates:
                if key in lm:
                    return lm[key]

        # If we reach here, we could not resolve the latent
        raise KeyError(
            f"[CelebDataset] Could not match latent for image index {index}: {self.images[index]}"
        )

    def __getitem__(self, index: int):
        """
        Return either:
          - image tensor in [-1,1], or
          - latent,
        optionally paired with conditioning dict {'text': str, 'image': mask_tensor}.
        """
        # -------------------------
        # 1) Build conditioning
        # -------------------------
        cond_inputs: Dict[str, Any] = {}  # output conditioning dict

        # Text conditioning: sample one caption if available; else omit 'text'
        if "text" in self.condition_types:
            caps = self.texts[index]  # list of captions (possibly empty)
            if len(caps) > 0:  # only sample if we have any
                cond_inputs["text"] = random.choice(caps)  # pick 1 caption
            else:
                # Do nothing; leave 'text' absent if no captions exist for this image
                pass

        # Image conditioning: load mask tensor if path exists
        if "image" in self.condition_types:
            mask_path = self.masks[index]  # Path or empty Path
            if mask_path and mask_path.exists():  # guard missing mask
                cond_inputs["image"] = self._load_mask(index)  # [C,H,W] float tensor
            else:
                # No mask for this image; you may choose to raise instead
                # For robustness, we skip adding the image cond if missing
                pass

        # -------------------------
        # 2) Return latents OR images
        # -------------------------
        if self.use_latents:
            # Resolve latent by position (list/array) or by dict key
            latent = self._resolve_latent(index)  # Any tensor-like latent
            # Return latent alone or paired with cond_inputs
            if len(cond_inputs) == 0:
                return latent
            return latent, cond_inputs

        # Otherwise, load and transform the RGB image
        img_path: Path = self.images[index]  # Path to image
        with Image.open(img_path) as im:
            im = im.convert("RGB")  # enforce 3 channels
            im_tensor = self.image_transform(im)  # [0,1] tensor, shape [C,H,W]

        # Sanity check channel count
        if im_tensor.shape[0] != self.im_channels:
            # If mismatch, you could adapt here; we force-check to catch surprises
            raise ValueError(
                f"[CelebDataset] Expected {self.im_channels} channels but got {im_tensor.shape[0]} for {img_path}"
            )

        # Normalize from [0,1] to [-1,1] by an affine transform (vectorized & fast)
        im_tensor = im_tensor.mul(2.0).sub(1.0)  # im = 2*im - 1

        # Return image alone or paired with cond_inputs
        if len(cond_inputs) == 0:
            return im_tensor
        return im_tensor, cond_inputs
