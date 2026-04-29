from __future__ import annotations

from collections import OrderedDict
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .features import FeatureConfig, compute_traditional_features, uses_gpu_features
from .patches import PatchCoord, compute_patch_grid, extract_patch

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class PatchRecord:
    pair: ImageMaskPair
    coord: PatchCoord
    category: str


def _resolve_file(directory: Path, token: str) -> Path:
    token = token.strip()
    candidate = directory / token
    if candidate.suffix and candidate.exists():
        return candidate
    stem = Path(token).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {token!r} in {directory}")


def _split_file_name(config: dict[str, Any], split: str) -> str:
    key = f"{split}_split_file"
    if key in config:
        return str(config[key])
    if split == "train":
        return str(config.get("train_split_file", "train_subset_1.txt"))
    return f"{split}.txt"


def find_pairs(root: str | Path, image_dir: str, mask_dir: str, split: str, data_cfg: dict[str, Any] | None = None) -> list[ImageMaskPair]:
    root = Path(root)
    data_cfg = data_cfg or {}
    image_root = root / image_dir
    mask_root = root / mask_dir

    split_file = root / _split_file_name(data_cfg, split)
    if split_file.exists():
        if not image_root.exists():
            raise FileNotFoundError(f"Image directory not found: {image_root}")
        if not mask_root.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_root}")
        pairs: list[ImageMaskPair] = []
        with split_file.open("r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if not token or token.startswith("#"):
                    continue
                pairs.append(ImageMaskPair(_resolve_file(image_root, token), _resolve_file(mask_root, token)))
        if not pairs:
            raise ValueError(f"No image/mask pairs listed in split file {split_file}")
        return pairs

    image_root = image_root / split
    mask_root = mask_root / split
    if not image_root.exists():
        raise FileNotFoundError(f"Image split directory not found: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask split directory not found: {mask_root}")

    masks_by_stem = {p.stem: p for p in mask_root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    pairs: list[ImageMaskPair] = []
    for image_path in sorted(image_root.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"No mask with matching stem for {image_path.name}")
        pairs.append(ImageMaskPair(image_path, mask_path))
    if not pairs:
        raise ValueError(f"No image/mask pairs found in split {split!r}")
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.int64)


def mask_has_boundary(mask: np.ndarray, ignore_index: int = 255) -> bool:
    valid = mask != ignore_index
    foreground = (mask > 0) & valid
    if not foreground.any():
        return False
    dy = np.zeros_like(foreground, dtype=bool)
    dx = np.zeros_like(foreground, dtype=bool)
    dy[1:] = foreground[1:] != foreground[:-1]
    dx[:, 1:] = foreground[:, 1:] != foreground[:, :-1]
    return bool((dy | dx).any())


def categorize_patch(mask_patch: np.ndarray, ignore_index: int = 255) -> str:
    foreground = (mask_patch > 0) & (mask_patch != ignore_index)
    if foreground.any():
        if mask_has_boundary(mask_patch, ignore_index):
            return "boundary"
        return "defect"
    return "normal"


class RandomAugment:
    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.hflip = float(config.get("horizontal_flip", 0.5))
        self.vflip = float(config.get("vertical_flip", 0.2))
        self.rotate90 = float(config.get("rotate90", 0.5))
        self.brightness = float(config.get("brightness", 0.15))
        self.contrast = float(config.get("contrast", 0.15))

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return image, mask
        if random.random() < self.hflip:
            image = np.ascontiguousarray(np.flip(image, axis=1))
            mask = np.ascontiguousarray(np.flip(mask, axis=1))
        if random.random() < self.vflip:
            image = np.ascontiguousarray(np.flip(image, axis=0))
            mask = np.ascontiguousarray(np.flip(mask, axis=0))
        if random.random() < self.rotate90:
            k = random.randint(0, 3)
            image = np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
            mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
        pil = Image.fromarray(image)
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            pil = ImageEnhance.Brightness(pil).enhance(factor)
        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            pil = ImageEnhance.Contrast(pil).enhance(factor)
        return np.asarray(pil, dtype=np.uint8), mask


class CCMPatchDataset(Dataset):
    def __init__(self, config: dict[str, Any], split: str = "train", training: bool = False) -> None:
        self.config = config
        data_cfg = config["dataset"]
        self.split = split
        self.training = training
        self.patch_size = int(data_cfg.get("patch_size", 512))
        self.overlap = float(data_cfg.get("overlap", 0.5))
        self.ignore_index = int(data_cfg.get("ignore_index", 255))
        self.cache_eval_images = bool(data_cfg.get("cache_eval_images", True)) and not training
        self.cache_eval_max_items = max(0, int(data_cfg.get("cache_eval_max_items", 4)))
        self._image_mask_cache: OrderedDict[tuple[Path, Path], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.feature_config = FeatureConfig.from_dict(config.get("features", {}))
        self.defer_features = uses_gpu_features(self.feature_config)
        self.augment = RandomAugment(data_cfg.get("augment", {})) if training else RandomAugment({"enabled": False})
        self.pairs = find_pairs(
            data_cfg["root"],
            data_cfg.get("image_dir", "images"),
            data_cfg.get("mask_dir", "annotations"),
            split,
            data_cfg,
        )
        self.records = self._build_records()
        self.sampled_indices = self._build_sampling_indices() if training else list(range(len(self.records)))

    def _build_records(self) -> list[PatchRecord]:
        records: list[PatchRecord] = []
        for pair in self.pairs:
            mask = load_mask(pair.mask_path)
            for coord in compute_patch_grid(mask.shape[0], mask.shape[1], self.patch_size, self.overlap):
                patch = extract_patch(mask, coord, self.patch_size, pad_value=self.ignore_index)
                records.append(PatchRecord(pair, coord, categorize_patch(patch, self.ignore_index)))
        return records

    def _build_sampling_indices(self) -> list[int]:
        sampling = self.config["dataset"].get("train_sampling", {"defect": 0.6, "boundary": 0.2, "normal": 0.2})
        by_category: dict[str, list[int]] = {"defect": [], "boundary": [], "normal": []}
        for idx, record in enumerate(self.records):
            by_category[record.category].append(idx)
        total = max(len(self.records), 1)
        sampled: list[int] = []
        for category, ratio in sampling.items():
            source = by_category.get(category, [])
            if not source:
                continue
            count = max(1, int(round(total * float(ratio))))
            sampled.extend(random.choices(source, k=count))
        return sampled or list(range(len(self.records)))

    def __len__(self) -> int:
        return len(self.sampled_indices)

    def _load_pair(self, pair: ImageMaskPair) -> tuple[np.ndarray, np.ndarray]:
        if not self.cache_eval_images or self.cache_eval_max_items <= 0:
            return load_rgb(pair.image_path), load_mask(pair.mask_path)
        key = (pair.image_path, pair.mask_path)
        cached = self._image_mask_cache.get(key)
        if cached is not None:
            self._image_mask_cache.move_to_end(key)
            return cached
        image = load_rgb(pair.image_path)
        mask = load_mask(pair.mask_path)
        self._image_mask_cache[key] = (image, mask)
        self._image_mask_cache.move_to_end(key)
        while len(self._image_mask_cache) > self.cache_eval_max_items:
            self._image_mask_cache.popitem(last=False)
        return image, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | tuple[int, int, int, int]]:
        record = self.records[self.sampled_indices[idx]]
        image, mask = self._load_pair(record.pair)
        image_patch = extract_patch(image, record.coord, self.patch_size, pad_value=0)
        mask_patch = extract_patch(mask, record.coord, self.patch_size, pad_value=self.ignore_index)
        image_patch, mask_patch = self.augment(image_patch, mask_patch)
        if self.defer_features:
            trad = np.zeros((0, self.patch_size, self.patch_size), dtype=np.float32)
        else:
            trad = compute_traditional_features(image_patch, self.feature_config)
        rgb = torch.from_numpy(image_patch.astype(np.float32).transpose(2, 0, 1) / 255.0)
        features = torch.from_numpy(trad)
        target = torch.from_numpy(mask_patch.astype(np.int64))
        return {
            "image": rgb,
            "features": features,
            "mask": target,
            "image_path": str(record.pair.image_path),
            "coord": (record.coord.y, record.coord.x, record.coord.h, record.coord.w),
        }
