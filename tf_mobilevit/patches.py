from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PatchCoord:
    y: int
    x: int
    h: int
    w: int


def compute_patch_grid(height: int, width: int, patch_size: int = 512, overlap: float = 0.5) -> list[PatchCoord]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, int(round(patch_size * (1 - overlap))))

    def starts(length: int) -> list[int]:
        if length <= patch_size:
            return [0]
        values = list(range(0, length - patch_size + 1, stride))
        last = length - patch_size
        if values[-1] != last:
            values.append(last)
        return values

    return [PatchCoord(y, x, min(patch_size, height - y), min(patch_size, width - x)) for y in starts(height) for x in starts(width)]


def extract_patch(array: np.ndarray, coord: PatchCoord, patch_size: int, pad_value: int | float = 0) -> np.ndarray:
    patch = array[coord.y : coord.y + coord.h, coord.x : coord.x + coord.w]
    if coord.h == patch_size and coord.w == patch_size:
        return patch
    if array.ndim == 2:
        out = np.full((patch_size, patch_size), pad_value, dtype=array.dtype)
        out[: coord.h, : coord.w] = patch
    else:
        out = np.full((patch_size, patch_size, array.shape[2]), pad_value, dtype=array.dtype)
        out[: coord.h, : coord.w, :] = patch
    return out


def stitch_probabilities(
    patches: list[torch.Tensor],
    coords: list[PatchCoord],
    output_size: tuple[int, int],
) -> torch.Tensor:
    if not patches:
        raise ValueError("patches must not be empty")
    channels = patches[0].shape[0]
    height, width = output_size
    device = patches[0].device
    acc = torch.zeros((channels, height, width), dtype=patches[0].dtype, device=device)
    counts = torch.zeros((1, height, width), dtype=patches[0].dtype, device=device)
    if len(patches) != len(coords):
        raise ValueError(f"patches and coords must have the same length, got {len(patches)} and {len(coords)}")
    for patch, coord in zip(patches, coords):
        valid = patch[:, : coord.h, : coord.w]
        acc[:, coord.y : coord.y + coord.h, coord.x : coord.x + coord.w] += valid
        counts[:, coord.y : coord.y + coord.h, coord.x : coord.x + coord.w] += 1
    return acc / counts.clamp_min(1)
