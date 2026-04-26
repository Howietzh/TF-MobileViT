from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageOps


@dataclass(frozen=True)
class FeatureConfig:
    enabled: bool = True
    use_gabor: bool = True
    use_lbp: bool = True
    use_clahe: bool = True
    compute_device: str = "cpu"
    gpu_features: dict[str, bool] | None = None
    gabor_orientations: tuple[int, ...] = (0, 30, 60, 90, 120, 150)
    gabor_kernel_size: int = 21
    gabor_sigma: float = 4.0
    gabor_lambda: float = 10.0
    gabor_gamma: float = 0.5
    lbp_radius: int = 1
    lbp_points: int = 8
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FeatureConfig":
        values = dict(config)
        values["gabor_orientations"] = tuple(values.get("gabor_orientations", cls.gabor_orientations))
        values["clahe_tile_grid_size"] = tuple(values.get("clahe_tile_grid_size", cls.clahe_tile_grid_size))
        return cls(**values)


def feature_channel_count(config: FeatureConfig) -> int:
    if not config.enabled:
        return 0
    count = 0
    if config.use_gabor:
        count += len(config.gabor_orientations)
    if config.use_lbp:
        count += 1
    if config.use_clahe:
        count += 1
    return count


def uses_gpu_features(config: FeatureConfig) -> bool:
    return config.enabled and config.compute_device == "gpu"


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    gray = gray.astype(np.float32)
    if gray.max() > 1.5:
        gray /= 255.0
    return gray.clip(0.0, 1.0)


def normalize01(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    array = array.astype(np.float32)
    mn = float(array.min())
    mx = float(array.max())
    if mx - mn < eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array - mn) / (mx - mn)


def gabor_kernel(size: int, theta_degrees: float, sigma: float, wavelength: float, gamma: float) -> torch.Tensor:
    if size % 2 == 0:
        size += 1
    half = size // 2
    y, x = torch.meshgrid(torch.arange(-half, half + 1), torch.arange(-half, half + 1), indexing="ij")
    theta = math.radians(theta_degrees)
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)
    envelope = torch.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2))
    wave = torch.cos(2 * math.pi * x_theta / wavelength)
    kernel = envelope * wave
    kernel -= kernel.mean()
    kernel /= kernel.abs().sum().clamp_min(1e-6)
    return kernel.float()


def normalize_tensor01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(2)
    mn = flat.min(dim=2).values[:, :, None, None]
    mx = flat.max(dim=2).values[:, :, None, None]
    return (x - mn) / (mx - mn).clamp_min(eps)


class TraditionalFeatureExtractor(nn.Module):
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        gpu_features = config.gpu_features or {}
        self.use_gabor = config.enabled and config.use_gabor and bool(gpu_features.get("gabor", True))
        self.use_lbp = config.enabled and config.use_lbp and bool(gpu_features.get("lbp", True))
        # CLAHE is intentionally kept out of the GPU path; it is histogram based.
        self.use_clahe = False
        if self.use_gabor:
            kernels = [
                gabor_kernel(
                    config.gabor_kernel_size,
                    theta,
                    config.gabor_sigma,
                    config.gabor_lambda,
                    config.gabor_gamma,
                )
                for theta in config.gabor_orientations
            ]
            self.register_buffer("gabor_kernels", torch.stack(kernels)[:, None])
        else:
            self.register_buffer("gabor_kernels", torch.empty(0, 1, 1, 1))

    @property
    def out_channels(self) -> int:
        return (len(self.config.gabor_orientations) if self.use_gabor else 0) + (1 if self.use_lbp else 0)

    @staticmethod
    def rgb_to_gray_tensor(image: torch.Tensor) -> torch.Tensor:
        return image[:, 0:1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not self.config.enabled or self.out_channels == 0:
            return image.new_zeros((image.shape[0], 0, image.shape[2], image.shape[3]))
        gray = self.rgb_to_gray_tensor(image)
        channels: list[torch.Tensor] = []
        if self.use_gabor:
            kernels = self.gabor_kernels.to(dtype=image.dtype, device=image.device)
            responses = F.conv2d(gray, kernels, padding=kernels.shape[-1] // 2).abs()
            channels.append(normalize_tensor01(responses))
        if self.use_lbp:
            channels.append(self.compute_lbp(gray, self.config.lbp_radius))
        return torch.cat(channels, dim=1) if channels else image.new_zeros((image.shape[0], 0, image.shape[2], image.shape[3]))

    @staticmethod
    def compute_lbp(gray: torch.Tensor, radius: int = 1) -> torch.Tensor:
        padded = F.pad(gray, (radius, radius, radius, radius), mode="replicate")
        h, w = gray.shape[-2:]
        center = padded[:, :, radius : radius + h, radius : radius + w]
        offsets = [
            (-radius, -radius),
            (-radius, 0),
            (-radius, radius),
            (0, radius),
            (radius, radius),
            (radius, 0),
            (radius, -radius),
            (0, -radius),
        ]
        code = torch.zeros_like(gray)
        for bit, (dy, dx) in enumerate(offsets):
            neigh = padded[:, :, radius + dy : radius + dy + h, radius + dx : radius + dx + w]
            code = code + (neigh >= center).to(gray.dtype) * float(1 << bit)
        return code / 255.0


def compute_gabor(gray: np.ndarray, config: FeatureConfig) -> np.ndarray:
    tensor = torch.from_numpy(gray).float()[None, None]
    responses: list[np.ndarray] = []
    for theta in config.gabor_orientations:
        kernel = gabor_kernel(
            config.gabor_kernel_size,
            theta,
            config.gabor_sigma,
            config.gabor_lambda,
            config.gabor_gamma,
        )[None, None]
        response = F.conv2d(tensor, kernel, padding=kernel.shape[-1] // 2).abs().squeeze().numpy()
        responses.append(normalize01(response))
    return np.stack(responses, axis=0)


def compute_lbp(gray: np.ndarray, radius: int = 1, points: int = 8) -> np.ndarray:
    if points != 8:
        # The implementation keeps the dependency footprint small. Config accepts
        # other values for future OpenCV/skimage-backed variants.
        points = 8
    padded = np.pad(gray, radius, mode="edge")
    center = padded[radius : radius + gray.shape[0], radius : radius + gray.shape[1]]
    offsets = [
        (-radius, -radius),
        (-radius, 0),
        (-radius, radius),
        (0, radius),
        (radius, radius),
        (radius, 0),
        (radius, -radius),
        (0, -radius),
    ]
    code = np.zeros_like(gray, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neigh = padded[radius + dy : radius + dy + gray.shape[0], radius + dx : radius + dx + gray.shape[1]]
        code |= ((neigh >= center).astype(np.uint8) << bit)
    return normalize01(code.astype(np.float32))


def compute_clahe_like(gray: np.ndarray, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    # PIL has global histogram equalization. Applying it per tile and blending by
    # resizing gives a lightweight CLAHE-style local contrast prior without
    # requiring OpenCV at runtime.
    h, w = gray.shape
    tiles_y, tiles_x = tile_grid_size
    tile_h = max(1, math.ceil(h / tiles_y))
    tile_w = max(1, math.ceil(w / tiles_x))
    out = np.zeros_like(gray, dtype=np.float32)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0, x0 = ty * tile_h, tx * tile_w
            y1, x1 = min(h, y0 + tile_h), min(w, x0 + tile_w)
            tile = (gray[y0:y1, x0:x1] * 255).astype(np.uint8)
            eq = ImageOps.equalize(Image.fromarray(tile))
            out[y0:y1, x0:x1] = np.asarray(eq, dtype=np.float32) / 255.0
    return normalize01(out)


def compute_traditional_features(image: np.ndarray, config: FeatureConfig) -> np.ndarray:
    if not config.enabled:
        return np.zeros((0, image.shape[0], image.shape[1]), dtype=np.float32)
    gray = rgb_to_gray(image)
    channels: list[np.ndarray] = []
    if config.use_gabor:
        channels.extend(list(compute_gabor(gray, config)))
    if config.use_lbp:
        channels.append(compute_lbp(gray, config.lbp_radius, config.lbp_points))
    if config.use_clahe:
        channels.append(compute_clahe_like(gray, config.clahe_tile_grid_size))
    if not channels:
        return np.zeros((0, image.shape[0], image.shape[1]), dtype=np.float32)
    return np.stack(channels, axis=0).astype(np.float32)
