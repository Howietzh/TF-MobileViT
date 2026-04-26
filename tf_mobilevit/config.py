from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "num_workers": 2,
    "dataset": {
        "root": "data/CCM_Defect",
        "image_dir": "images",
        "mask_dir": "annotations",
        "layout": "split_files",
        "train_split_file": "train_subset_1.txt",
        "val_split_file": "val.txt",
        "test_split_file": "test.txt",
        "classes": ["background", "FPC cracks", "foreign particles", "end face notches", "lens scratches"],
        "patch_size": 512,
        "overlap": 0.5,
        "ignore_index": 255,
    },
    "features": {
        "enabled": True,
        "use_gabor": True,
        "use_lbp": True,
        "use_clahe": True,
        "gabor_orientations": [0, 30, 60, 90, 120, 150],
        "gabor_kernel_size": 21,
        "gabor_sigma": 4.0,
        "gabor_lambda": 10.0,
        "gabor_gamma": 0.5,
        "lbp_radius": 1,
        "lbp_points": 8,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": [8, 8],
    },
    "model": {
        "name": "tf_mobilevit",
        "timm_backbone": "mobilevit_xxs",
        "pretrained": False,
        "encoder_channels": [32, 48, 64, 96],
        "decoder_channels": 64,
        "use_tfgam": True,
    },
    "train": {
        "epochs": 50,
        "batch_size": 4,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "loss": {"dice": 1.0, "focal": 1.0, "boundary": 0.0},
        "output_dir": "runs/tf_mobilevit",
        "save_every": 10,
    },
    "eval": {"batch_size": 2, "threshold": 0.5},
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    with Path(path).open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_config)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
