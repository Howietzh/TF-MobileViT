from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DEFAULT_CONFIG
from .engine import evaluate, infer_image, train


def _write_split(root: Path, split: str) -> None:
    image_dir = root / "images" / split
    mask_dir = root / "masks" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        image[..., 0] = np.linspace(0, 255, 96, dtype=np.uint8)[None, :]
        image[..., 1] = 80 + idx * 20
        image[24:72, 40 + idx : 43 + idx, :] = 240
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[24:72, 40 + idx : 43 + idx] = 1 + idx
        Image.fromarray(image).save(image_dir / f"sample_{idx}.png")
        Image.fromarray(mask).save(mask_dir / f"sample_{idx}.png")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ccm"
        for split in ("train", "val", "test"):
            _write_split(root, split)
        cfg = deepcopy(DEFAULT_CONFIG)
        cfg["dataset"]["root"] = str(root)
        cfg["dataset"]["patch_size"] = 64
        cfg["dataset"]["overlap"] = 0.5
        cfg["dataset"]["classes"] = ["background", "fpc_crack", "end_face_scratch"]
        cfg["features"]["gabor_kernel_size"] = 9
        cfg["model"]["timm_backbone"] = None
        cfg["model"]["encoder_channels"] = [8, 12, 16, 24]
        cfg["model"]["decoder_channels"] = 16
        cfg["train"]["epochs"] = 1
        cfg["train"]["batch_size"] = 1
        cfg["train"]["output_dir"] = str(Path(tmp) / "run")
        cfg["eval"]["batch_size"] = 1
        metrics = train(cfg)
        evaluate(cfg, Path(cfg["train"]["output_dir"]) / "best.pt")
        out = infer_image(cfg, Path(cfg["train"]["output_dir"]) / "best.pt", root / "images" / "test" / "sample_0.png", Path(tmp) / "pred")
        print({"smoke": "ok", "metrics": metrics, "prediction": str(out)})


if __name__ == "__main__":
    main()

