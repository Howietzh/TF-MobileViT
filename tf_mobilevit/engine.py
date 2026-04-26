from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .config import save_config
from .data import CCMPatchDataset, load_rgb
from .features import FeatureConfig, compute_traditional_features, feature_channel_count
from .losses import CombinedLoss
from .metrics import SegmentationMetrics
from .model import build_model
from .patches import compute_patch_grid, extract_patch, stitch_probabilities
from .utils import count_parameters, ensure_dir, get_device, load_checkpoint, save_checkpoint, seed_everything, write_metrics


def make_loader(config: dict[str, Any], split: str, training: bool) -> DataLoader:
    dataset = CCMPatchDataset(config, split=split, training=training)
    batch_size = int(config["train" if training else "eval"].get("batch_size", 1))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        image = batch["image"].to(device)
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image, features)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * image.shape[0]
        count += image.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    model.eval()
    num_classes = len(config["dataset"]["classes"])
    ignore_index = int(config["dataset"].get("ignore_index", 255))
    metrics = SegmentationMetrics(num_classes, ignore_index)
    started = time.perf_counter()
    samples = 0
    for batch in loader:
        image = batch["image"].to(device)
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        logits = model(image, features)
        metrics.update(logits, mask)
        samples += image.shape[0]
    result = metrics.compute()
    result["fps"] = samples / max(time.perf_counter() - started, 1e-6)
    return result


def train(config: dict[str, Any]) -> dict[str, float]:
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["train"]["output_dir"])
    save_config(config, output_dir / "config.yaml")

    feature_config = FeatureConfig.from_dict(config.get("features", {}))
    trad_channels = feature_channel_count(feature_config)
    model = build_model(config, trad_channels).to(device)
    criterion = CombinedLoss(config, int(config["dataset"].get("ignore_index", 255)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 3e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    train_loader = make_loader(config, "train", training=True)
    val_loader = make_loader(config, "val", training=False)

    best_dice = -1.0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(config["train"].get("epochs", 50)) + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_model(model, val_loader, config, device)
        row = {"epoch": epoch, "loss": loss, "params": count_parameters(model), **metrics}
        history.append(row)
        write_metrics(output_dir / "history.csv", history)
        if metrics["foreground_dice"] > best_dice:
            best_dice = metrics["foreground_dice"]
            save_checkpoint(output_dir / "best.pt", model, config, metrics)
        if epoch % int(config["train"].get("save_every", 10)) == 0:
            save_checkpoint(output_dir / f"epoch_{epoch}.pt", model, config, metrics)
    final_metrics = dict(history[-1])
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    return {k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))}


def evaluate(config: dict[str, Any], checkpoint_path: str | Path) -> dict[str, float]:
    device = get_device(config.get("device", "auto"))
    feature_config = FeatureConfig.from_dict(config.get("features", {}))
    model = build_model(config, feature_channel_count(feature_config)).to(device)
    load_checkpoint(checkpoint_path, model, device)
    loader = make_loader(config, "test", training=False)
    metrics = evaluate_model(model, loader, config, device)
    out = ensure_dir(config["train"]["output_dir"])
    with (out / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


@torch.no_grad()
def infer_image(config: dict[str, Any], checkpoint_path: str | Path, image_path: str | Path, output_dir: str | Path) -> Path:
    device = get_device(config.get("device", "auto"))
    feature_config = FeatureConfig.from_dict(config.get("features", {}))
    model = build_model(config, feature_channel_count(feature_config)).to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()

    image = load_rgb(Path(image_path))
    height, width = image.shape[:2]
    patch_size = int(config["dataset"].get("patch_size", 512))
    overlap = float(config["dataset"].get("overlap", 0.5))
    coords = compute_patch_grid(height, width, patch_size, overlap)
    probs: list[torch.Tensor] = []
    for coord in coords:
        patch = extract_patch(image, coord, patch_size, pad_value=0)
        rgb = torch.from_numpy(patch.astype(np.float32).transpose(2, 0, 1) / 255.0)[None].to(device)
        trad = torch.from_numpy(compute_traditional_features(patch, feature_config))[None].to(device)
        logits = model(rgb, trad)
        probs.append(logits.softmax(dim=1).squeeze(0).cpu())
    stitched = stitch_probabilities(probs, coords, (height, width))
    pred = stitched.argmax(dim=0).numpy().astype(np.uint8)

    out_dir = ensure_dir(output_dir)
    out_path = out_dir / f"{Path(image_path).stem}_mask.png"
    Image.fromarray(pred).save(out_path)
    return out_path

