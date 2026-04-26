from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import save_config
from .data import CCMPatchDataset, load_rgb
from .features import FeatureConfig, TraditionalFeatureExtractor, compute_traditional_features, feature_channel_count, uses_gpu_features
from .losses import CombinedLoss
from .metrics import SegmentationMetrics
from .model import build_model
from .patches import compute_patch_grid, extract_patch, stitch_probabilities
from .utils import count_parameters, ensure_dir, get_device, load_checkpoint, save_checkpoint, seed_everything, write_metrics


def distributed_is_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if distributed_is_available() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(config: dict[str, Any]) -> tuple[torch.device, bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = get_device(config.get("device", "auto"))
    return device, distributed, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if distributed_is_available():
        dist.destroy_process_group()


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if distributed_is_available():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def make_loader(config: dict[str, Any], split: str, training: bool, distributed: bool = False) -> DataLoader:
    dataset = CCMPatchDataset(config, split=split, training=training)
    batch_size = int(config["train" if training else "eval"].get("batch_size", 1))
    sampler = DistributedSampler(dataset, shuffle=training) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def build_feature_extractor(feature_config: FeatureConfig, device: torch.device) -> TraditionalFeatureExtractor | None:
    if not uses_gpu_features(feature_config):
        return None
    extractor = TraditionalFeatureExtractor(feature_config).to(device)
    extractor.eval()
    return extractor


def feature_channels_for_model(feature_config: FeatureConfig) -> int:
    if uses_gpu_features(feature_config):
        return TraditionalFeatureExtractor(feature_config).out_channels
    return feature_channel_count(feature_config)


def get_batch_features(
    batch: dict[str, Any],
    image: torch.Tensor,
    feature_extractor: TraditionalFeatureExtractor | None,
    device: torch.device,
) -> torch.Tensor:
    if feature_extractor is not None:
        return feature_extractor(image)
    return batch["features"].to(device)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    writer: Any | None,
    global_step: int,
    log_interval: int,
    feature_extractor: TraditionalFeatureExtractor | None,
    print_logs: bool = True,
) -> tuple[float, int]:
    model.train()
    total = 0.0
    count = 0
    num_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        batch_started = time.perf_counter()
        image = batch["image"].to(device)
        features = get_batch_features(batch, image, feature_extractor, device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image, features)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        batch_time = time.perf_counter() - batch_started
        lr = float(optimizer.param_groups[0]["lr"])
        total += loss_value * image.shape[0]
        count += image.shape[0]
        global_step += 1

        if writer is not None:
            writer.add_scalar("train/loss", loss_value, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar("train/batch_time_sec", batch_time, global_step)

        if print_logs and log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == 1 or batch_idx == num_batches):
            print(
                f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx}/{num_batches}] "
                f"loss={loss_value:.4f} lr={lr:.2e} time={batch_time:.2f}s",
                flush=True,
            )
    total = reduce_sum(total, device)
    count = reduce_sum(count, device)
    return total / max(count, 1), global_step


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    feature_extractor: TraditionalFeatureExtractor | None = None,
) -> dict[str, float]:
    model.eval()
    num_classes = len(config["dataset"]["classes"])
    ignore_index = int(config["dataset"].get("ignore_index", 255))
    metrics = SegmentationMetrics(num_classes, ignore_index)
    started = time.perf_counter()
    samples = 0
    for batch in loader:
        image = batch["image"].to(device)
        features = get_batch_features(batch, image, feature_extractor, device)
        mask = batch["mask"].to(device)
        logits = model(image, features)
        metrics.update(logits, mask)
        samples += image.shape[0]
    if distributed_is_available():
        confusion = metrics.confusion.to(device)
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
        metrics.confusion = confusion.cpu()
        samples_tensor = torch.tensor(samples, dtype=torch.float64, device=device)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        samples = int(samples_tensor.item())
    result = metrics.compute()
    result["fps"] = samples / max(time.perf_counter() - started, 1e-6)
    return result


def train(config: dict[str, Any]) -> dict[str, float]:
    seed_everything(int(config.get("seed", 42)))
    device, distributed, rank, local_rank, world_size = setup_distributed(config)
    seed_everything(int(config.get("seed", 42)) + rank)
    output_dir = Path(config["train"]["output_dir"])
    if is_main_process():
        ensure_dir(output_dir)
        save_config(config, output_dir / "config.yaml")

    feature_config = FeatureConfig.from_dict(config.get("features", {}))
    trad_channels = feature_channels_for_model(feature_config)
    feature_extractor = build_feature_extractor(feature_config, device)
    model = build_model(config, trad_channels).to(device)
    if distributed:
        ddp_kwargs: dict[str, Any] = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        model = DistributedDataParallel(model, **ddp_kwargs)
    criterion = CombinedLoss(config, int(config["dataset"].get("ignore_index", 255)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 3e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    train_loader = make_loader(config, "train", training=True, distributed=distributed)
    val_loader = make_loader(config, "val", training=False, distributed=distributed)

    tensorboard_enabled = bool(config["train"].get("tensorboard", True))
    writer: Any | None = None
    if tensorboard_enabled and is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        except Exception as exc:
            warnings.warn(
                "TensorBoard logging is enabled, but SummaryWriter could not be initialized. "
                "Training will continue without TensorBoard. Install dependencies with "
                "`pip install -r requirements.txt` or set `train.tensorboard: false` in the config. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            writer = None

    best_dice = -1.0
    history: list[dict[str, float | int]] = []
    global_step = 0
    total_epochs = int(config["train"].get("epochs", 50))
    log_interval = int(config["train"].get("log_interval", 10))
    params = count_parameters(model)
    if is_main_process():
        print(
            f"Training device={device} distributed={distributed} world_size={world_size} "
            f"batch_size_per_process={config['train'].get('batch_size', 1)}",
            flush=True,
        )
    try:
        for epoch in range(1, total_epochs + 1):
            if distributed and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            epoch_started = time.perf_counter()
            loss, global_step = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                total_epochs,
                writer,
                global_step,
                log_interval,
                feature_extractor,
                is_main_process(),
            )
            metrics = evaluate_model(model, val_loader, config, device, feature_extractor)
            epoch_time = time.perf_counter() - epoch_started
            is_best = metrics["foreground_dice"] > best_dice
            if is_best and is_main_process():
                best_dice = metrics["foreground_dice"]
                save_checkpoint(output_dir / "best.pt", unwrap_model(model), config, metrics)
            elif is_best:
                best_dice = metrics["foreground_dice"]

            row = {
                "epoch": epoch,
                "train_loss": loss,
                "params": params,
                "epoch_time_sec": epoch_time,
                "best_foreground_dice": best_dice,
                **metrics,
            }
            history.append(row)
            if is_main_process():
                write_metrics(output_dir / "history.csv", history)

            if writer is not None and is_main_process():
                writer.add_scalar("epoch/train_loss", loss, epoch)
                writer.add_scalar("val/mean_dice", metrics["mean_dice"], epoch)
                writer.add_scalar("val/foreground_dice", metrics["foreground_dice"], epoch)
                writer.add_scalar("val/mean_iou", metrics["mean_iou"], epoch)
                writer.add_scalar("val/foreground_iou", metrics["foreground_iou"], epoch)
                writer.add_scalar("val/mean_recall", metrics["mean_recall"], epoch)
                writer.add_scalar("val/foreground_recall", metrics["foreground_recall"], epoch)
                writer.add_scalar("val/fps", metrics["fps"], epoch)
                writer.add_scalar("train/epoch_time_sec", epoch_time, epoch)
                writer.add_scalar("model/params", params, epoch)
                writer.flush()

            if is_main_process():
                print(
                    f"Epoch {epoch}/{total_epochs} done | train_loss={loss:.4f} "
                    f"| fg_dice={metrics['foreground_dice']:.4f} "
                    f"| fg_iou={metrics['foreground_iou']:.4f} "
                    f"| recall={metrics['foreground_recall']:.4f} "
                    f"| fps={metrics['fps']:.2f} | best={best_dice:.4f}"
                    f"{' | saved best' if is_best else ''}",
                    flush=True,
                )

            if is_main_process() and epoch % int(config["train"].get("save_every", 10)) == 0:
                save_checkpoint(output_dir / f"epoch_{epoch}.pt", unwrap_model(model), config, metrics)
        final_metrics = dict(history[-1])
        if is_main_process():
            with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(final_metrics, f, indent=2)
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()
    return {k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))}


def evaluate(config: dict[str, Any], checkpoint_path: str | Path) -> dict[str, float]:
    device = get_device(config.get("device", "auto"))
    feature_config = FeatureConfig.from_dict(config.get("features", {}))
    feature_extractor = build_feature_extractor(feature_config, device)
    model = build_model(config, feature_channels_for_model(feature_config)).to(device)
    load_checkpoint(checkpoint_path, model, device)
    loader = make_loader(config, "test", training=False)
    metrics = evaluate_model(model, loader, config, device, feature_extractor)
    out = ensure_dir(config["train"]["output_dir"])
    with (out / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


@torch.no_grad()
def infer_image(config: dict[str, Any], checkpoint_path: str | Path, image_path: str | Path, output_dir: str | Path) -> Path:
    device = get_device(config.get("device", "auto"))
    feature_config = FeatureConfig.from_dict(config.get("features", {}))
    feature_extractor = build_feature_extractor(feature_config, device)
    model = build_model(config, feature_channels_for_model(feature_config)).to(device)
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
        if feature_extractor is not None:
            trad = feature_extractor(rgb)
        else:
            trad = torch.from_numpy(compute_traditional_features(patch, feature_config))[None].to(device)
        logits = model(rgb, trad)
        probs.append(logits.softmax(dim=1).squeeze(0).cpu())
    stitched = stitch_probabilities(probs, coords, (height, width))
    pred = stitched.argmax(dim=0).numpy().astype(np.uint8)

    out_dir = ensure_dir(output_dir)
    out_path = out_dir / f"{Path(image_path).stem}_mask.png"
    Image.fromarray(pred).save(out_path)
    return out_path
