from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tf_mobilevit.config import load_config
from tf_mobilevit.data import ImageMaskPair, find_pairs, load_mask, load_rgb
from tf_mobilevit.engine import build_feature_extractor, cuda_autocast, feature_channels_for_model
from tf_mobilevit.features import FeatureConfig, compute_traditional_features
from tf_mobilevit.model import build_model
from tf_mobilevit.patches import compute_patch_grid, extract_patch, stitch_probabilities
from tf_mobilevit.utils import get_device, load_checkpoint


CLASS_COLORS = np.array(
    [
        [0, 0, 0],
        [230, 57, 70],
        [42, 157, 143],
        [244, 162, 97],
        [69, 123, 157],
        [131, 56, 236],
        [255, 190, 11],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class DefectSelection:
    class_id: int
    pair: ImageMaskPair
    center_y: int
    center_x: int
    area: int
    bbox: tuple[int, int, int, int]


@dataclass
class LoadedModel:
    config: dict[str, Any]
    model: torch.nn.Module
    feature_extractor: torch.nn.Module | None
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate class-wise qualitative comparison figures.")
    parser.add_argument("--baseline-run", default="runs/few_shot/mobilevit_no_trad/subset_2", help="Run dir containing baseline config.yaml and best.pt.")
    parser.add_argument("--tf-run", default="runs/few_shot/tf_mobilevit/subset_2", help="Run dir containing TF-MobileViT config.yaml and best.pt.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to visualize.")
    parser.add_argument("--output-dir", default="figures/qualitative_comparison/subset_2", help="Output directory.")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "both"], help="Output format.")
    parser.add_argument("--crop-size", type=int, default=512, help="Center crop size around the selected defect.")
    parser.add_argument("--class-ids", type=int, nargs="+", default=[1, 2, 3, 4], help="Foreground class IDs to visualize.")
    parser.add_argument("--sample-id", action="append", default=[], help="Optional CLASS_ID=IMAGE_STEM override. Can be repeated.")
    parser.add_argument("--max-samples", type=int, default=500, help="Maximum samples to scan per class when auto-selecting.")
    parser.add_argument("--batch-size", type=int, default=None, help="Patch inference batch size. Defaults to eval.batch_size.")
    parser.add_argument("--device", default="auto", help="Device override, e.g. auto, cpu, cuda:0.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for GT and prediction masks.")
    return parser.parse_args()


def parse_sample_overrides(items: list[str]) -> dict[int, str]:
    overrides: dict[int, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--sample-id must use CLASS_ID=IMAGE_STEM format, got {item!r}")
        class_text, sample = item.split("=", 1)
        overrides[int(class_text)] = Path(sample).stem
    return overrides


def safe_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.max(initial=0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def class_color(class_id: int) -> np.ndarray:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def overlay_multiclass(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    image = to_uint8(image).copy()
    foreground = mask > 0
    if not foreground.any():
        return image
    safe_mask = np.where(foreground, mask, 0).astype(np.int64)
    colors = CLASS_COLORS[safe_mask % len(CLASS_COLORS)]
    image[foreground] = ((1.0 - alpha) * image[foreground].astype(np.float32) + alpha * colors[foreground]).astype(np.uint8)
    return image


def largest_component(binary: np.ndarray) -> tuple[int, tuple[int, int, int, int], tuple[int, int]] | None:
    foreground = np.argwhere(binary)
    if foreground.size == 0:
        return None
    visited = np.zeros(binary.shape, dtype=bool)
    best_area = 0
    best_bbox = (0, 0, 0, 0)
    best_center = (0, 0)
    height, width = binary.shape
    for start_y, start_x in foreground:
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        area = 0
        min_y = max_y = int(start_y)
        min_x = max_x = int(start_x)
        sum_y = 0
        sum_x = 0
        while queue:
            y, x = queue.popleft()
            area += 1
            sum_y += y
            sum_x += x
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if area > best_area:
            best_area = area
            best_bbox = (min_y, min_x, max_y + 1, max_x + 1)
            best_center = (round(sum_y / area), round(sum_x / area))
    return best_area, best_bbox, best_center


def select_defect_sample(
    pairs: list[ImageMaskPair],
    class_id: int,
    sample_stem: str | None,
    max_samples: int,
) -> DefectSelection | None:
    candidates = pairs
    if sample_stem is not None:
        candidates = [pair for pair in pairs if pair.image_path.stem == sample_stem or pair.image_path.name == sample_stem]
    best: DefectSelection | None = None
    for pair in candidates[:max_samples]:
        mask = load_mask(pair.mask_path)
        component = largest_component(mask == class_id)
        if component is None:
            continue
        area, bbox, center = component
        if best is None or area > best.area:
            best = DefectSelection(class_id=class_id, pair=pair, center_y=center[0], center_x=center[1], area=area, bbox=bbox)
    return best


def center_crop(array: np.ndarray, center_y: int, center_x: int, crop_size: int, pad_value: int = 0) -> np.ndarray:
    half = crop_size // 2
    y0 = center_y - half
    x0 = center_x - half
    y1 = y0 + crop_size
    x1 = x0 + crop_size
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(array.shape[0], y1)
    src_x1 = min(array.shape[1], x1)
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    if array.ndim == 2:
        out = np.full((crop_size, crop_size), pad_value, dtype=array.dtype)
        out[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = array[src_y0:src_y1, src_x0:src_x1]
    else:
        out = np.full((crop_size, crop_size, array.shape[2]), pad_value, dtype=array.dtype)
        out[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def load_run(run_dir: Path, device_name: str) -> LoadedModel:
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "best.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    config = load_config(config_path)
    config["device"] = device_name
    device = get_device(device_name)
    feature_config = FeatureConfig.from_dict(config.get("features", {}))
    feature_extractor = build_feature_extractor(feature_config, device)
    model = build_model(config, feature_channels_for_model(feature_config)).to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()
    return LoadedModel(config=config, model=model, feature_extractor=feature_extractor, device=device)


def predict_full_mask(loaded: LoadedModel, image: np.ndarray, batch_size: int | None) -> np.ndarray:
    config = loaded.config
    feature_config = FeatureConfig.from_dict(config.get("features", {}))
    patch_size = int(config["dataset"].get("patch_size", 512))
    overlap = float(config["dataset"].get("overlap", 0.5))
    eval_batch_size = int(batch_size or config.get("eval", {}).get("batch_size", 1))
    amp_enabled = bool(config.get("eval", {}).get("amp", False)) and loaded.device.type == "cuda"
    coords = compute_patch_grid(image.shape[0], image.shape[1], patch_size, overlap)
    probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(coords), eval_batch_size):
            batch_coords = coords[start : start + eval_batch_size]
            patches = [extract_patch(image, coord, patch_size, pad_value=0) for coord in batch_coords]
            rgb = torch.from_numpy(np.stack(patches).astype(np.float32).transpose(0, 3, 1, 2) / 255.0).to(loaded.device)
            if loaded.feature_extractor is not None:
                trad = loaded.feature_extractor(rgb)
            else:
                trad_np = [compute_traditional_features(patch, feature_config) for patch in patches]
                trad = torch.from_numpy(np.stack(trad_np).astype(np.float32)).to(loaded.device)
            with cuda_autocast(amp_enabled):
                logits = loaded.model(rgb, trad)
            probabilities.extend(logits.softmax(dim=1).cpu())
    stitched = stitch_probabilities(probabilities, coords, (image.shape[0], image.shape[1]))
    return stitched.argmax(dim=0).numpy().astype(np.uint8)


def add_label(image: Image.Image, label: str, height: int = 34) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + height), "white")
    output.paste(image, (0, height))
    draw = ImageDraw.Draw(output)
    draw.text((8, 10), label, fill=(20, 20, 20), font=ImageFont.load_default())
    return output


def make_comparison_canvas(items: list[tuple[str, np.ndarray]], cell_size: int = 512) -> Image.Image:
    labeled: list[Image.Image] = []
    for label, image in items:
        pil = Image.fromarray(to_uint8(image)).convert("RGB")
        if pil.size != (cell_size, cell_size):
            pil = pil.resize((cell_size, cell_size), Image.Resampling.BILINEAR)
        labeled.append(add_label(pil, label))
    total_width = sum(item.width for item in labeled)
    total_height = max(item.height for item in labeled)
    canvas = Image.new("RGB", (total_width, total_height), "white")
    x = 0
    for item in labeled:
        canvas.paste(item, (x, 0))
        x += item.width
    return canvas


def save_canvas(canvas: Image.Image, output_base: Path, fmt: str) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if fmt in {"pdf", "both"}:
        canvas.save(output_base.with_suffix(".pdf"), "PDF", resolution=300.0)
    if fmt in {"png", "both"}:
        canvas.save(output_base.with_suffix(".png"))


def class_label(config: dict[str, Any], class_id: int) -> str:
    classes = config["dataset"].get("classes", [])
    if 0 <= class_id < len(classes):
        return str(classes[class_id])
    return f"class_{class_id}"


def main() -> None:
    args = parse_args()
    baseline_run = Path(args.baseline_run)
    tf_run = Path(args.tf_run)
    output_dir = Path(args.output_dir)
    sample_overrides = parse_sample_overrides(args.sample_id)

    baseline = load_run(baseline_run, args.device)
    tf_mobilevit = load_run(tf_run, args.device)
    data_cfg = baseline.config["dataset"]
    pairs = find_pairs(
        data_cfg["root"],
        data_cfg.get("image_dir", "images"),
        data_cfg.get("mask_dir", "annotations"),
        args.split,
        data_cfg,
    )

    for class_id in args.class_ids:
        selection = select_defect_sample(pairs, class_id, sample_overrides.get(class_id), args.max_samples)
        if selection is None:
            print(f"Warning: no test sample found for class {class_id}. Skipping.", flush=True)
            continue

        label = class_label(baseline.config, class_id)
        print(
            f"Generating class {class_id} ({label}) from {selection.pair.image_path.name} "
            f"center=({selection.center_y},{selection.center_x}) area={selection.area}",
            flush=True,
        )
        image = load_rgb(selection.pair.image_path)
        gt_mask = load_mask(selection.pair.mask_path)
        baseline_pred = predict_full_mask(baseline, image, args.batch_size)
        tf_pred = predict_full_mask(tf_mobilevit, image, args.batch_size)

        rgb_crop = center_crop(image, selection.center_y, selection.center_x, args.crop_size, pad_value=0)
        gt_crop = center_crop(gt_mask, selection.center_y, selection.center_x, args.crop_size, pad_value=0)
        baseline_crop = center_crop(baseline_pred, selection.center_y, selection.center_x, args.crop_size, pad_value=0)
        tf_crop = center_crop(tf_pred, selection.center_y, selection.center_x, args.crop_size, pad_value=0)

        canvas = make_comparison_canvas(
            [
                ("RGB", rgb_crop),
                ("Ground Truth", overlay_multiclass(rgb_crop, gt_crop, args.alpha)),
                ("MobileViT baseline", overlay_multiclass(rgb_crop, baseline_crop, args.alpha)),
                ("TF-MobileViT", overlay_multiclass(rgb_crop, tf_crop, args.alpha)),
            ],
            cell_size=args.crop_size,
        )
        output_base = output_dir / f"class_{class_id}_{safe_name(label)}"
        save_canvas(canvas, output_base, args.format)
        print(f"Saved {output_base.with_suffix('.' + ('pdf' if args.format != 'png' else 'png'))}", flush=True)


if __name__ == "__main__":
    main()
