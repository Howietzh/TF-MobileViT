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


@dataclass
class QualitativeRow:
    class_id: int
    label: str
    sample_name: str
    area: int
    items: list[tuple[str, np.ndarray]]


@dataclass
class SelectedPrediction:
    selection: DefectSelection
    baseline_pred: np.ndarray
    tf_pred: np.ndarray
    baseline_iou: float
    tf_iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready qualitative comparison figures.")
    parser.add_argument("--baseline-run", default="runs/few_shot/mobilevit_no_trad/subset_2", help="Run dir containing baseline config.yaml and best.pt.")
    parser.add_argument("--tf-run", default="runs/few_shot/tf_mobilevit/subset_2", help="Run dir containing TF-MobileViT config.yaml and best.pt.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to visualize.")
    parser.add_argument("--output-dir", default="figures/qualitative_comparison/subset_2", help="Output directory.")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "both"], help="Output format.")
    parser.add_argument("--crop-size", type=int, default=384, help="Center crop size around the selected defect.")
    parser.add_argument("--class-ids", type=int, nargs="+", default=[1, 2, 3, 4], help="Foreground class IDs to visualize.")
    parser.add_argument("--sample-id", action="append", default=[], help="Optional CLASS_ID=IMAGE_STEM override. Can be repeated.")
    parser.add_argument("--max-samples", type=int, default=500, help="Maximum samples to scan per class when auto-selecting.")
    parser.add_argument("--batch-size", type=int, default=None, help="Patch inference batch size. Defaults to eval.batch_size.")
    parser.add_argument("--device", default="auto", help="Device override, e.g. auto, cpu, cuda:0.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Kept for backward compatibility; masks are rendered directly by default.")
    parser.add_argument("--output-name", default="qualitative_comparison", help="Base filename for the single paper figure.")
    parser.add_argument("--classes-per-row", type=int, default=2, help="Number of defect classes per row in the paper figure.")
    parser.add_argument("--gt-inset-scale", type=float, default=0.32, help="GT inset side length as a fraction of prediction cell size.")
    parser.add_argument("--gt-inset-margin", type=int, default=8, help="Margin in pixels between GT inset and prediction cell border.")
    parser.add_argument("--gt-inset-border", type=int, default=2, help="White border width around the GT inset in pixels. Set 0 to disable.")
    parser.add_argument("--selection-mode", default="iou_gap", choices=["iou_gap", "largest_area"], help="Sample selection strategy.")
    parser.add_argument("--single-figure", dest="single_figure", action="store_true", default=True, help="Generate one paper-ready grid figure.")
    parser.add_argument("--separate-figures", dest="single_figure", action="store_false", help="Generate one figure per defect class.")
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


def colorize_multiclass_mask(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    mask = np.asarray(mask)
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    ignore = mask == ignore_index
    safe_mask = np.where(ignore, 0, mask).astype(np.int64)
    output = CLASS_COLORS[safe_mask % len(CLASS_COLORS)].astype(np.uint8)
    output[ignore] = [80, 80, 80]
    return output


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


def single_class_iou(pred: np.ndarray, target: np.ndarray, class_id: int, ignore_index: int = 255) -> float:
    valid = target != ignore_index
    pred_fg = (pred == class_id) & valid
    target_fg = (target == class_id) & valid
    union = pred_fg | target_fg
    if not union.any():
        return 0.0
    return float((pred_fg & target_fg).sum() / union.sum())


def select_by_iou_gap(
    pairs: list[ImageMaskPair],
    class_id: int,
    sample_stem: str | None,
    baseline: LoadedModel,
    tf_mobilevit: LoadedModel,
    batch_size: int | None,
    max_samples: int,
    prediction_cache: dict[Path, tuple[np.ndarray, np.ndarray]],
) -> SelectedPrediction | None:
    candidates = pairs
    if sample_stem is not None:
        candidates = [pair for pair in pairs if pair.image_path.stem == sample_stem or pair.image_path.name == sample_stem]
    best: SelectedPrediction | None = None
    for pair in candidates[:max_samples]:
        gt_mask = load_mask(pair.mask_path)
        component = largest_component(gt_mask == class_id)
        if component is None:
            continue
        if pair.image_path not in prediction_cache:
            image = load_rgb(pair.image_path)
            prediction_cache[pair.image_path] = (
                predict_full_mask(baseline, image, batch_size),
                predict_full_mask(tf_mobilevit, image, batch_size),
            )
        baseline_pred, tf_pred = prediction_cache[pair.image_path]
        baseline_iou = single_class_iou(baseline_pred, gt_mask, class_id)
        tf_iou = single_class_iou(tf_pred, gt_mask, class_id)
        area, bbox, center = component
        selection = DefectSelection(class_id=class_id, pair=pair, center_y=center[0], center_x=center[1], area=area, bbox=bbox)
        current = SelectedPrediction(
            selection=selection,
            baseline_pred=baseline_pred,
            tf_pred=tf_pred,
            baseline_iou=baseline_iou,
            tf_iou=tf_iou,
        )
        if best is None or (tf_iou - baseline_iou) > (best.tf_iou - best.baseline_iou):
            best = current
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


def add_gt_inset(
    prediction_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    inset_scale: float = 0.32,
    margin: int = 8,
    border: int = 2,
) -> np.ndarray:
    """Paste the GT mask as a small, unlabeled inset in the bottom-right corner."""
    pred = Image.fromarray(to_uint8(prediction_rgb)).convert("RGB")
    gt = Image.fromarray(to_uint8(gt_rgb)).convert("RGB")
    cell_size = min(pred.width, pred.height)
    inset_size = max(1, int(round(cell_size * inset_scale)))
    inset_size = min(inset_size, pred.width - 2 * margin, pred.height - 2 * margin)
    gt = gt.resize((inset_size, inset_size), Image.Resampling.NEAREST)

    x0 = pred.width - margin - inset_size
    y0 = pred.height - margin - inset_size
    if border > 0:
        draw = ImageDraw.Draw(pred)
        draw.rectangle(
            [x0 - border, y0 - border, x0 + inset_size + border - 1, y0 + inset_size + border - 1],
            fill=(255, 255, 255),
        )
    pred.paste(gt, (x0, y0))
    return np.asarray(pred, dtype=np.uint8)


def make_class_canvas(items: list[tuple[str, np.ndarray]], cell_size: int) -> Image.Image:
    tiles: list[Image.Image] = []
    for _, image in items:
        pil = Image.fromarray(to_uint8(image)).convert("RGB")
        if pil.size != (cell_size, cell_size):
            pil = pil.resize((cell_size, cell_size), Image.Resampling.BILINEAR)
        tiles.append(pil)
    total_width = sum(item.width for item in tiles)
    total_height = max(item.height for item in tiles)
    canvas = Image.new("RGB", (total_width, total_height), "white")
    x = 0
    for item in tiles:
        canvas.paste(item, (x, 0))
        x += item.width
    return canvas


def make_paper_canvas(rows: list[QualitativeRow], cell_size: int, classes_per_row: int = 2) -> Image.Image:
    """Arrange rows as groups of classes: default 2 classes per row, 3 tiles per class = 2 x 6 for 4 classes."""
    if not rows:
        raise ValueError("No qualitative rows were generated.")
    if classes_per_row <= 0:
        raise ValueError("classes_per_row must be positive.")

    tiles_per_class = len(rows[0].items)
    row_count = int(np.ceil(len(rows) / classes_per_row))
    col_count = classes_per_row * tiles_per_class
    row_gap = 8
    col_gap = 8
    width = col_count * cell_size + max(0, col_count - 1) * col_gap
    height = row_count * cell_size + max(0, row_count - 1) * row_gap
    canvas = Image.new("RGB", (width, height), "white")

    for class_idx, row in enumerate(rows):
        group_row = class_idx // classes_per_row
        group_col = class_idx % classes_per_row
        y0 = group_row * (cell_size + row_gap)
        for tile_idx, (_, image) in enumerate(row.items):
            col = group_col * tiles_per_class + tile_idx
            x0 = col * (cell_size + col_gap)
            pil = Image.fromarray(to_uint8(image)).convert("RGB")
            if pil.size != (cell_size, cell_size):
                pil = pil.resize((cell_size, cell_size), Image.Resampling.BILINEAR)
            canvas.paste(pil, (x0, y0))
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


def build_qualitative_row(
    selected: SelectedPrediction,
    label: str,
    crop_size: int,
    alpha: float,
    gt_inset_scale: float,
    gt_inset_margin: int,
    gt_inset_border: int,
) -> QualitativeRow:
    selection = selected.selection
    image = load_rgb(selection.pair.image_path)
    gt_mask = load_mask(selection.pair.mask_path)

    rgb_crop = center_crop(image, selection.center_y, selection.center_x, crop_size, pad_value=0)
    gt_crop = center_crop(gt_mask, selection.center_y, selection.center_x, crop_size, pad_value=0)
    baseline_crop = center_crop(selected.baseline_pred, selection.center_y, selection.center_x, crop_size, pad_value=0)
    tf_crop = center_crop(selected.tf_pred, selection.center_y, selection.center_x, crop_size, pad_value=0)
    gt_rgb = colorize_multiclass_mask(gt_crop)
    baseline_rgb = add_gt_inset(
        colorize_multiclass_mask(baseline_crop),
        gt_rgb,
        inset_scale=gt_inset_scale,
        margin=gt_inset_margin,
        border=gt_inset_border,
    )
    tf_rgb = add_gt_inset(
        colorize_multiclass_mask(tf_crop),
        gt_rgb,
        inset_scale=gt_inset_scale,
        margin=gt_inset_margin,
        border=gt_inset_border,
    )
    return QualitativeRow(
        class_id=selection.class_id,
        label=label,
        sample_name=selection.pair.image_path.name,
        area=selection.area,
        items=[
            ("RGB", rgb_crop),
            ("MobileViT baseline + GT inset", baseline_rgb),
            ("TF-MobileViT + GT inset", tf_rgb),
        ],
    )


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

    rows: list[QualitativeRow] = []
    prediction_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    for class_id in args.class_ids:
        sample_stem = sample_overrides.get(class_id)
        if args.selection_mode == "iou_gap" or sample_stem is not None:
            selected = select_by_iou_gap(
                pairs=pairs,
                class_id=class_id,
                sample_stem=sample_stem,
                baseline=baseline,
                tf_mobilevit=tf_mobilevit,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                prediction_cache=prediction_cache,
            )
        else:
            selection = select_defect_sample(pairs, class_id, sample_stem, args.max_samples)
            if selection is None:
                selected = None
            else:
                image = load_rgb(selection.pair.image_path)
                baseline_pred = predict_full_mask(baseline, image, args.batch_size)
                tf_pred = predict_full_mask(tf_mobilevit, image, args.batch_size)
                gt_mask = load_mask(selection.pair.mask_path)
                selected = SelectedPrediction(
                    selection=selection,
                    baseline_pred=baseline_pred,
                    tf_pred=tf_pred,
                    baseline_iou=single_class_iou(baseline_pred, gt_mask, class_id),
                    tf_iou=single_class_iou(tf_pred, gt_mask, class_id),
                )

        if selected is None:
            print(f"Warning: no test sample found for class {class_id}. Skipping.", flush=True)
            continue

        label = class_label(baseline.config, class_id)
        print(
            f"class={class_id} label={label} image={selected.selection.pair.image_path.name} "
            f"baseline_iou={selected.baseline_iou:.4f} tf_iou={selected.tf_iou:.4f} "
            f"gap={selected.tf_iou - selected.baseline_iou:.4f} "
            f"center=({selected.selection.center_y},{selected.selection.center_x}) area={selected.selection.area}",
            flush=True,
        )
        row = build_qualitative_row(
            selected,
            label,
            args.crop_size,
            args.alpha,
            args.gt_inset_scale,
            args.gt_inset_margin,
            args.gt_inset_border,
        )
        rows.append(row)

    if args.single_figure:
        canvas = make_paper_canvas(rows, args.crop_size, args.classes_per_row)
        output_base = output_dir / args.output_name
        save_canvas(canvas, output_base, args.format)
        print(f"Saved {output_base.with_suffix('.' + ('pdf' if args.format != 'png' else 'png'))}", flush=True)
    else:
        for row in rows:
            canvas = make_class_canvas(row.items, args.crop_size)
            output_base = output_dir / f"class_{row.class_id}_{safe_name(row.label)}"
            save_canvas(canvas, output_base, args.format)
            print(f"Saved {output_base.with_suffix('.' + ('pdf' if args.format != 'png' else 'png'))}", flush=True)


if __name__ == "__main__":
    main()
