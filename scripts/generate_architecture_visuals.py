from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tf_mobilevit.config import load_config
from tf_mobilevit.data import find_pairs, load_mask, load_rgb
from tf_mobilevit.features import FeatureConfig, compute_clahe_like, compute_gabor, compute_lbp, rgb_to_gray
from tf_mobilevit.patches import PatchCoord, compute_patch_grid, extract_patch


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TF-MobileViT paper architecture visual inputs/outputs.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to search.")
    parser.add_argument("--output-dir", default="runs/architecture_visuals", help="Directory for generated images.")
    parser.add_argument("--sample-id", default=None, help="Optional image stem or filename to visualize.")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size override.")
    parser.add_argument("--overlap", type=float, default=None, help="Patch overlap override.")
    parser.add_argument("--max-samples", type=int, default=200, help="Maximum samples to scan when auto-selecting.")
    parser.add_argument("--montage-width", type=int, default=1800, help="Width of the summary montage.")
    return parser.parse_args()


def to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(to_uint8(gray)[..., None], 3, axis=2)


def colorize_mask(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != ignore_index
    safe_mask = np.where(valid, mask, 0).astype(np.int64)
    colors = CLASS_COLORS[safe_mask % len(CLASS_COLORS)]
    output[valid] = colors[valid]
    output[~valid] = [80, 80, 80]
    return output


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45, ignore_index: int = 255) -> np.ndarray:
    image = to_uint8(image)
    mask_rgb = colorize_mask(mask, ignore_index)
    foreground = (mask > 0) & (mask != ignore_index)
    output = image.copy()
    output[foreground] = ((1.0 - alpha) * output[foreground] + alpha * mask_rgb[foreground]).astype(np.uint8)
    return output


def draw_patch_grid(
    image: np.ndarray,
    coords: list[PatchCoord],
    selected: PatchCoord,
    max_side: int = 1400,
) -> np.ndarray:
    pil = Image.fromarray(to_uint8(image))
    scale = min(1.0, max_side / max(pil.size))
    if scale < 1.0:
        pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(pil)
    for coord in coords:
        box = [coord.x * scale, coord.y * scale, (coord.x + coord.w) * scale, (coord.y + coord.h) * scale]
        draw.rectangle(box, outline=(255, 255, 255), width=1)
    selected_box = [
        selected.x * scale,
        selected.y * scale,
        (selected.x + selected.w) * scale,
        (selected.y + selected.h) * scale,
    ]
    draw.rectangle(selected_box, outline=(230, 57, 70), width=5)
    return np.asarray(pil)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image)).save(path)


def add_label(image: Image.Image, label: str, height: int = 34) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + height), "white")
    output.paste(image, (0, height))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    draw.text((8, 10), label, fill=(20, 20, 20), font=font)
    return output


def resize_for_montage(image: np.ndarray, width: int) -> Image.Image:
    pil = Image.fromarray(to_uint8(image)).convert("RGB")
    height = max(1, round(pil.height * width / pil.width))
    return pil.resize((width, height), Image.Resampling.BILINEAR)


def build_montage(items: list[tuple[str, np.ndarray]], output_width: int) -> Image.Image:
    columns = 3
    cell_width = output_width // columns
    labeled = [add_label(resize_for_montage(image, cell_width), label) for label, image in items]
    rows = [labeled[idx : idx + columns] for idx in range(0, len(labeled), columns)]
    row_images: list[Image.Image] = []
    for row in rows:
        row_height = max(item.height for item in row)
        canvas = Image.new("RGB", (cell_width * columns, row_height), "white")
        for col, item in enumerate(row):
            canvas.paste(item, (col * cell_width, 0))
        row_images.append(canvas)
    total_height = sum(row.height for row in row_images)
    montage = Image.new("RGB", (cell_width * columns, total_height), "white")
    y = 0
    for row in row_images:
        montage.paste(row, (0, y))
        y += row.height
    return montage


def find_defect_patch(
    config: dict,
    split: str,
    sample_id: str | None,
    patch_size: int,
    overlap: float,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[PatchCoord], PatchCoord, Path]:
    data_cfg = config["dataset"]
    pairs = find_pairs(
        data_cfg["root"],
        data_cfg.get("image_dir", "images"),
        data_cfg.get("mask_dir", "annotations"),
        split,
        data_cfg,
    )
    if sample_id:
        sample_stem = Path(sample_id).stem
        pairs = [pair for pair in pairs if pair.image_path.stem == sample_stem or pair.image_path.name == sample_id]
        if not pairs:
            raise ValueError(f"No sample matching {sample_id!r} was found in split {split!r}.")

    ignore_index = int(data_cfg.get("ignore_index", 255))
    best: tuple[int, np.ndarray, np.ndarray, list[PatchCoord], PatchCoord, Path] | None = None
    for pair in pairs[:max_samples]:
        image = load_rgb(pair.image_path)
        mask = load_mask(pair.mask_path)
        coords = compute_patch_grid(mask.shape[0], mask.shape[1], patch_size, overlap)
        for coord in coords:
            patch_mask = extract_patch(mask, coord, patch_size, pad_value=ignore_index)
            foreground = int(((patch_mask > 0) & (patch_mask != ignore_index)).sum())
            if foreground <= 0:
                continue
            if best is None or foreground > best[0]:
                best = (foreground, image, mask, coords, coord, pair.image_path)
    if best is None:
        raise RuntimeError(f"No defect patch found in the first {max_samples} samples of split {split!r}.")
    _, image, mask, coords, coord, image_path = best
    return image, mask, coords, coord, image_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["dataset"]
    patch_size = int(args.patch_size or data_cfg.get("patch_size", 512))
    overlap = float(args.overlap if args.overlap is not None else data_cfg.get("overlap", 0.5))
    ignore_index = int(data_cfg.get("ignore_index", 255))
    output_dir = Path(args.output_dir)

    image, mask, coords, coord, image_path = find_defect_patch(
        config=config,
        split=args.split,
        sample_id=args.sample_id,
        patch_size=patch_size,
        overlap=overlap,
        max_samples=args.max_samples,
    )
    image_patch = extract_patch(image, coord, patch_size, pad_value=0)
    mask_patch = extract_patch(mask, coord, patch_size, pad_value=ignore_index)

    feature_cfg = FeatureConfig.from_dict(
        {
            **config.get("features", {}),
            "enabled": True,
            "use_gabor": True,
            "use_lbp": True,
            "use_clahe": True,
            "compute_device": "cpu",
        }
    )
    gray = rgb_to_gray(image_patch)
    gabor = compute_gabor(gray, feature_cfg)
    lbp = compute_lbp(gray, feature_cfg.lbp_radius, feature_cfg.lbp_points)
    clahe = compute_clahe_like(gray, feature_cfg.clahe_tile_grid_size)

    high_res = image
    patch_grid = draw_patch_grid(image, coords, coord)
    rgb_patch = image_patch
    patch_mask = colorize_mask(mask_patch, ignore_index)
    patch_overlay = overlay_mask(image_patch, mask_patch, ignore_index=ignore_index)
    full_mask = colorize_mask(mask, ignore_index)
    full_overlay = overlay_mask(image, mask, ignore_index=ignore_index)

    save_image(output_dir / "01_high_resolution_ccm_image.png", high_res)
    save_image(output_dir / "02_overlapping_512x512_patches.png", patch_grid)
    save_image(output_dir / "03_rgb_defect_patch.png", rgb_patch)
    for idx, theta in enumerate(feature_cfg.gabor_orientations):
        save_image(output_dir / f"04_gabor_orientation_{theta:03d}.png", gray_to_rgb(gabor[idx]))
    save_image(output_dir / "05_lbp.png", gray_to_rgb(lbp))
    save_image(output_dir / "06_clahe.png", gray_to_rgb(clahe))
    save_image(output_dir / "07_patch_defect_mask_gt.png", patch_mask)
    save_image(output_dir / "08_patch_defect_overlay_gt.png", patch_overlay)
    save_image(output_dir / "09_full_resolution_segmentation_mask_gt.png", full_mask)
    save_image(output_dir / "10_full_resolution_segmentation_overlay_gt.png", full_overlay)

    montage_items = [
        ("High-resolution CCM image", high_res),
        ("Overlapping 512x512 patches", patch_grid),
        ("RGB defect patch", rgb_patch),
        ("Gabor 0 deg", gabor[0]),
        ("Gabor 30 deg", gabor[1]),
        ("Gabor 60 deg", gabor[2]),
        ("Gabor 90 deg", gabor[3]),
        ("Gabor 120 deg", gabor[4]),
        ("Gabor 150 deg", gabor[5]),
        ("LBP", lbp),
        ("CLAHE", clahe),
        ("Patch defect mask (GT)", patch_mask),
        ("Patch defect overlay (GT)", patch_overlay),
        ("Full-resolution mask (GT)", full_mask),
        ("Full-resolution overlay (GT)", full_overlay),
    ]
    montage = build_montage(montage_items, args.montage_width)
    montage.save(output_dir / "tf_mobilevit_architecture_visuals_montage.png")

    metadata = output_dir / "metadata.txt"
    metadata.write_text(
        "\n".join(
            [
                f"image_path: {image_path}",
                f"image_shape: {image.shape}",
                f"patch_coord_yxhw: {(coord.y, coord.x, coord.h, coord.w)}",
                f"patch_size: {patch_size}",
                f"overlap: {overlap}",
                f"foreground_pixels_in_patch: {int(((mask_patch > 0) & (mask_patch != ignore_index)).sum())}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated architecture visuals in {output_dir}")
    print(f"Selected sample: {image_path}")
    print(f"Selected patch y/x/h/w: {(coord.y, coord.x, coord.h, coord.w)}")


if __name__ == "__main__":
    main()
