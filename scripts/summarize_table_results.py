from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TABLE1_METHODS = [
    ("U-Net", "runs/table1/unet"),
    ("DeepLabv3+", "runs/table1/deeplabv3plus"),
    ("SegFormer-B0", "runs/table1/segformer_b0"),
    ("MobileViT baseline", "runs/few_shot/mobilevit_no_trad/subset_6"),
    ("TF-MobileViT", "runs/few_shot/tf_mobilevit/subset_6"),
]

TABLE2_METHODS = [
    ("MobileViT baseline", "runs/few_shot/mobilevit_no_trad/subset_6"),
    ("+ Gabor + TFGAM", "runs/table2_ablation/gabor"),
    ("+ Gabor + LBP + TFGAM", "runs/table2_ablation/gabor_lbp"),
    ("+ Gabor + LBP + CLAHE + TFGAM", "runs/table2_ablation/gabor_lbp_clahe"),
    ("TF-MobileViT", "runs/few_shot/tf_mobilevit/subset_6"),
]

SUMMARY_FIELDS = [
    "method",
    "source",
    "status",
    "params",
    "fps",
    "mean_dice",
    "foreground_dice",
    "mean_iou",
    "foreground_iou",
    "foreground_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize table1/table2 experiment metrics.")
    parser.add_argument("--existing-root", default="runs/few_shot", help="Root containing existing MobileViT/TF-MobileViT few-shot runs.")
    parser.add_argument("--existing-subset", type=int, default=6, help="Few-shot subset ID to reuse for existing results.")
    parser.add_argument("--table1-root", default="runs/table1", help="Root containing table1 new baseline runs.")
    parser.add_argument("--table2-root", default="runs/table2_ablation", help="Root containing table2 new ablation runs.")
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def read_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    best = max(rows, key=lambda row: to_float(row.get("foreground_dice")) or -1.0)
    return best


def load_metrics(run_dir: Path) -> tuple[str, dict[str, Any]]:
    test_metrics = read_json(run_dir / "test_metrics.json")
    metrics = read_json(run_dir / "metrics.json")
    history = read_history(run_dir / "history.csv")
    merged: dict[str, Any] = {}
    merged.update(history)
    merged.update(metrics)
    merged.update(test_metrics)
    if not merged:
        return "missing", {}
    return "ok", merged


def make_row(method: str, source: Path) -> dict[str, Any]:
    status, metrics = load_metrics(source)
    row: dict[str, Any] = {"method": method, "source": str(source), "status": status}
    for field in SUMMARY_FIELDS[3:]:
        value = to_float(metrics.get(field))
        row[field] = f"{value:.6f}" if value is not None else ""
    return row


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    existing_root = Path(args.existing_root)
    table1_root = Path(args.table1_root)
    table2_root = Path(args.table2_root)
    existing_subset = f"subset_{args.existing_subset}"

    table1_sources = [
        ("U-Net", table1_root / "unet"),
        ("DeepLabv3+", table1_root / "deeplabv3plus"),
        ("SegFormer-B0", table1_root / "segformer_b0"),
        ("MobileViT baseline", existing_root / "mobilevit_no_trad" / existing_subset),
        ("TF-MobileViT", existing_root / "tf_mobilevit" / existing_subset),
    ]
    table2_sources = [
        ("MobileViT baseline", existing_root / "mobilevit_no_trad" / existing_subset),
        ("+ Gabor + TFGAM", table2_root / "gabor"),
        ("+ Gabor + LBP + TFGAM", table2_root / "gabor_lbp"),
        ("+ Gabor + LBP + CLAHE + TFGAM", table2_root / "gabor_lbp_clahe"),
        ("TF-MobileViT", existing_root / "tf_mobilevit" / existing_subset),
    ]

    table1_rows = [make_row(method, source) for method, source in table1_sources]
    table2_rows = [make_row(method, source) for method, source in table2_sources]
    for row in table1_rows + table2_rows:
        if row["status"] != "ok":
            print(f"Warning: missing metrics for {row['method']} at {row['source']}")

    write_summary(table1_root / "table1_summary.csv", table1_rows)
    write_summary(table2_root / "table2_ablation_summary.csv", table2_rows)


if __name__ == "__main__":
    main()
