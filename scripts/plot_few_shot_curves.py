from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUBSET_TO_IMAGES_PER_CLASS = {
    1: 5,
    2: 10,
    3: 20,
    4: 40,
    5: 60,
    6: 80,
}

METHODS = {
    "mobilevit_no_trad": "MobileViT",
    "tf_mobilevit": "TF-MobileViT",
}

METRICS = {
    "foreground_dice": "Foreground Dice",
    "foreground_recall": "Foreground Recall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot few-shot fgDice/fgRecall curves from experiment outputs.")
    parser.add_argument("--run-root", default="runs/few_shot", help="Root directory containing few-shot runs.")
    parser.add_argument("--output-dir", default="figures", help="Directory for curve figures.")
    parser.add_argument("--subsets", default="1,2,3,4,5,6", help="Comma-separated subset IDs to include.")
    parser.add_argument("--format", default="pdf,png", help="Comma-separated output formats, e.g. pdf,png.")
    parser.add_argument("--percent", action="store_true", default=True, help="Plot metrics as percentages.")
    parser.add_argument("--no-percent", action="store_false", dest="percent", help="Plot metrics in [0, 1].")
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_metrics_json(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    values: dict[str, float] = {}
    for key in METRICS:
        value = to_float(raw.get(key))
        if value is not None:
            values[key] = value
    return values or None


def read_history_csv(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None

    best_row: dict[str, str] | None = None
    best_value = float("-inf")
    for row in rows:
        value = to_float(row.get("foreground_dice"))
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best_row = row
    if best_row is None:
        best_row = rows[-1]

    values: dict[str, float] = {}
    for key in METRICS:
        value = to_float(best_row.get(key))
        if value is not None:
            values[key] = value
    return values or None


def load_run_metrics(run_root: Path, method: str, subset: int) -> dict[str, float] | None:
    run_dir = run_root / method / f"subset_{subset}"
    metrics = read_metrics_json(run_dir / "metrics.json")
    if metrics is not None:
        return metrics
    return read_history_csv(run_dir / "history.csv")


def collect_results(run_root: Path, subsets: list[int]) -> dict[str, dict[str, list[float | None]]]:
    results: dict[str, dict[str, list[float | None]]] = {
        method: {metric: [] for metric in METRICS} for method in METHODS
    }
    for method in METHODS:
        for subset in subsets:
            metrics = load_run_metrics(run_root, method, subset)
            if metrics is None:
                print(f"Warning: missing metrics for {method}/subset_{subset}")
            for metric in METRICS:
                results[method][metric].append(metrics.get(metric) if metrics else None)
    return results


def plot_metric(ax: Any, xs: list[int], results: dict[str, dict[str, list[float | None]]], metric: str, percent: bool) -> None:
    styles = {
        "mobilevit_no_trad": {"color": "#6b7280", "linestyle": "--", "marker": "o"},
        "tf_mobilevit": {"color": "#2563eb", "linestyle": "-", "marker": "s"},
    }
    for method, label in METHODS.items():
        values = results[method][metric]
        plot_x = [x for x, value in zip(xs, values, strict=True) if value is not None]
        plot_y = [value * 100.0 if percent else value for value in values if value is not None]
        if not plot_x:
            continue
        ax.plot(plot_x, plot_y, label=label, linewidth=2.2, markersize=6, **styles[method])
    ax.set_xlabel("Training images per class")
    ax.set_ylabel(f"{METRICS[metric]} (%)" if percent else METRICS[metric])
    ax.set_xticks(xs)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False)


def save_all_formats(fig: Any, output_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=300)


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it with `pip install matplotlib`.") from exc

    subsets = [int(item.strip()) for item in args.subsets.split(",") if item.strip()]
    unknown = [subset for subset in subsets if subset not in SUBSET_TO_IMAGES_PER_CLASS]
    if unknown:
        raise ValueError(f"Unsupported subset IDs {unknown}. Supported subset IDs are {sorted(SUBSET_TO_IMAGES_PER_CLASS)}.")
    xs = [SUBSET_TO_IMAGES_PER_CLASS[subset] for subset in subsets]
    formats = [item.strip().lstrip(".") for item in args.format.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(Path(args.run_root), subsets)

    for metric, title in METRICS.items():
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        plot_metric(ax, xs, results, metric, args.percent)
        ax.set_title(title)
        save_all_formats(fig, output_dir, f"few_shot_{'fgdice' if metric == 'foreground_dice' else 'fgrecall'}_curve", formats)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharex=True)
    for ax, metric in zip(axes, METRICS, strict=True):
        plot_metric(ax, xs, results, metric, args.percent)
        ax.set_title(METRICS[metric])
    fig.tight_layout()
    save_all_formats(fig, output_dir, "few_shot_curves", formats)
    plt.close(fig)
    print(f"Saved few-shot curves to {output_dir}")


if __name__ == "__main__":
    main()
