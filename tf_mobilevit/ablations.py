from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .engine import train
from .utils import write_metrics


ABLATIONS = [
    ("mobilevit_baseline", False, False, False, False),
    ("gabor", True, False, False, False),
    ("gabor_lbp", True, True, False, False),
    ("gabor_lbp_clahe", True, True, True, False),
    ("tf_mobilevit", True, True, True, True),
]


def run_ablations(config: dict[str, Any]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    base_output = Path(config["train"]["output_dir"])
    for name, use_gabor, use_lbp, use_clahe, use_tfgam in ABLATIONS:
        cfg = deepcopy(config)
        cfg["features"]["enabled"] = any([use_gabor, use_lbp, use_clahe])
        cfg["features"]["use_gabor"] = use_gabor
        cfg["features"]["use_lbp"] = use_lbp
        cfg["features"]["use_clahe"] = use_clahe
        cfg["model"]["use_tfgam"] = use_tfgam
        cfg["train"]["output_dir"] = str(base_output / name)
        metrics = train(cfg)
        rows.append({"method": name, **metrics})
    write_metrics(base_output / "ablation_summary.csv", rows)
    return rows

