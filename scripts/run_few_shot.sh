#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/default.yaml}
DATA_ROOT=${DATA_ROOT:-data/CCM_Defect}
RUN_ROOT=${RUN_ROOT:-runs/few_shot}
SUBSETS=${SUBSETS:-"1 2 3 4 5 6 7"}
EXPERIMENTS=${EXPERIMENTS:-"tf_mobilevit mobilevit_no_trad"}
GPUS=${GPUS:-1}
FEATURE_COMPUTE_DEVICE=${FEATURE_COMPUTE_DEVICE:-gpu}
USE_CLAHE=${USE_CLAHE:-false}
PYTHON=${PYTHON:-python}

mkdir -p "${RUN_ROOT}/configs"

echo "Config: ${CONFIG}"
echo "Data root: ${DATA_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "Subsets: ${SUBSETS}"
echo "Experiments: ${EXPERIMENTS}"
echo "GPUs: ${GPUS}"
echo "Feature compute device: ${FEATURE_COMPUTE_DEVICE}"
echo "Use CLAHE: ${USE_CLAHE}"

for subset in ${SUBSETS}; do
  split_file="train_subset_${subset}.txt"
  split_path="${DATA_ROOT}/${split_file}"

  if [[ ! -f "${split_path}" ]]; then
    echo "Skipping missing split file: ${split_path}" >&2
    continue
  fi

  for experiment in ${EXPERIMENTS}; do
    output_dir="${RUN_ROOT}/${experiment}/subset_${subset}"
    generated_config="${RUN_ROOT}/configs/${experiment}_subset_${subset}.yaml"

    echo
    echo "==== Running ${experiment} | few-shot subset ${subset}: ${split_file} ===="

    "${PYTHON}" - "${CONFIG}" "${generated_config}" "${DATA_ROOT}" "${split_file}" "${output_dir}" "${experiment}" "${FEATURE_COMPUTE_DEVICE}" "${USE_CLAHE}" <<'PY'
import sys
from pathlib import Path

import yaml

src, dst, data_root, split_file, output_dir, experiment, feature_compute_device, use_clahe = sys.argv[1:]
use_clahe = use_clahe.lower() in {"1", "true", "yes", "y"}
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("dataset", {})
cfg["dataset"]["root"] = data_root
cfg["dataset"]["train_split_file"] = split_file

cfg.setdefault("train", {})
cfg["train"]["output_dir"] = output_dir

cfg.setdefault("model", {})
cfg["model"]["pretrained"] = True

if experiment == "mobilevit_no_trad":
    cfg.setdefault("features", {})
    cfg["features"]["enabled"] = False
    cfg["features"]["use_gabor"] = False
    cfg["features"]["use_lbp"] = False
    cfg["features"]["use_clahe"] = False
    cfg["model"]["use_tfgam"] = False
elif experiment == "tf_mobilevit":
    cfg.setdefault("features", {})
    cfg["features"]["enabled"] = True
    cfg["features"]["compute_device"] = feature_compute_device
    cfg["features"]["gpu_features"] = {"gabor": True, "lbp": True, "clahe": False}
    cfg["features"]["use_gabor"] = True
    cfg["features"]["use_lbp"] = True
    cfg["features"]["use_clahe"] = use_clahe and feature_compute_device != "gpu"
    cfg["model"]["use_tfgam"] = True
else:
    raise ValueError(f"Unknown experiment: {experiment}")

dst_path = Path(dst)
dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

    if [[ "${GPUS}" -gt 1 ]]; then
      "${PYTHON}" -m torch.distributed.run --nproc_per_node="${GPUS}" -m tf_mobilevit.cli train --config "${generated_config}"
    else
      "${PYTHON}" -m tf_mobilevit.cli train --config "${generated_config}"
    fi
  done
done

echo
echo "Few-shot runs finished. Results are under ${RUN_ROOT}/{tf_mobilevit,mobilevit_no_trad}/subset_*"
