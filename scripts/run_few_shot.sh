#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/default.yaml}
DATA_ROOT=${DATA_ROOT:-data/CCM_Defect}
RUN_ROOT=${RUN_ROOT:-runs/few_shot}
SUBSETS=${SUBSETS:-"1 2 3 4 5 6 7"}
PYTHON=${PYTHON:-python}

mkdir -p "${RUN_ROOT}/configs"

echo "Config: ${CONFIG}"
echo "Data root: ${DATA_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "Subsets: ${SUBSETS}"

for subset in ${SUBSETS}; do
  split_file="train_subset_${subset}.txt"
  split_path="${DATA_ROOT}/${split_file}"
  output_dir="${RUN_ROOT}/subset_${subset}"
  generated_config="${RUN_ROOT}/configs/subset_${subset}.yaml"

  if [[ ! -f "${split_path}" ]]; then
    echo "Missing split file: ${split_path}" >&2
    exit 1
  fi

  echo
  echo "==== Running few-shot subset ${subset}: ${split_file} ===="

  "${PYTHON}" - "${CONFIG}" "${generated_config}" "${DATA_ROOT}" "${split_file}" "${output_dir}" <<'PY'
import sys
from pathlib import Path

import yaml

src, dst, data_root, split_file, output_dir = sys.argv[1:]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("dataset", {})
cfg["dataset"]["root"] = data_root
cfg["dataset"]["train_split_file"] = split_file

cfg.setdefault("train", {})
cfg["train"]["output_dir"] = output_dir

dst_path = Path(dst)
dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  "${PYTHON}" -m tf_mobilevit.cli train --config "${generated_config}"
done

echo
echo "Few-shot runs finished. Results are under ${RUN_ROOT}/subset_*"

