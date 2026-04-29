#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-runs/few_shot}
SUBSETS=${SUBSETS:-"1 2 3 4 5 6"}
EXPERIMENTS=${EXPERIMENTS:-"mobilevit_no_trad tf_mobilevit"}
PYTHON=${PYTHON:-python}
TEST_SPLIT_FILE=${TEST_SPLIT_FILE:-test.txt}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-}
EVAL_LOG_INTERVAL=${EVAL_LOG_INTERVAL:-10}
EVAL_AMP=${EVAL_AMP:-true}
CACHE_EVAL_IMAGES=${CACHE_EVAL_IMAGES:-true}
CACHE_EVAL_MAX_ITEMS=${CACHE_EVAL_MAX_ITEMS:-4}
EVAL_CONFIG_DIR=${EVAL_CONFIG_DIR:-"${RUN_ROOT}/eval_configs"}

echo "Run root: ${RUN_ROOT}"
echo "Subsets: ${SUBSETS}"
echo "Experiments: ${EXPERIMENTS}"
echo "Test split: ${TEST_SPLIT_FILE}"
echo "Eval batch size override: ${EVAL_BATCH_SIZE:-config default}"
echo "Eval log interval: ${EVAL_LOG_INTERVAL}"
echo "Eval AMP: ${EVAL_AMP}"
echo "Cache eval images: ${CACHE_EVAL_IMAGES} (max items: ${CACHE_EVAL_MAX_ITEMS})"

mkdir -p "${EVAL_CONFIG_DIR}"

for experiment in ${EXPERIMENTS}; do
  for subset in ${SUBSETS}; do
    output_dir="${RUN_ROOT}/${experiment}/subset_${subset}"
    checkpoint="${output_dir}/best.pt"
    config="${output_dir}/config.yaml"
    eval_config="${EVAL_CONFIG_DIR}/${experiment}_subset_${subset}_test.yaml"
    log_file="${output_dir}/test_eval.log"

    echo
    echo "==== Evaluating ${experiment} subset_${subset} on test set ===="

    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing checkpoint: ${checkpoint}" >&2
      continue
    fi

    if [[ ! -f "${config}" ]]; then
      echo "Missing config: ${config}" >&2
      continue
    fi

    "${PYTHON}" - "${config}" "${eval_config}" "${output_dir}" "${TEST_SPLIT_FILE}" \
      "${EVAL_BATCH_SIZE}" "${EVAL_LOG_INTERVAL}" "${EVAL_AMP}" \
      "${CACHE_EVAL_IMAGES}" "${CACHE_EVAL_MAX_ITEMS}" <<'PY'
import sys
from pathlib import Path

import yaml

(
    src,
    dst,
    output_dir,
    test_split_file,
    eval_batch_size,
    eval_log_interval,
    eval_amp,
    cache_eval_images,
    cache_eval_max_items,
) = sys.argv[1:]

with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("train", {})
cfg["train"]["output_dir"] = output_dir

cfg.setdefault("dataset", {})
cfg["dataset"]["test_split_file"] = test_split_file
cfg["dataset"]["cache_eval_images"] = cache_eval_images.lower() in {"1", "true", "yes", "y"}
cfg["dataset"]["cache_eval_max_items"] = int(cache_eval_max_items)

cfg.setdefault("eval", {})
if eval_batch_size.strip():
    cfg["eval"]["batch_size"] = int(eval_batch_size)
cfg["eval"]["log_interval"] = int(eval_log_interval)
cfg["eval"]["amp"] = eval_amp.lower() in {"1", "true", "yes", "y"}

dst_path = Path(dst)
dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

    {
      echo "==== Test eval started at $(date '+%Y-%m-%d %H:%M:%S') ===="
      echo "Experiment: ${experiment}"
      echo "Subset: subset_${subset}"
      echo "Source config: ${config}"
      echo "Eval config: ${eval_config}"
      echo "Checkpoint: ${checkpoint}"
      echo "Test split: ${TEST_SPLIT_FILE}"
      echo "Eval batch size override: ${EVAL_BATCH_SIZE:-config default}"
      echo "Eval log interval: ${EVAL_LOG_INTERVAL}"
      echo "Eval AMP: ${EVAL_AMP}"
      echo "Cache eval images: ${CACHE_EVAL_IMAGES} (max items: ${CACHE_EVAL_MAX_ITEMS})"
    } | tee "${log_file}"

    set +e
    "${PYTHON}" -m tf_mobilevit.cli eval \
      --config "${eval_config}" \
      --checkpoint "${checkpoint}" \
      --test-split-file "${TEST_SPLIT_FILE}" \
      2>&1 | tee -a "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if [[ "${status}" -ne 0 ]]; then
      echo "==== Test eval failed for ${experiment} subset_${subset} with exit code ${status} ====" | tee -a "${log_file}" >&2
      exit "${status}"
    fi

    echo "==== Test eval finished at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${log_file}"
    echo "Metrics saved to: ${output_dir}/test_metrics.json"
  done
done
