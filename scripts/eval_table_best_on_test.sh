#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
TABLES=${TABLES:-"table1 table2"}
METHODS=${METHODS:-}
EXISTING_ROOT=${EXISTING_ROOT:-runs/few_shot}
EXISTING_SUBSET=${EXISTING_SUBSET:-6}
TABLE1_ROOT=${TABLE1_ROOT:-runs/table1}
TABLE2_ROOT=${TABLE2_ROOT:-runs/table2_ablation}
TEST_SPLIT_FILE=${TEST_SPLIT_FILE:-test.txt}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-}
EVAL_LOG_INTERVAL=${EVAL_LOG_INTERVAL:-10}
EVAL_AMP=${EVAL_AMP:-true}
CACHE_EVAL_IMAGES=${CACHE_EVAL_IMAGES:-true}
CACHE_EVAL_MAX_ITEMS=${CACHE_EVAL_MAX_ITEMS:-4}
EVAL_CONFIG_DIR=${EVAL_CONFIG_DIR:-runs/table_eval_configs}
RUN_SUMMARY=${RUN_SUMMARY:-1}

mkdir -p "${EVAL_CONFIG_DIR}"

echo "Tables: ${TABLES}"
echo "Methods: ${METHODS:-all methods in selected tables}"
echo "Existing root: ${EXISTING_ROOT}"
echo "Existing subset: subset_${EXISTING_SUBSET}"
echo "Table1 root: ${TABLE1_ROOT}"
echo "Table2 root: ${TABLE2_ROOT}"
echo "Test split: ${TEST_SPLIT_FILE}"
echo "Eval batch size override: ${EVAL_BATCH_SIZE:-config default}"
echo "Eval log interval: ${EVAL_LOG_INTERVAL}"
echo "Eval AMP: ${EVAL_AMP}"
echo "Cache eval images: ${CACHE_EVAL_IMAGES} (max items: ${CACHE_EVAL_MAX_ITEMS})"

emit_specs() {
  for table in ${TABLES}; do
    case "${table}" in
      table1)
        printf '%s|%s|%s\n' "table1" "unet" "${TABLE1_ROOT}/unet"
        printf '%s|%s|%s\n' "table1" "deeplabv3plus" "${TABLE1_ROOT}/deeplabv3plus"
        printf '%s|%s|%s\n' "table1" "segformer_b0" "${TABLE1_ROOT}/segformer_b0"
        printf '%s|%s|%s\n' "table1" "mobilevit_no_trad" "${EXISTING_ROOT}/mobilevit_no_trad/subset_${EXISTING_SUBSET}"
        printf '%s|%s|%s\n' "table1" "tf_mobilevit" "${EXISTING_ROOT}/tf_mobilevit/subset_${EXISTING_SUBSET}"
        ;;
      table2)
        printf '%s|%s|%s\n' "table2" "mobilevit_no_trad" "${EXISTING_ROOT}/mobilevit_no_trad/subset_${EXISTING_SUBSET}"
        printf '%s|%s|%s\n' "table2" "gabor" "${TABLE2_ROOT}/gabor"
        printf '%s|%s|%s\n' "table2" "gabor_lbp" "${TABLE2_ROOT}/gabor_lbp"
        printf '%s|%s|%s\n' "table2" "gabor_lbp_clahe" "${TABLE2_ROOT}/gabor_lbp_clahe"
        printf '%s|%s|%s\n' "table2" "tf_mobilevit" "${EXISTING_ROOT}/tf_mobilevit/subset_${EXISTING_SUBSET}"
        ;;
      *)
        echo "Unsupported table: ${table}. Use TABLES=\"table1 table2\"." >&2
        exit 2
        ;;
    esac
  done
}

method_selected() {
  local candidate="$1"
  if [[ -z "${METHODS}" ]]; then
    return 0
  fi
  for selected in ${METHODS}; do
    if [[ "${selected}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

evaluated_dirs="|"
matched_methods="|"
while IFS='|' read -r table method output_dir; do
  [[ -n "${table}" ]] || continue

  if ! method_selected "${method}"; then
    continue
  fi
  matched_methods="${matched_methods}${method}|"

  if [[ "${evaluated_dirs}" == *"|${output_dir}|"* ]]; then
    echo
    echo "==== Skipping duplicate eval for ${table}/${method}: ${output_dir} already evaluated in this run ===="
    continue
  fi
  evaluated_dirs="${evaluated_dirs}${output_dir}|"

  checkpoint="${output_dir}/best.pt"
  config="${output_dir}/config.yaml"
  eval_config="${EVAL_CONFIG_DIR}/${table}_${method}_test.yaml"
  log_file="${output_dir}/test_eval.log"

  echo
  echo "==== Evaluating ${table}/${method} on test set ===="

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
    cfg = yaml.safe_load(f) or {}

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
    echo "Table: ${table}"
    echo "Method: ${method}"
    echo "Run dir: ${output_dir}"
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
    echo "==== Test eval failed for ${table}/${method} with exit code ${status} ====" | tee -a "${log_file}" >&2
    exit "${status}"
  fi

  echo "==== Test eval finished at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${log_file}"
  echo "Metrics saved to: ${output_dir}/test_metrics.json"
done < <(emit_specs)

if [[ -n "${METHODS}" ]]; then
  for selected in ${METHODS}; do
    if [[ "${matched_methods}" != *"|${selected}|"* ]]; then
      echo "Warning: requested method was not found in selected tables: ${selected}" >&2
    fi
  done
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  echo
  echo "==== Summarizing table metrics ===="
  "${PYTHON}" scripts/summarize_table_results.py \
    --existing-root "${EXISTING_ROOT}" \
    --existing-subset "${EXISTING_SUBSET}" \
    --table1-root "${TABLE1_ROOT}" \
    --table2-root "${TABLE2_ROOT}"
fi
