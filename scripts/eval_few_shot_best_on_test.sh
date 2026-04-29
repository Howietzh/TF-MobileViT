#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-runs/few_shot}
SUBSETS=${SUBSETS:-"1 2 3 4 5 6"}
EXPERIMENTS=${EXPERIMENTS:-"mobilevit_no_trad tf_mobilevit"}
PYTHON=${PYTHON:-python}
TEST_SPLIT_FILE=${TEST_SPLIT_FILE:-test.txt}

echo "Run root: ${RUN_ROOT}"
echo "Subsets: ${SUBSETS}"
echo "Experiments: ${EXPERIMENTS}"
echo "Test split: ${TEST_SPLIT_FILE}"

for experiment in ${EXPERIMENTS}; do
  for subset in ${SUBSETS}; do
    output_dir="${RUN_ROOT}/${experiment}/subset_${subset}"
    checkpoint="${output_dir}/best.pt"
    config="${output_dir}/config.yaml"
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

    {
      echo "==== Test eval started at $(date '+%Y-%m-%d %H:%M:%S') ===="
      echo "Experiment: ${experiment}"
      echo "Subset: subset_${subset}"
      echo "Config: ${config}"
      echo "Checkpoint: ${checkpoint}"
      echo "Test split: ${TEST_SPLIT_FILE}"
    } | tee "${log_file}"

    set +e
    "${PYTHON}" -m tf_mobilevit.cli eval \
      --config "${config}" \
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
