#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/default.yaml}
DATA_ROOT=${DATA_ROOT:-data/CCM_Defect}
RUN_ROOT=${RUN_ROOT:-runs/experiments}
METHODS=${METHODS:-}
TRAIN_SPLIT_FILE=${TRAIN_SPLIT_FILE:-train_subset_6.txt}
GPUS=${GPUS:-2}
PYTHON=${PYTHON:-python}
FORCE=${FORCE:-0}
FEATURE_COMPUTE_DEVICE=${FEATURE_COMPUTE_DEVICE:-gpu}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

if [[ -z "${METHODS}" ]]; then
  echo "METHODS must be set, e.g. METHODS=\"unet deeplabv3plus\"" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/configs"

echo "Config: ${CONFIG}"
echo "Data root: ${DATA_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "Methods: ${METHODS}"
echo "Train split: ${TRAIN_SPLIT_FILE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "GPUs per run: ${GPUS}"
echo "DDP master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Force: ${FORCE}"

for method in ${METHODS}; do
  output_dir="${RUN_ROOT}/${method}"
  generated_config="${RUN_ROOT}/configs/${method}.yaml"
  train_log="${output_dir}/train.log"
  eval_log="${output_dir}/eval.log"
  mkdir -p "${output_dir}"

  echo
  echo "==== Preparing ${method} ===="
  "${PYTHON}" - "${CONFIG}" "${generated_config}" "${DATA_ROOT}" "${TRAIN_SPLIT_FILE}" "${output_dir}" "${method}" "${FEATURE_COMPUTE_DEVICE}" <<'PY'
import sys
from pathlib import Path

import yaml

src, dst, data_root, train_split_file, output_dir, method, feature_compute_device = sys.argv[1:]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("dataset", {})
cfg["dataset"]["root"] = data_root
cfg["dataset"]["train_split_file"] = train_split_file

cfg.setdefault("train", {})
cfg["train"]["output_dir"] = output_dir

cfg.setdefault("model", {})
cfg.setdefault("features", {})

def disable_features() -> None:
    cfg["features"]["enabled"] = False
    cfg["features"]["use_gabor"] = False
    cfg["features"]["use_lbp"] = False
    cfg["features"]["use_clahe"] = False

def gpu_features(use_gabor: bool, use_lbp: bool) -> None:
    cfg["features"]["enabled"] = use_gabor or use_lbp
    cfg["features"]["compute_device"] = feature_compute_device
    cfg["features"]["gpu_features"] = {"gabor": use_gabor, "lbp": use_lbp, "clahe": False}
    cfg["features"]["use_gabor"] = use_gabor
    cfg["features"]["use_lbp"] = use_lbp
    cfg["features"]["use_clahe"] = False

if method in {"unet", "deeplabv3plus", "segformer_b0"}:
    disable_features()
    cfg["model"]["name"] = method
    cfg["model"]["use_tfgam"] = False
    cfg["model"]["direct_trad_fusion"] = False
    if method == "deeplabv3plus":
        cfg["model"]["deeplab_pretrained_path"] = cfg["model"].get("deeplab_pretrained_path", "weights/resnet50-11ad3fa6.pth")
    if method == "segformer_b0":
        cfg["model"]["segformer_pretrained_path"] = cfg["model"].get("segformer_pretrained_path", "weights/nvidia_mit_b0")
elif method == "gabor":
    gpu_features(True, False)
    cfg["model"]["name"] = "gabor"
    cfg["model"]["use_tfgam"] = False
    cfg["model"]["direct_trad_fusion"] = True
elif method == "gabor_lbp":
    gpu_features(True, True)
    cfg["model"]["name"] = "gabor_lbp"
    cfg["model"]["use_tfgam"] = False
    cfg["model"]["direct_trad_fusion"] = True
elif method == "gabor_lbp_clahe":
    cfg["features"]["enabled"] = True
    cfg["features"]["compute_device"] = "cpu"
    cfg["features"]["use_gabor"] = True
    cfg["features"]["use_lbp"] = True
    cfg["features"]["use_clahe"] = True
    cfg["model"]["name"] = "gabor_lbp_clahe"
    cfg["model"]["use_tfgam"] = False
    cfg["model"]["direct_trad_fusion"] = True
elif method == "mobilevit_no_trad":
    disable_features()
    cfg["model"]["name"] = "mobilevit_no_trad"
    cfg["model"]["use_tfgam"] = False
    cfg["model"]["direct_trad_fusion"] = False
elif method == "tf_mobilevit":
    gpu_features(True, True)
    cfg["model"]["name"] = "tf_mobilevit"
    cfg["model"]["use_tfgam"] = True
    cfg["model"]["direct_trad_fusion"] = False
else:
    raise ValueError(f"Unsupported method: {method}")

dst_path = Path(dst)
dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  if [[ -f "${output_dir}/best.pt" && "${FORCE}" != "1" ]]; then
    echo "==== Skipping ${method}: ${output_dir}/best.pt exists. Set FORCE=1 to rerun. ====" | tee -a "${train_log}"
  else
    echo "==== Training ${method} started at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee "${train_log}"
    set +e
    "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node="${GPUS}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      -m tf_mobilevit.cli train --config "${generated_config}" 2>&1 | tee -a "${train_log}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "${status}" -ne 0 ]]; then
      echo "==== Training ${method} failed with exit code ${status} at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${train_log}" >&2
      exit "${status}"
    fi
    echo "==== Training ${method} finished at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${train_log}"
  fi

  if [[ -f "${output_dir}/best.pt" ]]; then
    echo "==== Evaluating ${method} started at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee "${eval_log}"
    set +e
    "${PYTHON}" -m tf_mobilevit.cli eval --config "${generated_config}" --checkpoint "${output_dir}/best.pt" 2>&1 | tee -a "${eval_log}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "${status}" -ne 0 ]]; then
      echo "==== Evaluation ${method} failed with exit code ${status} at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${eval_log}" >&2
      exit "${status}"
    fi
    echo "==== Evaluating ${method} finished at $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${eval_log}"
  else
    echo "No checkpoint found for ${method}: ${output_dir}/best.pt" >&2
    exit 1
  fi
done
