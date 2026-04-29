#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export RUN_ROOT=${RUN_ROOT:-runs/table2_ablation}
export METHODS=${METHODS:-"gabor gabor_lbp gabor_lbp_clahe"}
export GPUS=${GPUS:-2}
export MASTER_PORT=${MASTER_PORT:-29503}

bash "$(dirname "$0")/run_experiment_group.sh"
