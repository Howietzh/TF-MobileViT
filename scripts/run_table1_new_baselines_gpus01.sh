#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export RUN_ROOT=${RUN_ROOT:-runs/table1}
export METHODS=${METHODS:-"unet deeplabv3plus"}
export GPUS=${GPUS:-2}
export MASTER_PORT=${MASTER_PORT:-29501}

bash "$(dirname "$0")/run_experiment_group.sh"
