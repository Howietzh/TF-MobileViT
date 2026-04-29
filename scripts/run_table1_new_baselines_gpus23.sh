#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export RUN_ROOT=${RUN_ROOT:-runs/table1}
export METHODS=${METHODS:-"segformer_b0"}
export GPUS=${GPUS:-2}
export MASTER_PORT=${MASTER_PORT:-29502}

bash "$(dirname "$0")/run_experiment_group.sh"
