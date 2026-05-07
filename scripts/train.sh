#!/usr/bin/env bash
# Train the full ModelLens model on the unified leaderboard corpus.
#
# Usage:
#   bash scripts/train.sh                          # single-GPU
#   USE_DDP=1 bash scripts/train.sh                # multi-GPU (set device_ids in YAML)
#
# Outputs:
#   checkpoint/mlp/<data_name>/<trail_name>/best.pt
#   log/mlp/<data_name>/<trail_name>/train.log

set -euo pipefail

CONFIG=${CONFIG:-config/FinalModel_unified_augmented.yaml}

if [[ "${USE_DDP:-0}" == "1" ]]; then
    NPROC=${NPROC:-4}
    torchrun --nproc_per_node="${NPROC}" src/main.py --config "${CONFIG}"
else
    python src/main.py --config "${CONFIG}"
fi
