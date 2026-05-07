#!/usr/bin/env bash
# Reproduce the loss-objective ablation in Section 4.4 of the paper.
# Each variant trains a separate MLPMetricFull model with a different
# combination of listwise / pairwise / pointwise objectives.

set -euo pipefail

CONFIGS=(
    "config/method_ablation/MLPMetricFull_listwise.yaml"
    "config/method_ablation/MLPMetricFull_pairwise.yaml"
    "config/method_ablation/MLPMetricFull_listwise_pairwise.yaml"
    "config/method_ablation/MLPMetricFull_listwise_pointwise.yaml"
    "config/method_ablation/MLPMetricFull_pairwise_pointwise.yaml"
    "config/FinalModel_unified_augmented.yaml"  # full ensemble (L+P+Pt)
)

for cfg in "${CONFIGS[@]}"; do
    echo "[$(date +%T)] >>> running ${cfg}"
    python src/main.py --config "${cfg}"
done
