#!/usr/bin/env bash
# Reproduce the information-source ablations in Section 4.4 / Appendix A.7.
# Toggles which subsets of structural, semantic, and interaction signals
# the ranker has access to.

set -euo pipefail

for cfg in config/ablation_information/*.yaml \
           config/ablation_size/*.yaml \
           config/ablation_family/*.yaml; do
    echo "[$(date +%T)] >>> running ${cfg}"
    python src/main.py --config "${cfg}"
done
