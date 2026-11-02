#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
uaml-ab train --config "$project_root/configs/c05_m22_sample.yaml" \
  --epochs 2 --n-train 200 --output-dir "$project_root/results/demo"
uaml-ab predict \
  --checkpoint "$project_root/results/demo/Chen-2026-C05_M22_ED2_Sample/onehot/seed42/noise0/ModelArcEnsemble/checkpoint.pt" \
  --input "$project_root/examples/predict_sequences.csv" \
  --output "$project_root/results/demo/new_predictions.csv"
