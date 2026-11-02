#!/usr/bin/env bash
# Inputs: the two retrospective ED5 datasets, including low-order initialization variants.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/run_paper_bo.sh M22_ED5_BO.yaml SI06_ED5_BO.yaml" >&2
  exit 2
fi
# alestd follows the archived shell; the manuscript leaves the component ambiguous.
for acquisition in greedy lcb ucb ei random; do
  for seed in 42 43 44 45 46 47 48 49 50 51; do
    uaml-ab bo --configs "$1" "$2" --antigen_names M22 SI06 --combine min \
      --acquisition "$acquisition" --adaptive_weight --n_init 1000 \
      --init_max_mut 2 --batch_size 50 --n_rounds 5 --held_out_frac 0.2 \
      --acq_normalize gate_margin --acq_sigma alestd --seed "$seed" \
      --out_dir "results/paper_bo/${acquisition}_seed${seed}"
  done
done
