#!/usr/bin/env bash
# Inputs: two full ED2 configs and their UAML-AB checkpoints. Data are not bundled.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "Usage: bash scripts/run_paper_ga.sh M22_ED2.yaml SI06_ED2.yaml M22.pt SI06.pt" >&2
  exit 2
fi
uaml-ab design --configs "$1" "$2" --checkpoints "$3" "$4" \
  --antigen_names M22 SI06 --combine min \
  --acquisitions greedy lcb ei --sigma_types total --weight_modes calibrated \
  --seedings top_binders --ga_seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
  --pop 1500 --seed_pool_size 1500 --generations 2000 --max_evals 1000000 \
  --min_mut 6 --trust_radius 11 --n_select 15 --crossover_rate 0.8 \
  --mutation_rate 0.06 --tournament_k 3 --elitism_frac 0.05 \
  --alphabet_mode all --gate_percentile 99 --acq_normalize gate_margin \
  --sigma_cap_mult 3.0 --out_dir results/paper_ga
uaml-ab shortlist --run_dir results/paper_ga --sigma_type total \
  --weight_mode calibrated --n_per_acquisition 15 --n_experimental 6
