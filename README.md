# UAML-AB

Code for *Uncertainty-aware machine learning guides multi-objective antibody
engineering*.

UAML-AB is an uncertainty-aware machine learning framework for multi-objective
antibody engineering. It combines an architecture-diverse probabilistic ensemble
with genetic search and Bayesian optimization to model antibody-binding
landscapes, quantify aleatoric and epistemic uncertainty, and guide candidate
selection across multiple antigens.

## Install

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[benchmarks]"
```

For the lightweight Model Arc Ensemble workflow, `pip install -e .` is enough.
Frozen ESM2 tooling is optional: `pip install -e ".[esm2]"`.

Run `uaml-ab --help` or `python -m uaml_ab --help` for commands.

## Quick start

Train the UAML-AB ensemble on the Adams CDR1H example:

```bash
uaml-ab train --config configs/adams_cdr1h.yaml \
  --method ModelArcEnsemble --epochs 2 --n-train 100
```

Run every reported model on both Adams examples:

```bash
bash examples/run_adams_benchmark.sh --epochs 2 --n-train 100 \
  --output-dir results/adams_quick
```

These are short functionality checks: 90 training and 10 validation rows per
case, with the usual 20% held out for testing. Omit the overrides for full
development sets and configured training epochs. The C05 example is available
through `bash examples/run_demo.sh`.

## Models and data

| Method | Predictive model |
| --- | --- |
| `Linreg` | Ordinary least squares |
| `Deterministic` | Deterministic MLP |
| `MVE` | Mean-variance MLP |
| `CNN` | Mean-variance CNN |
| `BootstrapEnsemble` | Five-member bootstrap ensemble |
| `ModelArcEnsemble` | Six-member UAML-AB architecture ensemble |
| `BNN` | Bayesian neural network |
| `NeuralLinear` | Neural features with Bayesian output heads |
| `ConcreteDropout` | Learned concrete dropout network |
| `GP` | Exact Gaussian process |
| `Evidential` | Normal-inverse-gamma evidential regression |

`uaml-ab models` lists the central registry. Training, tuning, prediction and
evaluation support all eleven methods; the paper's GA/BO design workflows use
Model Arc Ensemble. Deterministic models have no invented uncertainty estimates.
Some uncertainty methods provide total standard deviation without a component
decomposition. Outputs use `pred`, `epistd`, `alestd`, and `totalstd` where
those quantities are available; the uncertainty columns are standard deviations.

The included filtered datasets are Adams CDR1H (700 variants), Adams CDR3H
(376 variants), and the C05 M22 ED2 sample (10,000 variants). The Adams label is
`fitness_score_mean_neglog10KD`; C05 uses `binding_enrichment_adj`. Each dataset
is fitted and evaluated separately. Dataset hashes and source-case names are
recorded in `data/provenance.json`.

`data/cases/` contains the benchmark datasets. `examples/` contains small
demonstration inputs and launch scripts. The C05 filename explicitly identifies
it as a sample.


## Checkpoints, prediction and evaluation

Each training run writes:

```text
results/<dataset>/<encoding>/seed<seed>/noise<level>/<method>/
  checkpoint.pt    Weights, reconstruction settings, seeds, splits and provenance
  predictions.csv Held-out predictions and available uncertainties
  metrics.json    Numerical held-out metrics
  run.json        Human-readable copy of checkpoint experiment metadata
  splits.csv      Human-readable row partition table
  hpo.json        Trial records when tuning is requested
```

Evaluate any method's saved held-out rows:

```bash
uaml-ab evaluate --checkpoint path/to/checkpoint.pt
```

Predict a new unlabeled CSV using that same checkpoint:

```bash
uaml-ab predict --checkpoint path/to/checkpoint.pt \
  --input new_sequences.csv --output results/new_predictions.csv
```

The input column defaults to the checkpoint's sequence column. Override it with
`--sequence-col`. Prediction never supplies dummy labels or computes a loss.
Use `evaluate --external-data external.csv` for all rows of an explicitly
provided labeled dataset. If data moved, pass `--config` with the relocated case.
Saved held-out evaluation verifies source data/feature fingerprints.

New checkpoints are self-contained for model reconstruction and partition
metadata; they still require the input data. Legacy raw Model Arc weights and
earlier version-one wrappers can be imported with `--config`. See
[reproducibility notes](docs/REPRODUCIBILITY.md) for legacy split reconstruction.

## Tuning, noise and ESM2

Use the same command with `--hpo` to search before the final fit. HPO uses
validation Spearman correlation, up to 5,000 development observations, 20
trials, and up to 50 epochs per trial (100 for BNN). Install `.[hpo]` if using
the lightweight environment; the benchmark extra already includes Optuna.

For example, train with label noise without changing held-out labels:

```bash
uaml-ab train --config configs/c05_m22_sample.yaml \
  --noise-level 0.3 --epochs 2 --n-train 200
```

Sweep the same model across noise levels with `benchmark --methods
ModelArcEnsemble --noise-levels 0 0.1 0.2 0.3`. Each level uses the source's
fixed Gaussian pattern, scaled by the original training/validation label SD.

Optional ESM2 generation and use:

```bash
uaml-ab embed --config configs/adams_cdr1h.yaml
uaml-ab benchmark --configs configs/adams_cdr1h.yaml --encoding esm2
```

This downloads the 650M ESM2 model on first generation. Compatible cached HDF5
features can be used directly. The supplied config retains the archived Adams
position order. Generation and benchmark dependencies are optional and imported
only when requested. Actual 650M inference was not run in release verification.

## Genetic search and retrospective BO

Use `uaml-ab design --help`, `uaml-ab shortlist --help`, and `uaml-ab bo --help`.
The manuscript workflow scripts accept full-data configs/checkpoints as inputs:

```bash
bash scripts/run_paper_ga.sh M22_ED2.yaml SI06_ED2.yaml M22.pt SI06.pt
bash scripts/run_paper_bo.sh M22_ED5_BO.yaml SI06_ED5_BO.yaml
```

These scripts use the full-scale experimental settings described in the manuscript and may take considerable time to run. The scripts state the parameters explicitly.
GA canonical checkpoints use their saved held-out calibration identities and
verify the reference dataset. Final selection pools per-search shortlists and
writes 15 candidates and the first six per acquisition.

BO uses a held-out reference set, but its per-round checkpoints do not store
which sequences belong to that set. To evaluate a BO checkpoint, supply the
labeled reference dataset explicitly with `--external-data`.

## Repository structure

```text
src/uaml_ab/
  cli.py, experiments.py        Commands and shared experiment orchestration
  config.py, data.py, encodings.py
  models/registry.py           All eleven methods, defaults and search spaces
  models/, training/          Scientific architectures and trainer policies
  runtime.py, checkpoints.py   Shared fit, prediction, save and load services
  metrics.py, records.py       Numerical metrics and experiment records
  hpo.py, embeddings.py        Optional tuning and frozen feature generation
  design/                     Shared acquisition, genetic search and BO
configs/                      Case settings and encoding-specific tuned profiles
data/cases/                   Adams benchmark cases and the C05 sample
tests/                        Data, model lifecycle, experiment and design tests
docs/                         Configuration and scientific compatibility
examples/                     Small prediction input and runnable demonstrations
scripts/                      Explicit manuscript workflow commands
```

Relative dataset/cache paths resolve beside their YAML config, so commands work
from other directories after installation. No import-path modifications or
tracking-service account are required.

## Tests and compatibility

```bash
python -m unittest discover -s tests -v
```

Optional-method tests skip when their dependencies are absent. Tests cover all
reported methods, both feature representations, true unlabeled prediction,
checkpoint restoration, shared splits/noise, source fingerprints and small
GA/BO runs. CI runs core and optional-model installations.

Read [configuration](docs/CONFIGURATION.md) and
[reproducibility](docs/REPRODUCIBILITY.md) before comparing archived results.
In particular, source training/stopping conventions remain explicit, and an
undefined-memory bug in Concrete Dropout regularization is corrected.

## Citation and license

Please cite the accompanying manuscript; metadata are in `CITATION.cff`.
The original [MIT license](LICENSE) is retained.

## Contributors
- **NingNing Chen** ([@NingNing-C](https://github.com/NingNing-C)), ETH Zurich
- **Lorenz Kaiser** ([@lokaiser](https://github.com/lokaiser)), ETH Zurich
