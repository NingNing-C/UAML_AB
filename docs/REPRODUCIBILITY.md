# Scientific compatibility and reproduction limits

The repository organization, experiment services, command interface, checkpoint
service and tests were rebuilt around shared interfaces. Scientific model and
trainer bodies retain the validated source implementations. This is a code
release with runnable examples, not a claim that all manuscript scores have
been regenerated.

## Included scope

All eleven reported methods, ordinary fitting, validation-only HPO, numerical
noise experiments, held-out/external evaluation, optional frozen ESM2 inputs,
GA selection and retrospective BO are included. The filtered Adams CDR1H/CDR3H
CSVs and M22 sample are copied unchanged from the supplied archive. Their hashes
are recorded in `data/provenance.json`. One-hot/ESM2 tuned settings are consolidated
into the three case configs; duplicated HPO result files are not required at run
time. The templates do not supply verified settings for unavailable SI06 data.

Publication figures, post-run analyses, exploratory notebooks, unreported
model variants, structural analyses, pretrained weights and cached embeddings
are excluded. The complete paired C05 ED2/ED5 and extrapolation datasets are
absent, so full manuscript reproduction requires additional data and verified
experiment records.

## Predictive conventions

Model Arc Ensemble uses three MLPs of widths `[128]`, `[128,64]`, and
`[128,64,32]`, and three CNNs with kernels `[3]`, `[3,5]`, and `[3,5,7]`, with
32 filters per kernel. Each member predicts a mean and log standard deviation,
uses its own Adam optimizer and Gaussian NLL, with gradient norm clipped to 1.
The NLL minimum variance is `1e-5`. The original 24-channel one-hot mapping and
leading zero position are retained for raw checkpoint compatibility.

Architecture/bootstrap/Monte Carlo aggregation uses sample variance of member
means (`correction=1`), mean predicted member variance for aleatoric variance,
and their sum for total variance. The direct label-free inference service is
tested against the retained scientific trainer evaluation routines. Individual
model families return only the uncertainty quantities their source routine
provides. Deterministic models have no uncertainty fields.

## Training policy

The source's unusual stopping and validation-loss behavior remains explicit
in the scientific trainer modules. This includes cumulative division in the
architecture ensemble validation signal, counting negative NLL as an unfavorable
stopping signal, deterministic validation broadcasting, BNN cumulative loss
conventions, and retaining last-epoch weights rather than restoring a best epoch.
These are known compatibility limitations, not ordinary statistically corrected
loss definitions. Held-out metrics are computed separately from prediction
records; inference itself never computes those legacy losses.

Concrete Dropout is intentionally corrected: its source allocated five
regularization slots, initialized three, then summed all five. The rebuilt
model sums exactly three initialized layer terms. Exact weights from an older
run affected by this undefined-memory defect are not reproducible by design.

Ordinary Model Arc initialization defaults to 42 independently of the split
seed, following the source. BO explicitly initializes with its run seed.
Other methods use their specified run seed. Initialization, partition and
inference seeds are separate checkpoint metadata. Stochastic inference with
an explicit seed uses an isolated RNG context and does not change caller RNG
state. Exact regenerated weights can still depend on hardware/library versions
and the order of random operations.

## Metrics and HPO

The main numerical comparison is held-out Pearson correlation and Spearman
correlation between total predictive SD and absolute error. Auxiliary Gaussian
coverage, sharpness, NLL and calibration-area values use the cleaned numerical
implementation. Calibration area is a coverage-grid approximation and should
not be assumed identical to archived uncertainty-toolbox outputs.

One HPO engine maximizes validation Spearman without test labels. It limits
development observations to 5,000, requests 20 trials, and uses at most 50
epochs per trial (100 for BNN). Registry-defined search spaces retain the
reported parameter choices. New seeded in-memory studies do not resume old
SQLite studies. Tuning budgets can be reduced explicitly for software checks.

## Features and checkpoints

Existing ESM2 HDF5 position order is preserved. The Adams configs record the
original unsorted cache order, so regeneration retains those CNN inputs.
Newly inferred positions are sorted if no explicit order is supplied. Cached
loading and a mocked generator are tested; no actual 650M model download or
inference is part of the release checks.

Canonical version-two checkpoints hold weights, architecture settings, encoding,
RNG policy, GP conditioning data where needed, and experiment metadata. Training
metadata includes the resolved case, exact split rows and source fingerprints,
including the input/cache file bytes. A changed embedding value or position order
therefore cannot silently reuse the old held-out evaluation.

Raw legacy Model Arc state dictionaries and earlier wrapped version-one
checkpoints remain readable. Supply a case config to locate inputs. If no exact
split metadata is embedded, evaluation reconstructs the random/mutation split
using explicit known settings; it cannot recover unknown historical partitions.
Prefer explicit external data when those settings are unavailable.

## Design and optimization

GA/BO use the shared runtime rather than building their own models. GA uses
canonical held-out calibration rows with dataset-fingerprint verification.
Legacy checkpoints use a logged split-seed fallback. BO checkpoints have no
ordinary case test partition and are flagged accordingly; use explicit external
labeled data for evaluation. They cannot supply a known held-out GA calibration
set merely by being a canonical checkpoint.

`scripts/run_paper_ga.sh` states population 1,500, 20 seeds, 6–11 substitutions,
uniform crossover probability 0.8, mutation probability 0.06, 5% elitism, one
million sequence-scoring events per search and a 2,000-generation ceiling.
The source's sigma cap at three times the reference 99.9th percentile remains
enabled, although that cap is not explicit in the manuscript text.

Final selection pools and deduplicates per-search shortlists, ranks by minimum
raw antigen prediction then proposal frequency, applies the Hamming-distance-two
preference with score-based top-up, and takes the first six from each final
15-entry shortlist. It does not impose the later cross-strategy uniqueness
filter that appeared in the alternate archive.

The BO script uses the source shell's explicit `alestd` setting. The source
Python default is `totalstd`, and the manuscript does not clearly resolve that
component choice. Both remain available. Retrospective ED5 inputs must include
the low-order variants used for initialization; they differ from the curated
extrapolation test libraries.
