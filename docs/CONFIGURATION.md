# Case configuration and experiment inputs

Each case has one YAML file with sequence/label metadata, paths, WT sequence,
one-hot model settings, and optional ESM2 overrides. Relative paths resolve
against that YAML file's parent directory. The three supplied configs contain
the tuned settings from the provided archive; input dimensions are inferred.

```yaml
global:
  dataset: my_antibody_target
  data_path: ../data/cases/my_case.csv
  seq_col: sequence
  label_col: binding
  wt: HMSMQQVVSAGWERADLVGDAFDV
  encoder: onehot
methods:
  ModelArcEnsemble:
    batch_size: 32
    lr: 0.001
    weight_decay: 0.00001
```

The values in this schematic example are starting settings, not verified
parameters for a new dataset. Methods without overrides use central registry
defaults. Include one finite numeric label and one aligned sequence per row
for training/evaluation. Curate replicate observations before splitting; the
experiment service rejects duplicated sequences to avoid train/test leakage.

The standard split uses the original pandas sampling identities: 20% test,
then 10% of the remainder for validation. `--n-train` caps the development pool
before validation. Exact GP additionally caps it at 40,000; the test rows are
unchanged. Mutation-number partitions can be selected explicitly:

```bash
uaml-ab train --config my_case.yaml --method ModelArcEnsemble \
  --split-method mutation_number --mutation-split-setting train_0_1_2_test_3+
```

Training/validation/test sets must all be nonempty. HPO needs at least two
validation observations. Generated checkpoints record the exact split identities;
new evaluations use those identities instead of rerunning the split procedure.

## Encoding profiles

Frozen ESM2 caches are configured under `global.esm2_embedding_path`.
Encoding-specific model settings override the one-hot defaults:

```yaml
encoding_overrides:
  esm2:
    methods:
      ModelArcEnsemble:
        batch_size: 32
        lr: 0.0001
        weight_decay: 0.00001
```

`--encoding esm2` selects that profile. Profile selection is independent and
reversible; selecting a profile never mutates the canonical base settings.
The loader preserves cached position order, uses mean-pooled features for MLPs
and positional features for CNNs, and pairs rows by their original identities.

For generation, configure full-domain context columns, explicit residue order,
and optionally WT heavy/light context:

```yaml
global:
  esm2_embedding_path: ../data/cache/my_case.h5
  full_sequence_cols: [sequence_vh, sequence_vl]
  embedding_positions: [100, 106, 103, 109, 102, 105, 108]
```

Residue coordinates are one-based over concatenated heavy/light context. The
generator accounts for ESM2's leading CLS token. Explicit position order is
preserved. Without explicit coordinates, mutation annotations are parsed and
positions sorted deterministically; WT-region fallback handles unannotated
contiguous regions. Use the exact archived order when reproducing old CNN inputs.

## Prediction inputs

`predict` accepts unlabeled sequence CSVs and compatible HDF5 caches without
fitness labels. It uses the same encoders and data loader, with labels disabled.
The model's prediction routine cannot inspect the input label column and does
not invoke trainer evaluation. Metrics are calculated only by `evaluate` or
the held-out stage of training.

For a relocated or different CSV schema, use `--config`, `--sequence-col`, and
the evaluation-only `--label-col`. A different labeled dataset must be supplied
explicitly with `--external-data`; source fingerprint checks prevent accidentally
treating changed data as the original held-out partition.

## Outputs

Single training and benchmark sweeps use the same directory convention:

`<output>/<dataset>/<encoding>/seed<N>/noise<L>/<method>/`.

Use separate output roots to retain runs with different epoch limits or custom
settings at the same dataset/encoding/seed/noise identity. The checkpoint stores
weights and metadata together. The JSON and CSV records are readable copies;
deleting a sidecar does not remove the checkpoint's reconstruction information.
