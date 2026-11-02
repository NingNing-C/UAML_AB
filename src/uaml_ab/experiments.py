"""Shared experiment service: single runs and benchmark sweeps use the same fit.

This module owns partitions, provenance and output records. Model construction,
fitting, label-free prediction and checkpoint serialization live in the shared
runtime; individual command handlers do not implement model-specific branches.
"""
import copy
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .config import load_config, select_encoding
from .data import load_case, load_data, split_case, get_data_splits, add_gaussian_noise
from .records import (json_value, write_json, sequence_values, prediction_table,
                      prediction_metrics, source_fingerprint, save_split_table)


def resolve_device(device="auto"):
    return "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device


def prepare_case(config_path, encoding=None):
    config, flat, positional = load_case(config_path, encoding=encoding)
    if sequence_values(flat).duplicated().any():
        raise ValueError("Training cases require one row per sequence. Curate replicate observations before splitting.")
    if not flat.index.is_unique:
        raise ValueError("Input row identifiers must be unique.")
    encoding = config["global"]["encoder"]
    return config, flat, positional, source_fingerprint(config, flat, encoding)


def _dataset_directory(name):
    result = "".join(character if character.isalnum() or character in "._-" else "-" for character in str(name))
    if result in {"", ".", ".."}:
        raise ValueError("Dataset names must identify a nonempty output directory.")
    return result


def _partition(flat, positional, method, seed, n_train, split_method, mutation_split_setting, test_on_rest):
    if n_train == -1:
        n_train = None
    cap = min(n_train, 40000) if method == "GP" and n_train is not None else 40000 if method == "GP" else n_train
    if split_method == "random":
        return split_case(flat, positional, seed=seed, n_train=cap)
    options = SimpleNamespace(split_method=split_method, mutation_split_setting=mutation_split_setting,
                              test_on_rest=test_on_rest, n_train=cap, n_train_frac=None)
    parts = get_data_splits(flat, options, seed)
    return parts, tuple(positional.loc[part.index].copy() for part in parts)


def train(config_path, method="ModelArcEnsemble", *, encoding=None, seed=42,
          epochs=None, n_train=None, noise_level=0.0, output_dir="results",
          device="auto", hpo=False, hpo_trials=20, hpo_epochs=50,
          hpo_bnn_epochs=100, split_method="random", mutation_split_setting=None,
          test_on_rest=False, initialization_seed=None, prepared=None):
    """Train any registered method and write one self-contained checkpoint."""
    from .runtime import fit
    from .models.registry import default_settings
    case = prepare_case(config_path, encoding) if prepared is None else prepared
    config, flat, positional, fingerprint = case
    encoding = config["global"]["encoder"]
    settings = default_settings(method)
    settings.update(copy.deepcopy(config.get("methods", {}).get(method, {})))
    settings.pop("input_size", None)
    settings.pop("input_dim", None)
    epoch_limit = int(settings.get("epochs", 100) if epochs is None else epochs)
    if epoch_limit < 1:
        raise ValueError("epochs must be positive.")
    if not np.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and nonnegative.")
    splits, cnn_splits = _partition(flat, positional, method, seed, n_train,
                                    split_method, mutation_split_setting, test_on_rest)
    training, validation, test = [part.copy() for part in splits]
    training_cnn, validation_cnn, test_cnn = [part.copy() for part in cnn_splits]
    if min(len(training), len(validation), len(test)) < 1:
        raise ValueError("Training, validation and test partitions must all be nonempty.")
    training = add_gaussian_noise(training, noise_level=noise_level, seed=42)
    validation = add_gaussian_noise(validation, noise_level=noise_level, seed=42)
    training_cnn["label"] = training.label
    validation_cnn["label"] = validation.label
    dataset = str(config["global"].get("dataset", Path(config_path).stem))
    # Dataset names are labels, never arbitrary output paths.
    safe_name = _dataset_directory(dataset)
    run_dir = Path(output_dir) / safe_name / encoding / f"seed{seed}" / f"noise{noise_level:g}" / method
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    hpo_report = None
    if hpo:
        from .hpo import tune
        selected, hpo_report = tune(method, training, validation, settings, device, seed,
                                    encoding=encoding, train_cnn=training_cnn,
                                    validation_cnn=validation_cnn, n_trials=hpo_trials,
                                    epochs=hpo_epochs, bnn_epochs=hpo_bnn_epochs,
                                    initialization_seed=initialization_seed)
        settings.update(selected)
        write_json(run_dir / "hpo.json", hpo_report)
    model = fit(method, training, validation, settings, device, seed, epoch_limit,
                encoding=encoding, train_cnn=training_cnn, val_cnn=validation_cnn,
                initialization_seed=initialization_seed)
    result = model.predict(test, df_cnn=test_cnn, seed=seed)
    predictions = prediction_table(test, result)
    metrics = prediction_metrics(predictions)
    metadata = {"record_version": 1, "dataset": dataset, "method": method,
                "encoding": encoding, "config": config,
                "config_path": str(Path(config_path).resolve()), "source_fingerprint": fingerprint,
                "splits": {name: [int(value) for value in part.index]
                           for name, part in zip(("train", "validation", "test"), splits)},
                "split_method": split_method, "mutation_split_setting": mutation_split_setting,
                "test_on_rest": test_on_rest, "split_seed": seed, "inference_seed": seed,
                "initialization_seed": initialization_seed if initialization_seed is not None else 42 if method == "ModelArcEnsemble" else seed,
                "noise_level": noise_level, "noise_seed": 42, "epochs_requested": epoch_limit,
                "n_train": len(training), "n_validation": len(validation), "n_test": len(test),
                "settings": model.settings, "hpo": hpo_report,
                "software": {"torch": str(torch.__version__), "numpy": np.__version__}}
    model.metadata = json_value(metadata)
    model.save(run_dir / "checkpoint.pt")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    save_split_table(run_dir / "splits.csv", flat, *splits)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "run.json", model.metadata)
    logging.info("Saved %s %s -> %s", dataset, method, run_dir)
    return {"dataset": dataset, "method": method, "encoding": encoding, "seed": seed,
            "noise_level": noise_level, "status": "complete", "run_dir": str(run_dir),
            "n_train": len(training), "n_validation": len(validation), **metrics}


def benchmark(config_paths, methods=None, *, seeds=(42,), noise_levels=(0.0,), **options):
    """Coordinate ordinary train calls; never implement a second model pipeline."""
    from .models.registry import METHODS
    selected = list(METHODS if methods is None else methods)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(METHODS):
        raise ValueError(f"Select distinct reported methods from {list(METHODS)}.")
    if len(set(seeds)) != len(seeds) or len(set(noise_levels)) != len(noise_levels):
        raise ValueError("Benchmark seed and noise lists must not contain duplicates.")
    root = Path(options.get("output_dir", "results"))
    root.mkdir(parents=True, exist_ok=True)
    rows, failures, seen = [], [], set()
    for path in config_paths:
        case = prepare_case(path, options.get("encoding"))
        dataset = case[0]["global"].get("dataset", Path(path).stem)
        directory = _dataset_directory(dataset)
        if directory in seen:
            raise ValueError(f"Duplicate dataset name {dataset!r} in this sweep.")
        seen.add(directory)
        for seed in seeds:
            for noise in noise_levels:
                for method in selected:
                    try:
                        rows.append(train(path, method, seed=seed, noise_level=noise,
                                          prepared=case, **options))
                    except Exception as error:
                        logging.exception("Failed benchmark case %s %s", dataset, method)
                        failure = {"dataset": dataset, "method": method, "seed": seed,
                                   "noise_level": noise, "status": "failed",
                                   "error": f"{type(error).__name__}: {error}"}
                        rows.append(failure)
                        failures.append(failure)
    table = pd.DataFrame(rows)
    table.to_csv(root / "summary.csv", index=False)
    write_json(root / "benchmark.json", {"configs": list(config_paths), "methods": selected,
                                       "seeds": list(seeds), "noise_levels": list(noise_levels),
                                       "options": options, "runs": rows})
    if failures:
        raise RuntimeError(f"{len(failures)} benchmark run(s) failed; see {root/'summary.csv'}.")
    return table


def _model_context(checkpoint, config_path, device):
    from .runtime import load
    config = load_config(config_path) if config_path is not None else None
    model = load(checkpoint, device=resolve_device(device), config=config)
    config = config if config is not None else copy.deepcopy(model.metadata.get("config"))
    if config is None:
        raise ValueError("This checkpoint has no dataset configuration; supply --config.")
    return model, select_encoding(config, model.encoding)


def _prediction_input(config, encoding, input_path, require_labels, sequence_col=None, label_col=None):
    selected = select_encoding(config, encoding)
    # These are explicit per-call input overrides, not a change of profile.
    selected.pop("_base_settings", None)
    settings = selected["global"]
    if sequence_col:
        settings["seq_col"] = sequence_col
    if label_col:
        settings["label_col"] = label_col
    return load_data(selected, encoding=encoding, require_labels=require_labels,
                     path=None if input_path is None else str(Path(input_path).resolve()))


def predict(checkpoint, input_path, *, config_path=None, output="results/predictions.csv",
            device="auto", seed=None, sequence_col=None):
    model, config = _model_context(checkpoint, config_path, device)
    flat, positional = _prediction_input(config, model.encoding, input_path, False, sequence_col)
    inference_seed = int(model.metadata.get("inference_seed", model.seed) if seed is None else seed)
    result = model.predict(flat, df_cnn=positional, seed=inference_seed)
    table = prediction_table(flat, result, include_labels=False)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    write_json(destination.with_suffix(".json"), {"checkpoint": str(Path(checkpoint).resolve()),
                                               "encoding": model.encoding, "method": model.method,
                                               "seed": inference_seed, "n_predicted": len(table)})
    logging.info("Predicted %s rows -> %s", len(table), destination)
    return table


def evaluate(checkpoint, *, config_path=None, external_data=None, output_dir=None,
             output=None, device="auto", seed=None, sequence_col=None, label_col=None,
             n_train=None, split_method="random", mutation_split_setting=None):
    model, config = _model_context(checkpoint, config_path, device)
    flat, positional = _prediction_input(config, model.encoding, external_data, True, sequence_col, label_col)
    if external_data is not None:
        scope = "external_all_rows"
    elif model.metadata.get("splits"):
        if source_fingerprint(config, flat, model.encoding) != model.metadata.get("source_fingerprint"):
            raise ValueError("Source data or features changed since training. Use --external_data for a different dataset.")
        indices = model.metadata["splits"]["test"]
        if len(set(indices)) != len(indices) or not set(indices).issubset(flat.index):
            raise ValueError("Checkpoint held-out row identifiers are invalid for this dataset.")
        flat, positional = flat.loc[indices], positional.loc[indices]
        scope = "saved_held_out"
    else:
        # Only legacy imports need reconstruction; every new run stores its split.
        if not model.metadata.get("legacy"):
            raise ValueError("This checkpoint has no saved test partition. Supply --external-data for explicit labeled evaluation.")
        split_seed = int(model.metadata.get("split_seed", model.seed) if seed is None else seed)
        splits, cnn_splits = _partition(flat, positional, model.method, split_seed, n_train,
                                       split_method, mutation_split_setting, False)
        flat, positional = splits[2], cnn_splits[2]
        scope = "legacy_reconstructed_held_out"
    inference_seed = int(model.metadata.get("inference_seed", model.seed) if seed is None else seed)
    predictions = prediction_table(flat, model.predict(flat, df_cnn=positional, seed=inference_seed))
    metrics = prediction_metrics(predictions)
    destination = Path(output) if output is not None else Path(output_dir or Path(checkpoint).parent / "evaluation") / "predictions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    write_json(destination.with_suffix(".metrics.json"), metrics)
    write_json(destination.with_suffix(".evaluation.json"), {"checkpoint": str(Path(checkpoint).resolve()),
                "scope": scope, "method": model.method, "encoding": model.encoding,
                "seed": inference_seed, "n_evaluated": len(predictions)})
    logging.info("Evaluated %s %s rows -> %s", scope, len(predictions), destination)
    return predictions, metrics
