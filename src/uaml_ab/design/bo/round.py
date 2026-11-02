"""Prepare a BO development split and use the shared model runtime."""
import copy
from pathlib import Path
from uaml_ab.data import split_train_val
from uaml_ab.runtime import fit


def fit_round(labeled, config, settings, device, epochs, seed, val_frac=0.1, checkpoint_dir=None):
    train, validation = split_train_val(labeled, val_frac=val_frac, random_state=seed)
    if not len(train) or not len(validation):
        raise ValueError("BO initialization must provide enough observations for nonempty training and validation partitions.")
    fitted = fit("ModelArcEnsemble", train, validation, settings, device, seed, epochs,
                 encoding=config.get("encoder", "onehot"), train_cnn=train, val_cnn=validation,
                 initialization_seed=seed)
    fitted.metadata.update({"workflow": "retrospective_bo",
                            "config": {"global": copy.deepcopy(config),
                                       "methods": {"ModelArcEnsemble": copy.deepcopy(fitted.settings)}},
                            "no_saved_test_partition": True, "n_train": len(train),
                            "n_validation": len(validation), "initialization_seed": seed})
    if checkpoint_dir is not None:
        fitted.save(Path(checkpoint_dir) / "model.pt")
    return fitted
