"""The eleven reported methods, defaults and hyperparameter search spaces.

This registry imports no optional model library. The trial adapter accepts an
Optuna-compatible object without importing Optuna into the core installation.
"""
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    defaults: dict
    representation: str = "flat"
    uncertainty: bool = True
    optional_dependencies: tuple = ()


_COMMON = {"batch_size": 32, "epochs": 100}
_ADAM = {"lr": 0.001, "weight_decay": 1e-5}
_MLP = {"hidden_size": 128}

def _settings(*parts, **overrides):
    result = dict(_COMMON)
    for part in parts:
        result.update(part)
    result.update(overrides)
    return result

METHOD_REGISTRY = {
    "Linreg": MethodSpec("Linreg", _settings(), uncertainty=False,
                         optional_dependencies=("sklearn",)),
    "Deterministic": MethodSpec("Deterministic", _settings(_ADAM, _MLP, output_size=1), uncertainty=False),
    "MVE": MethodSpec("MVE", _settings(_ADAM, _MLP, output_size=2)),
    "CNN": MethodSpec("CNN", _settings(_ADAM, num_filters=32, filter_size=5, output_size=2), representation="positional"),
    "BootstrapEnsemble": MethodSpec("BootstrapEnsemble", _settings(_ADAM, _MLP, num_models=5, output_size=2)),
    "ModelArcEnsemble": MethodSpec("ModelArcEnsemble", _settings(_ADAM, output_size=2), representation="paired"),
    "BNN": MethodSpec("BNN", _settings(_ADAM, _MLP, num_samples=10, output_size=2), optional_dependencies=("torchbnn",)),
    "NeuralLinear": MethodSpec("NeuralLinear", _settings(_ADAM, _MLP, num_samples=20, output_size=1), optional_dependencies=("torchbnn",)),
    "ConcreteDropout": MethodSpec("ConcreteDropout", _settings(_ADAM, _MLP, num_samples=10, output_size=2,
                                  weight_regularizer=1e-6, dropout_regularizer=1e-5)),
    "GP": MethodSpec("GP", _settings(_ADAM), optional_dependencies=("gpytorch",)),
    "Evidential": MethodSpec("Evidential", _settings(_ADAM, _MLP, output_size=4)),
}
METHODS = tuple(METHOD_REGISTRY)
DETERMINISTIC_METHODS = tuple(name for name, spec in METHOD_REGISTRY.items() if not spec.uncertainty)
STOCHASTIC_METHODS = ("BNN", "NeuralLinear", "ConcreteDropout")


def method_spec(method):
    try:
        return METHOD_REGISTRY[method]
    except KeyError as error:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.") from error


def default_settings(method):
    return deepcopy(method_spec(method).defaults)


def suggest_settings(trial, method):
    """The reported search spaces; architecture ensemble widths remain fixed."""
    method_spec(method)
    if method == "Linreg":
        return {}
    low, high = (1e-4, 1e-3) if method == "BNN" else ((1e-5, 1e-1) if method == "GP" else (1e-5, 1e-3))
    result = {"lr": trial.suggest_float("lr", low, high, log=True),
              "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)}
    if method in {"Deterministic", "MVE", "BNN", "Evidential", "ConcreteDropout", "BootstrapEnsemble", "NeuralLinear"}:
        result["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 256])
    if method == "NeuralLinear":
        result["n_layers"] = trial.suggest_int("n_layers", 1, 3)
    if method == "CNN":
        result["num_filters"] = trial.suggest_categorical("num_filters", [8, 16, 32])
        result["filter_size"] = trial.suggest_categorical("filter_size", [3, 5])
    return result
