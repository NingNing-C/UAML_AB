"""One tensor-only checkpoint service for training, prediction and search.

Version 2 checkpoints contain reconstruction settings, RNG policies, weights,
GP conditioning observations when needed, and workflow metadata. Legacy raw
Model Arc state dictionaries and earlier version 1 wrappers remain readable.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from uaml_ab.models.registry import METHODS, default_settings

FORMAT_VERSION = 2


def _plain(value):
    """Keep user metadata portable and compatible with weights_only=True."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Checkpoint metadata contains unsupported type {type(value).__name__}.")


def save(fitted, path):
    """Persist the fitted model and its metadata together; no sidecar is needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fitted.method == "Linreg":
        weights = {"kind": "linear", "coef": torch.as_tensor(fitted.model.coef_).cpu(),
                   "intercept": torch.as_tensor(fitted.model.intercept_).cpu()}
    else:
        weights = {"kind": "state_dict", "state_dict": {
            name: value.detach().cpu() for name, value in fitted.model.state_dict().items()}}
    payload = {"format": "uaml-ab", "format_version": FORMAT_VERSION,
               "method": fitted.method, "encoding": fitted.encoding,
               "settings": _plain(fitted.settings), "spec": _plain(fitted.spec),
               "seeds": {"run": int(fitted.seed), "initialization": int(fitted.initialization_seed)},
               "weights": weights, "metadata": _plain(fitted.metadata)}
    if fitted.method == "GP":
        payload["conditioning"] = {"train_x": fitted.gp_train_x.detach().cpu(),
                                    "train_y": fitted.gp_train_y.detach().cpu()}
    torch.save(payload, path)
    return path


def _legacy_wrapper(payload, config):
    method = payload["method"]
    weights = ({"kind": "linear", "coef": payload["linear_coef"], "intercept": payload["linear_intercept"]}
               if method == "Linreg" else {"kind": "state_dict", "state_dict": payload["state_dict"]})
    seed = int(payload.get("seed", 42))
    metadata = deepcopy(payload.get("metadata", {}))
    metadata.update(legacy=True, legacy_format=1)
    if config is not None:
        metadata.setdefault("config", config)
    result = {"method": method, "encoding": payload["encoding"],
              "settings": payload["settings"], "spec": payload["spec"],
              "seeds": {"run": seed, "initialization": 42 if method == "ModelArcEnsemble" else seed},
              "weights": weights, "metadata": metadata}
    if method == "GP":
        result["conditioning"] = {"train_x": payload["gp_train_x"], "train_y": payload["gp_train_y"]}
    return result


def _legacy_raw(payload, config):
    if config is None:
        raise ValueError("Legacy raw ensemble weights require a configuration for encoding and sequence metadata.")
    required = ("models.0.fc1.weight", "models.3.conv1.weight")
    if not isinstance(payload, dict) or any(name not in payload for name in required):
        raise ValueError("Checkpoint is neither canonical nor a supported raw Model Arc state dictionary.")
    global_config = config.get("global", {})
    encoding = global_config.get("encoder", "onehot")
    if encoding not in ("onehot", "esm2"):
        raise ValueError("Legacy raw weights require a onehot or esm2 configuration.")
    seed = int(config.get("seed", global_config.get("seed", 42)))
    return {"method": "ModelArcEnsemble", "encoding": encoding,
            "settings": config.get("methods", {}).get("ModelArcEnsemble", {}),
            "spec": {"input_size": int(payload[required[0]].shape[1]),
                     "input_dim": int(payload[required[1]].shape[1]), "train_count": 0},
            "seeds": {"run": seed, "initialization": 42},
            "weights": {"kind": "state_dict", "state_dict": payload},
            "metadata": {"legacy": True, "legacy_format": "raw_state_dict", "config": deepcopy(config)}}


def load(path, device="cpu", config=None, reference_data=None, reference_cnn=None, payload=None):
    """Load canonical v2, a legacy v1 wrapper, or raw Model Arc weights.

    Config is required only for raw weights. Weight shapes provide their input
    dimensions, so inference does not require reading the old training labels.
    reference_data/reference_cnn are accepted for compatibility with callers;
    the canonical specification is sufficient for reconstruction.
    """
    from uaml_ab.runtime import FittedModel, _build
    if payload is None:
        payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("A checkpoint must contain a model dictionary.")
    version = payload.get("format_version")
    if version == FORMAT_VERSION:
        if payload.get("format") != "uaml-ab":
            raise ValueError("Unrecognized checkpoint format.")
        normalized = payload
    elif version == 1:
        normalized = _legacy_wrapper(payload, config)
    elif version is None:
        normalized = _legacy_raw(payload, config)
    else:
        raise ValueError(f"Unsupported checkpoint version: {version!r}.")
    method = normalized.get("method")
    if method not in METHODS:
        raise ValueError(f"Unsupported checkpoint method: {method!r}.")
    settings = default_settings(method)
    settings.update(normalized["settings"])
    spec, encoding = normalized["spec"], normalized["encoding"]
    if encoding not in ("onehot", "esm2"):
        raise ValueError("Checkpoint encoding must be onehot or esm2.")
    seed = int(normalized["seeds"]["run"])
    initialization_seed = int(normalized["seeds"]["initialization"])
    conditioning = normalized.get("conditioning", {})
    gp_x, gp_y = conditioning.get("train_x"), conditioning.get("train_y")
    model, trainer = _build(method, settings, spec, torch.device(device), seed, train=False,
                            gp_train_x=gp_x, gp_train_y=gp_y, initialization_seed=initialization_seed)
    weights = normalized["weights"]
    if method == "Linreg":
        model.coef_ = weights["coef"].detach().cpu().numpy()
        model.intercept_ = weights["intercept"].detach().cpu().numpy()
        model.n_features_in_ = spec["input_size"]
    else:
        model.load_state_dict(weights["state_dict"])
        model.eval()
    return FittedModel(method, model, trainer, settings, spec, encoding, seed, device, gp_x, gp_y,
                       initialization_seed=initialization_seed, metadata=deepcopy(normalized.get("metadata", {})))
