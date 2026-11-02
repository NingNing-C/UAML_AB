"""Portable configuration and encoding-specific method settings."""
from copy import deepcopy
from pathlib import Path

import yaml

PATH_KEYS = ("data_path", "data_dir", "model_dir", "esm2_embedding_path", "embedding_path")


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def select_encoding(config, encoding=None):
    """Return an independent configuration with the selected profile merged.

    The familiar `methods` mapping supplies defaults. Entries under
    `encoding_overrides.<encoding>.methods` override those defaults. The input
    dictionary is never modified, including embedded checkpoint snapshots.
    """
    result = deepcopy(config)
    if not isinstance(result, dict) or not isinstance(result.get("global"), dict):
        raise ValueError("Configuration must contain a global mapping.")
    selected = encoding or result["global"].get("encoder", "onehot")
    if selected not in ("onehot", "esm2"):
        raise ValueError("Encoding must be 'onehot' or 'esm2'.")
    baseline = result.get("_base_settings")
    if baseline is None:
        baseline = {"global": deepcopy(result["global"]), "methods": deepcopy(result.get("methods", {}))}
    if not isinstance(baseline, dict) or not isinstance(baseline.get("global"), dict):
        raise ValueError("Canonical configuration settings must be mappings.")
    result["global"] = deepcopy(baseline["global"])
    result["methods"] = deepcopy(baseline.get("methods", {}))
    methods = result["methods"]
    if not isinstance(methods, dict) or any(not isinstance(value, dict) for value in methods.values()):
        raise ValueError("methods must map method names to settings mappings.")
    profiles = result.get("encoding_overrides", {})
    if not isinstance(profiles, dict):
        raise ValueError("encoding_overrides must be a mapping.")
    profile = profiles.get(selected, {})
    if not isinstance(profile, dict):
        raise ValueError("Each encoding override must be a mapping.")
    _merge(result, profile)
    if not isinstance(result.get("global"), dict) or not isinstance(result.get("methods"), dict):
        raise ValueError("Encoding overrides must retain global and methods mappings.")
    if any(not isinstance(value, dict) for value in result["methods"].values()):
        raise ValueError("Each method's settings must be a mapping.")
    result["global"]["encoder"] = selected
    result["global"]["cnn_encoder"] = selected + "_cnn"
    result["_base_settings"] = deepcopy(baseline)
    return result


def load_config(path, encoding=None, expected_encoder=None):
    """Read YAML; relative input/cache paths resolve beside that YAML file."""
    path = Path(path).expanduser().resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("global"), dict):
        raise ValueError("Configuration must contain a global mapping.")
    mappings = [config["global"]]
    profiles = config.get("encoding_overrides", {})
    if isinstance(profiles, dict):
        mappings.extend(profile["global"] for profile in profiles.values()
                        if isinstance(profile, dict) and isinstance(profile.get("global"), dict))
    for settings in mappings:
        for key in PATH_KEYS:
            value = settings.get(key)
            if isinstance(value, str) and value:
                location = Path(value).expanduser()
                settings[key] = str(location if location.is_absolute() else (path.parent / location).resolve())
    selected = select_encoding(config, encoding)
    if expected_encoder is not None and selected["global"]["encoder"] != expected_encoder:
        raise ValueError(f"This workflow requires the {expected_encoder!r} encoder.")
    return selected
