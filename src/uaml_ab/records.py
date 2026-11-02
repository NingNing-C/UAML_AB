"""Shared experiment records, prediction validation and numerical metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import regression_metrics, uq_metrics, UQ_Metrics_fns


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def sequence_values(frame):
    values = frame["mut_loci_seq"] if "mut_loci_seq" in frame else frame["sequence"]
    if not values.map(lambda item: isinstance(item, str)).all():
        raise ValueError("Encoded inputs must retain original sequence identities in mut_loci_seq.")
    return values


def prediction_table(frame, predictions, include_labels=True):
    table = pd.DataFrame({"row_id": frame.index.to_numpy(),
                          "sequence": sequence_values(frame).to_numpy()})
    if include_labels:
        if "label" not in frame:
            raise ValueError("Evaluation requires labels; use predict for unlabeled inputs.")
        table["label"] = frame["label"].to_numpy()
    if "n_mut" in frame:
        table["n_mut"] = frame["n_mut"].to_numpy()
    if "pred" not in predictions:
        raise ValueError("The model returned no mean prediction.")
    for name in ("pred", "epistd", "alestd", "totalstd"):
        if predictions.get(name) is None:
            continue
        values = np.asarray(predictions[name])
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain one finite value per input row.")
        if name != "pred" and (values < 0).any():
            raise ValueError(f"{name} must contain nonnegative standard deviations.")
        table[name] = values
    return table


def prediction_metrics(table):
    result = regression_metrics(table.pred.to_numpy(), table.label.to_numpy())
    result.update({name: float("nan") for name in UQ_Metrics_fns})
    available = "totalstd" in table
    if available:
        result.update(uq_metrics(table.pred.to_numpy(), table.totalstd.to_numpy(), table.label.to_numpy()))
    result.update(n_evaluated=len(table), uq_available=available)
    return result


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(config, frame, encoding):
    """Fingerprint source bytes and interpreted sequence/label/encoding metadata.

    For HDF5 this includes the complete cache bytes, so changing embeddings or
    their stored position order cannot silently reuse a saved evaluation split.
    """
    settings = config["global"]
    path = settings["esm2_embedding_path"] if encoding == "esm2" else settings["data_path"]
    digest = hashlib.sha256()
    digest.update(file_digest(path).encode())
    digest.update(json.dumps({"encoding": encoding, "sequence_col": settings.get("seq_col"),
                              "label_col": settings.get("label_col"), "wt": settings.get("wt")},
                             sort_keys=True).encode())
    for sequence in sequence_values(frame):
        encoded = sequence.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    if "label" in frame:
        digest.update(np.asarray(frame.label, dtype="<f8").tobytes())
    return digest.hexdigest()


def save_split_table(path, full_data, train, validation, test):
    rows = pd.DataFrame({"row_id": full_data.index, "split": "unused",
                         "partition_order": np.nan}).set_index("row_id")
    for label, part in (("train", train), ("validation", validation), ("test", test)):
        rows.loc[part.index, "split"] = label
        rows.loc[part.index, "partition_order"] = np.arange(len(part))
    rows.reset_index().to_csv(path, index=False)
