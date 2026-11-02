"""
Per-surrogate prediction cache and model-evaluation ledger.

A surrogate prediction (pred, epistd, alestd, totalstd) depends only on the
(surrogate, sequence) pair, NOT on the acquisition function, sigma-type or
uncertainty weight. So a single cache shared across every GA arm that uses the
same surrogate makes acquisition/sigma/weight variety essentially free on the
model-evaluation budget: the 5M-prediction ceiling is spent by distinct
sequences scored (pop x generations x restarts), not by the number of arms.

The cache also doubles as the *retrospective universe*: every sequence the
surrogate ever scored, persisted to CSV, ready to be joined against the full
experimental data for true labels.
"""
import os
import numpy as np
import pandas as pd

PRED_FIELDS = ("pred", "epistd", "alestd", "totalstd")


class PredictionCache:
    """
    seq -> (pred, epistd, alestd, totalstd), backed by an arbitrary predict_fn.

    predict_fn(list[str]) -> dict[str, np.ndarray] with keys PRED_FIELDS, each
    array aligned to the input list (this is satisfied by fitness.make_predict_fn).

    `budget` (optional) is the maximum number of distinct sequences the cache is
    allowed to evaluate; exceeding it raises RuntimeError so a run fails loudly
    instead of silently blowing past the model-eval ceiling.
    """

    def __init__(self, predict_fn, budget=None, predict_batch_size=4096):
        self._predict_fn = predict_fn
        self.budget = budget
        self.predict_batch_size = predict_batch_size
        self._store = {field: {} for field in PRED_FIELDS}  # field -> {seq: value}
        self.n_evals = 0  # distinct sequences actually sent to the surrogate

    def __len__(self):
        return len(self._store["pred"])

    def get(self, seqs):
        """
        Return a dict of np.ndarrays (keys PRED_FIELDS) aligned to `seqs`,
        evaluating and caching any sequences not seen before.
        """
        seqs = list(seqs)
        misses = [s for s in dict.fromkeys(seqs) if s not in self._store["pred"]]

        if misses:
            if self.budget is not None and self.n_evals + len(misses) > self.budget:
                raise RuntimeError(
                    f"Prediction budget exceeded: {self.n_evals} used + {len(misses)} "
                    f"new > budget {self.budget}. Reduce pop/generations/restarts."
                )
            for start in range(0, len(misses), self.predict_batch_size):
                chunk = misses[start:start + self.predict_batch_size]
                preds = self._predict_fn(chunk)
                for field in PRED_FIELDS:
                    vals = np.asarray(preds[field]).ravel()
                    for s, v in zip(chunk, vals):
                        self._store[field][s] = float(v)
            self.n_evals += len(misses)

        return {field: np.array([self._store[field][s] for s in seqs])
                for field in PRED_FIELDS}

    def to_dataframe(self):
        """All cached sequences and their predictions (the retrospective universe)."""
        seqs = list(self._store["pred"].keys())
        data = {"sequence": seqs}
        for field in PRED_FIELDS:
            data[field] = [self._store[field][s] for s in seqs]
        return pd.DataFrame(data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

    @classmethod
    def warm_from_csv(cls, predict_fn, path, **kwargs):
        """Reconstruct a cache pre-populated from a saved CSV (no re-eval of
        those sequences). Useful to resume / extend a run."""
        cache = cls(predict_fn, **kwargs)
        if os.path.exists(path):
            df = pd.read_csv(path)
            for field in PRED_FIELDS:
                cache._store[field] = dict(zip(df["sequence"], df[field]))
        return cache
