import logging
import os

import numpy as np
import pandas as pd

from uaml_ab.data import split_data_random


class ProteinPool:
    """
    Tracks the evolving state of a retrospective BO simulation.

    - labeled_pool:   sequences whose labels have been "revealed" and are used
                       to (re-)train the surrogate.
    - candidate_pool: sequences with known true labels (this is a retrospective
                       simulation) that the surrogate has not been trained on
                       yet and that the acquisition function can propose.
    - held_out_eval:  fixed slice never touched by the loop, used only to track
                       trajectory metrics on an independent set.
    """

    def __init__(self, labeled_pool, candidate_pool, held_out_eval):
        self.labeled_pool = labeled_pool.reset_index(drop=True)
        self.candidate_pool = candidate_pool.reset_index(drop=True)
        self.held_out_eval = held_out_eval.reset_index(drop=True)

    @classmethod
    def from_dataframe(cls, data, n_init, seed, held_out_frac=0.2, init_max_mut=None):
        # val_frac=0.0 -> all non-held-out data stays in the "train" split,
        # which becomes the ground-truth pool the BO loop draws from.
        ground_truth_pool, _, held_out_eval = split_data_random(
            data, n_train=-1, test_size=held_out_frac, val_frac=0.0, random_state=seed
        )

        # Restrict *only the initial seed* to low-mutation sequences when
        # init_max_mut is set. The excluded high-mutation sequences stay in the
        # candidate_pool below, so the loop can still acquire them in later
        # rounds — this just constrains where the search starts.
        init_source = ground_truth_pool
        if init_max_mut is not None:
            init_source = ground_truth_pool[ground_truth_pool['n_mut'] <= init_max_mut]
            if len(init_source) < n_init:
                logging.warning(
                    "init_max_mut=%s leaves only %d sequences (n_mut <= %s) for the "
                    "initial seed, fewer than n_init=%d; using all %d of them.",
                    init_max_mut, len(init_source), init_max_mut, n_init, len(init_source),
                )

        n_init = min(n_init, len(init_source))
        labeled_pool = init_source.sample(n=n_init, random_state=seed)
        candidate_pool = ground_truth_pool.drop(labeled_pool.index)

        return cls(labeled_pool, candidate_pool, held_out_eval)

    def acquire(self, indices, round_idx):
        """
        Move the candidate_pool rows at `indices` (the original DataFrame
        index, not a sequence value) to labeled_pool, revealing their labels.
        Matching by index rather than by sequence string avoids pulling in
        extra rows if candidate_pool ever contains duplicate-sequence rows
        (e.g. replicate measurements) beyond the ones actually selected.
        """
        mask = self.candidate_pool.index.isin(indices)
        acquired = self.candidate_pool[mask].copy()
        acquired['round_acquired'] = round_idx

        self.candidate_pool = self.candidate_pool[~mask].reset_index(drop=True)

        if 'round_acquired' not in self.labeled_pool.columns:
            self.labeled_pool['round_acquired'] = -1  # part of the initial seed set

        self.labeled_pool = pd.concat([self.labeled_pool, acquired], ignore_index=True)
        return acquired

    def save_state(self, out_dir):
        labeled = self.labeled_pool.copy()
        labeled['pool'] = 'labeled'

        candidate = self.candidate_pool.copy()
        candidate['pool'] = 'candidate'
        candidate['round_acquired'] = np.nan

        held_out = self.held_out_eval.copy()
        held_out['pool'] = 'held_out'
        held_out['round_acquired'] = np.nan

        state = pd.concat([labeled, candidate, held_out], ignore_index=True)
        state.to_csv(os.path.join(out_dir, 'pool_state.csv'), index=False)
