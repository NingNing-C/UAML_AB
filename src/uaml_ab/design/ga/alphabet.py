"""
Per-position amino-acid alphabet and seed-pool construction for the GA.

Which positions are mutable is always set by the training dataset (only positions
the library actually varied are touched). `build_position_alphabet(aa_mode=...)`
controls the residues allowed at those positions: "library" = only residues
observed there (novel = unseen *combination* of seen residues), "all" = all 20
amino acids (restricted by position but not by residue — explores unseen
residues, more OOD). The seed pool (initial population) is built from the same
training dataset.
"""
import numpy as np

_AA20 = list("ACDEFGHIKLMNPQRSTVWY")     # the 20 canonical amino acids


def hamming(a, b):
    """Number of positions at which equal-length strings a and b differ."""
    return sum(1 for x, y in zip(a, b) if x != y)


def hamming_to_wt(seq, wt):
    return hamming(seq, wt)


def build_position_alphabet(train_df, wt, seq_col="sequence", aa_mode="library"):
    """
    Per-position list of allowed residues. Which positions are *mutable* is always
    set by the library (a position is mutable iff the training data varied it, i.e.
    >1 residue observed there); `aa_mode` only controls *which residues* are
    allowed at those mutable positions:

      - "library" (default): only residues actually observed at that position in
        the training data (WT included). "Novel" = an unseen *combination* of seen
        residues — keeps proposals inside the regime the surrogate has seen.
      - "all": all 20 amino acids at every mutable position (restricted by
        *position* but no longer by *residue*). Constant (never-varied) positions
        stay fixed to their single residue in BOTH modes, so the mutable position
        set is identical — only the per-position choice set widens. This explores
        unseen residues (more OOD; lean on the σ-vs-ED / gate to judge reliability).

    Returns a list of length len(wt); element i is a sorted list of allowed
    residues for position i.
    """
    seqs = train_df[seq_col].astype(str).tolist()
    L = len(wt)
    seqs = [s for s in seqs if len(s) == L]
    if not seqs:
        raise ValueError(
            f"No training sequences of length {L} (WT length) found in column "
            f"'{seq_col}'. Cannot build a position alphabet."
        )
    if aa_mode not in ("library", "all"):
        raise ValueError(f"aa_mode must be 'library' or 'all', got '{aa_mode}'.")

    arr = np.array([list(s) for s in seqs])
    alphabet = []
    for i in range(L):
        observed = set(arr[:, i].tolist())
        observed.add(wt[i])               # WT residue is always allowed
        if aa_mode == "all" and len(observed) > 1:   # variable position → all 20 AAs
            alphabet.append(sorted(set(_AA20) | observed))
        else:                                        # library mode, or a constant position
            alphabet.append(sorted(observed))
    return alphabet


def _random_mutant(wt, position_alphabet, trust_radius, rng, min_mut=0):
    """One random sequence with between `min_mut` and `trust_radius` mutations of
    WT (min_mut=0 -> at least 1), using only per-position library-observed residues."""
    L = len(wt)
    # mutable positions = those with >1 observed residue
    mutable = [i for i in range(L) if len(position_alphabet[i]) > 1]
    if not mutable:
        return wt
    hi = min(trust_radius, len(mutable))
    lo = min(max(1, min_mut), hi)
    k = rng.integers(lo, hi + 1)
    positions = rng.choice(mutable, size=int(k), replace=False)
    chars = list(wt)
    for i in positions:
        choices = [a for a in position_alphabet[i] if a != chars[i]]
        if choices:
            chars[i] = choices[rng.integers(len(choices))]
    return "".join(chars)


def build_seed_pool(train_df, wt, position_alphabet, mode, n, rng,
                    trust_radius, seq_col="sequence", label_col="label", min_mut=0):
    """
    Build the initial seed sequences for a GA run.

    mode:
      - 'top_binders': the `n` highest-label sequences from the training dataset
        (starts the search in a known-functional basin).
      - 'random_wt':   `n` random mutants of WT within `trust_radius`, using the
        per-position library alphabet (WT-centered, more exploratory).

    Returns a list of unique sequences (length may be < n if the data is small).
    """
    if mode == "top_binders":
        if label_col not in train_df.columns:
            raise ValueError(f"mode='top_binders' needs a '{label_col}' column.")
        ordered = (train_df.dropna(subset=[label_col])
                           .sort_values(label_col, ascending=False)[seq_col]
                           .astype(str).tolist())
        seen, pool = set(), []
        for s in ordered:
            if len(s) == len(wt) and s not in seen:
                seen.add(s)
                pool.append(s)
            if len(pool) >= n:
                break
        return pool

    if mode == "random_wt":
        seen, pool = set(), []
        attempts = 0
        while len(pool) < n and attempts < n * 50:
            attempts += 1
            s = _random_mutant(wt, position_alphabet, trust_radius, rng, min_mut=min_mut)
            if s not in seen:
                seen.add(s)
                pool.append(s)
        return pool

    raise ValueError(f"Unknown seeding mode '{mode}' (expected 'top_binders' or 'random_wt').")
