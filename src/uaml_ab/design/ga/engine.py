"""
Genetic-algorithm engine for one arm (one acquisition / sigma / weight / seeding
/ ga-seed combination).

Operators on fixed-length amino-acid strings:
  - tournament selection (rank-based -> invariant to the wildly different score
    scales of greedy vs lcb vs ei, keeping arms comparable),
  - uniform crossover (recombines per-position mutations of two parents -> the
    epistatic-combination move that reaches deep ED; the edge over the paper's
    single-chain MCMC),
  - per-position mutation to a library-observed residue (back-to-WT allowed, so
    ED can decrease),
  - elitism, and trust-radius repair (revert excess substitutions to WT).

The engine is surrogate-agnostic: it is handed a `score_fn(seqs) -> DataFrame`
(see fitness.make_score_fn) and maximizes the returned `score`. A full genealogy
is recorded so the dominant-parent lineage of each selected sequence can be
reconstructed for sequence-landscape overlays.
"""
import numpy as np
import pandas as pd

from .alphabet import hamming, hamming_to_wt


class GAEngine:
    def __init__(self, wt, position_alphabet, seed_pool, score_fn, rng,
                 pop_size=256, max_generations=300, trust_radius=11, min_mut=0,
                 crossover_rate=0.7, mutation_rate=0.06, tournament_k=3,
                 elitism_frac=0.02, n_select=15, min_select_hamming=2,
                 stall_patience=25, qualify_cols=None, forbidden_seqs=None,
                 max_evals=None):
        self.wt = wt
        self.L = len(wt)
        self.alphabet = position_alphabet
        self.seed_pool = list(seed_pool)
        self.score_fn = score_fn
        self.rng = rng

        self.pop_size = pop_size
        self.max_generations = max_generations     # hard ceiling (backstop)
        self.trust_radius = trust_radius           # MAX mutations from WT
        self.min_mut = min_mut                     # MIN mutations from WT (0 = off);
        #   e.g. 6 forces every proposal strictly beyond an ED2 model's 1-5 training
        #   range, i.e. fully OOD, to probe generalisation into deeper mutational space
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.n_elite = max(1, int(round(elitism_frac * pop_size)))
        self.n_select = n_select
        self.min_select_hamming = min_select_hamming
        self.stall_patience = stall_patience       # stop after this many gens with no shortlist improvement
        # Fixed per-arm evaluation budget (sequences scored). When set, it SUPERSEDES
        # stall-based early stopping: the arm keeps running (even without improvement)
        # until it has scored this many sequences, so every arm does equal search work.
        # `max_generations` is still a hard backstop, so set it high enough that the
        # budget binds first (budget/(pop-n_elite) + a couple of generations).
        self.max_evals = max_evals
        self._n_scored = 0                          # cumulative sequences scored this arm
        # qualification gate: list of (pred_col, threshold); a NOVEL sequence
        # qualifies iff it beats every threshold (neutral, uncertainty-free bar
        # = "predicted better than the best training prediction for all antigens").
        self.qualify_cols = list(qualify_cols) if qualify_cols else []
        self.forbidden = set(forbidden_seqs) if forbidden_seqs is not None else set()

        self.mutable = [i for i in range(self.L) if len(self.alphabet[i]) > 1]
        self._next_id = 0
        self._score_cols = None     # numeric columns emitted by score_fn (discovered)
        self.records = {}           # id -> record dict (the genealogy)
        self._qual = {}             # novel QUALIFYING seq -> best acquisition score
        self._best_novel_seen = float("-inf")      # best NOVEL acquisition score so far
        self._best_novel_pred_seen = float("-inf") # best NOVEL predicted fitness so far
        self.best_score_per_gen = []               # running best NOVEL score per generation
        self.best_pred_per_gen = []                # running best NOVEL pred per generation
        self.mean_score_per_gen = []
        self.n_qualifiers_per_gen = []             # distinct novel qualifiers found by gen g
        self.shortlist_ref_per_gen = []            # n_select-th best qualifier score (or climbing pred)
        self.stop_reason = None
        # Per-generation REALISED operator statistics (measurement only — these are
        # derived by comparing the intermediate strings the operators already
        # produce, with NO extra RNG draws, so they never perturb the search).
        # crossover_rate/mutation_rate/min_mut/trust_radius are the *configured*
        # rates; how they actually play out (coin flips that fire, mutations that
        # land, repairs that trigger) is what gets recorded here for later analysis.
        self.operator_stats_per_gen = []

    def _qualifies(self, rec):
        """A record clears the neutral gate (beats every per-antigen threshold)."""
        if not self.qualify_cols:
            return False
        return all(rec.get(col, float("-inf")) > thr for col, thr in self.qualify_cols)

    # ------------------------------------------------------------------ helpers
    def _new_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    def _register(self, seq, gen, parents, dominant_parent, score_row):
        if self._score_cols is None:   # discover the score_fn's numeric columns once
            self._score_cols = [c for c in score_row.index if c != "sequence"]
        rec = {
            "id": self._new_id(),
            "sequence": seq,
            "gen": gen,
            "parent_ids": list(parents),
            "dominant_parent": dominant_parent,
            "num_mut": hamming_to_wt(seq, self.wt),
        }
        for c in self._score_cols:
            rec[c] = float(score_row[c])
        self.records[rec["id"]] = rec
        # Track the best NOVEL score / qualifiers (forbidden training seeds can
        # never be proposed, so they must not anchor stopping / the shortlist).
        if seq not in self.forbidden:
            if rec["score"] > self._best_novel_seen:
                self._best_novel_seen = rec["score"]
            if rec["pred"] > self._best_novel_pred_seen:
                self._best_novel_pred_seen = rec["pred"]
            if self._qualifies(rec):
                prev = self._qual.get(seq)
                if prev is None or rec["score"] > prev:
                    self._qual[seq] = rec["score"]
        return rec

    def _score_population(self, seqs):
        """Score a list of sequences, returning per-sequence rows aligned to seqs."""
        self._n_scored += len(seqs)                 # count every scoring event (for max_evals)
        scored = self.score_fn(seqs).set_index("sequence")
        return [scored.loc[s] for s in seqs]

    # --------------------------------------------------------------- operators
    def _tournament(self, population):
        idx = self.rng.choice(len(population), size=min(self.tournament_k, len(population)),
                              replace=False)
        contenders = [population[i] for i in idx]
        return max(contenders, key=lambda r: r["score"])

    def _uniform_crossover(self, a, b):
        mask = self.rng.random(self.L) < 0.5
        return "".join(a[i] if mask[i] else b[i] for i in range(self.L))

    def _mutate(self, seq):
        chars = list(seq)
        for i in self.mutable:
            if self.rng.random() < self.mutation_rate:
                choices = [c for c in self.alphabet[i] if c != chars[i]]
                if choices:
                    chars[i] = choices[self.rng.integers(len(choices))]
        return "".join(chars)

    def _repair(self, seq):
        """Keep the sequence within [min_mut, trust_radius] mutations of WT: revert
        random excess substitutions if it is too far, and add random mutations at
        WT positions if it is too close (min_mut, to push proposals into the deeper
        / OOD mutational space beyond the training range)."""
        chars = list(seq)
        diff = [i for i in range(self.L) if chars[i] != self.wt[i]]
        if len(diff) > self.trust_radius:                       # too far -> trim back
            excess = self.rng.choice(diff, size=len(diff) - self.trust_radius, replace=False)
            for i in excess:
                chars[i] = self.wt[i]
            diff = [i for i in range(self.L) if chars[i] != self.wt[i]]
        if self.min_mut and len(diff) < self.min_mut:           # too close -> mutate up
            wt_pos = [i for i in self.mutable if chars[i] == self.wt[i]]
            need = min(self.min_mut - len(diff), len(wt_pos))
            add = self.rng.choice(wt_pos, size=need, replace=False) if need > 0 else []
            for i in add:
                choices = [c for c in self.alphabet[i] if c != self.wt[i]]
                if choices:
                    chars[i] = choices[self.rng.integers(len(choices))]
        return "".join(chars)

    # ----------------------------------------------------- operator measurement
    def _child_op_row(self, parent_a, parent_b, pre, mutated, repaired, used_crossover):
        """REALISED operator effects for one produced child, by string comparison
        only (no RNG). `pre` = the string after crossover/clone but BEFORE mutation;
        `mutated` = after `_mutate`; `repaired` = after `_repair` (the final child).
          n_mut   = positions the mutation operator actually changed (mutated!=pre);
                    `_mutate` only ever moves to a DIFFERENT residue, so this is
                    exactly the number of mutation coin-flips that fired.
          n_trim  = repair edits reverting an excess substitution back to WT
                    (trust-radius pull-in); n_add = repair edits introducing a
                    substitution at a WT position (min_mut push-out). Every repaired
                    position is one or the other, so n_repair = n_trim + n_add.
          n_from_b/n_parent_diff (crossover children only) = of the positions where
                    the two parents differ, how many the child inherited from B —
                    i.e. how much recombination actually mixed the parents."""
        L = self.L
        n_mut = sum(1 for i in range(L) if mutated[i] != pre[i])
        n_trim = sum(1 for i in range(L)
                     if repaired[i] != mutated[i] and repaired[i] == self.wt[i])
        n_add = sum(1 for i in range(L)
                    if repaired[i] != mutated[i] and repaired[i] != self.wt[i])
        row = {"used_crossover": int(used_crossover), "n_mut": n_mut,
               "n_repair": n_trim + n_add, "n_trim": n_trim, "n_add": n_add}
        if used_crossover and parent_b is not None:
            row["n_parent_diff"] = sum(1 for i in range(L) if parent_a[i] != parent_b[i])
            row["n_from_b"] = sum(1 for i in range(L)
                                  if pre[i] == parent_b[i] and parent_a[i] != parent_b[i])
        return row

    def _record_op_stats(self, gen, op_rows, phase):
        """Aggregate the per-child operator rows for one generation into a single
        realised-rate record and append it to operator_stats_per_gen."""
        nm = len(self.mutable)
        rec = {"gen": gen, "phase": phase, "n_children": len(op_rows), "n_mutable": nm,
               "cfg_crossover_rate": self.crossover_rate, "cfg_mutation_rate": self.mutation_rate,
               "min_mut": self.min_mut, "trust_radius": self.trust_radius}
        if op_rows:
            used = np.array([r["used_crossover"] for r in op_rows], dtype=float)
            n_mut = np.array([r["n_mut"] for r in op_rows], dtype=float)
            n_rep = np.array([r["n_repair"] for r in op_rows], dtype=float)
            n_trim = np.array([r["n_trim"] for r in op_rows], dtype=float)
            n_add = np.array([r["n_add"] for r in op_rows], dtype=float)
            rec.update({
                "realised_crossover_rate": float(used.mean()),
                "n_crossover": int(used.sum()), "n_clone": int((used == 0).sum()),
                "realised_mut_rate": float(n_mut.mean() / nm) if nm else 0.0,
                "mean_mut_per_child": float(n_mut.mean()),
                "repair_frac": float((n_rep > 0).mean()),
                "mean_repair_edits": float(n_rep.mean()),
                "mean_repair_trim": float(n_trim.mean()),
                "mean_repair_add": float(n_add.mean()),
            })
            cross = [r for r in op_rows if r["used_crossover"] and "n_from_b" in r]
            if cross:
                nfb = np.array([r["n_from_b"] for r in cross], dtype=float)
                ndiff = np.array([r["n_parent_diff"] for r in cross], dtype=float)
                rec["mean_recomb_from_b"] = float(nfb.mean())
                rec["mean_parent_diff"] = float(ndiff.mean())
                eff = nfb[ndiff > 0] / ndiff[ndiff > 0]
                rec["recomb_effectiveness"] = float(eff.mean()) if eff.size else None
        self.operator_stats_per_gen.append(rec)

    # --------------------------------------------------------------------- run
    def _init_population(self):
        seqs = list(dict.fromkeys(self.seed_pool))           # de-dup, keep order
        # Fill up to pop_size by mutating randomly chosen seeds (the "padding").
        op_rows = []
        while len(seqs) < self.pop_size and seqs:
            parent = seqs[self.rng.integers(len(seqs))]
            mutated = self._mutate(parent)
            child = self._repair(mutated)
            seqs.append(child)
            op_rows.append(self._child_op_row(parent, None, parent, mutated, child,
                                              used_crossover=False))
        seqs = list(dict.fromkeys(seqs))[:self.pop_size]
        self._record_op_stats(0, op_rows, phase="padding")   # gen-0 fills (no crossover)

        rows = self._score_population(seqs)
        return [self._register(s, 0, [], None, r) for s, r in zip(seqs, rows)]

    def _make_offspring(self, population, n):
        children, parent_meta, seen = [], [], set()
        op_rows = []
        attempts = 0
        max_attempts = n * 25
        while len(children) < n and attempts < max_attempts:
            attempts += 1
            pa = self._tournament(population)
            if self.rng.random() < self.crossover_rate:
                pb = self._tournament(population)
                pre = self._uniform_crossover(pa["sequence"], pb["sequence"])
                parents = [pa["id"], pb["id"]]
                dominant = pa["id"] if pa["score"] >= pb["score"] else pb["id"]
                used_crossover, pb_seq = True, pb["sequence"]
            else:
                pre = pa["sequence"]
                parents = [pa["id"]]
                dominant = pa["id"]
                used_crossover, pb_seq = False, None

            mutated = self._mutate(pre)
            child = self._repair(mutated)
            if child in seen or child in self.forbidden:
                continue
            seen.add(child)
            children.append(child)
            parent_meta.append((parents, dominant))
            op_rows.append(self._child_op_row(pa["sequence"], pb_seq, pre, mutated,
                                              child, used_crossover))

        self._record_op_stats(self._cur_gen, op_rows, phase="offspring")
        if not children:
            return []
        rows = self._score_population(children)
        return [self._register(c, gen=self._cur_gen, parents=pm[0],
                               dominant_parent=pm[1], score_row=r)
                for c, pm, r in zip(children, parent_meta, rows)]

    def run(self):
        population = self._init_population()
        self._log_gen(population)                 # gen 0

        # Run until the qualifier shortlist stops improving (no fixed generation
        # target). "Improvement" = more distinct qualifiers found (up to n_select),
        # or the n_select-th-best qualifier score rising; before the first
        # qualifier exists, the climbing metric is the best novel pred (so we don't
        # quit while still approaching the threshold). The generation ceiling is a
        # backstop in case improvement never truly flatlines.
        best_nq, best_ref = self._progress()
        stall = 0
        self.stop_reason = "max_generations"
        for gen in range(1, self.max_generations + 1):
            self._cur_gen = gen
            elites = sorted(population, key=lambda r: r["score"], reverse=True)[:self.n_elite]
            offspring = self._make_offspring(population, self.pop_size - len(elites))
            population = (elites + offspring) if offspring else elites
            self._log_gen(population)
            nq, ref = self._progress()
            improved = (nq > best_nq) or (nq == best_nq and ref > best_ref + 1e-9)
            if improved:
                best_nq, best_ref, stall = nq, ref, 0
            else:
                stall += 1
            if self.max_evals is not None:
                # fixed-budget mode: run to the evaluation budget regardless of stalling
                if self._n_scored >= self.max_evals:
                    self.stop_reason = "max_evals"
                    break
            elif stall >= self.stall_patience:
                self.stop_reason = "stalled"
                break

        return self._finalize()

    def _progress(self):
        """Return (capped qualifier count, reference score) for stall detection.
        ref = n_select-th best qualifier score once any qualifier exists, else the
        best novel pred (climbing toward the threshold)."""
        nq = len(self._qual)
        if nq:
            scores = sorted(self._qual.values(), reverse=True)
            ref = scores[min(self.n_select, nq) - 1]
        else:
            ref = self._best_novel_pred_seen
        return min(nq, self.n_select), ref

    def _log_gen(self, population):
        b = self._best_novel_seen
        p = self._best_novel_pred_seen
        self.best_score_per_gen.append(float(b) if np.isfinite(b) else None)
        self.best_pred_per_gen.append(float(p) if np.isfinite(p) else None)
        self.mean_score_per_gen.append(float(np.mean([r["score"] for r in population])))
        _, ref = self._progress()
        self.n_qualifiers_per_gen.append(len(self._qual))    # true count (uncapped)
        self.shortlist_ref_per_gen.append(float(ref) if np.isfinite(ref) else None)

    # ---------------------------------------------------------------- finalize
    def _select(self):
        """The shortlist = top-by-score among novel sequences that PASS THE GATE
        (beat the per-antigen thresholds), with a minimum pairwise-Hamming
        diversity constraint. May return fewer than n_select if not enough
        qualifiers were found (a legitimate 'this target is hard' result). If no
        gate is configured, falls back to all novel sequences."""
        gated = bool(self.qualify_cols)
        by_seq = {}
        for r in self.records.values():
            if r["sequence"] in self.forbidden:
                continue                                   # novel only
            if r["num_mut"] < self.min_mut:
                continue                                   # enforce min edit distance (OOD floor)
            if gated and not self._qualifies(r):
                continue                                   # must clear the gate
            cur = by_seq.get(r["sequence"])
            if cur is None or r["score"] > cur["score"]:
                by_seq[r["sequence"]] = r
        ranked = sorted(by_seq.values(), key=lambda r: r["score"], reverse=True)

        selected = []
        for r in ranked:
            if len(selected) >= self.n_select:
                break
            if all(hamming(r["sequence"], s["sequence"]) >= self.min_select_hamming
                   for s in selected):
                selected.append(r)
        # top up by score (still gated) if diversity left us short
        if len(selected) < self.n_select:
            chosen_ids = {s["id"] for s in selected}
            for r in ranked:
                if len(selected) >= self.n_select:
                    break
                if r["id"] not in chosen_ids:
                    selected.append(r)
        return selected

    def _trajectory(self, leaf_id):
        """Dominant-parent path from seed -> selected sequence."""
        path = []
        node = self.records[leaf_id]
        while node is not None:
            path.append(node)
            dp = node["dominant_parent"]
            node = self.records[dp] if dp is not None else None
        return list(reversed(path))                        # seed first

    def _finalize(self):
        selected = self._select()
        score_cols = self._score_cols or ["score", "pred", "epistd", "alestd", "totalstd"]

        def _row(r, extra):
            base = {"id": r["id"], "sequence": r["sequence"], "num_mut": r["num_mut"]}
            base.update({c: r[c] for c in score_cols})
            base.update(extra)
            return base

        shortlist = pd.DataFrame([
            _row(r, {"first_gen": r["gen"], "qualifies": self._qualifies(r)})
            for r in selected])
        if shortlist.empty:                               # keep header when no qualifiers
            shortlist = pd.DataFrame(columns=["id", "sequence", "num_mut", *score_cols,
                                              "first_gen", "qualifies"])

        genealogy = pd.DataFrame([
            _row(r, {"gen": r["gen"],
                     "parent_ids": ";".join(map(str, r["parent_ids"])),
                     "dominant_parent": r["dominant_parent"]})
            for r in self.records.values()])

        traj_rows = []
        for r in selected:
            for step, node in enumerate(self._trajectory(r["id"])):
                traj_rows.append({
                    "trajectory_id": r["id"], "step": step, "gen": node["gen"],
                    "sequence": node["sequence"], "num_mut": node["num_mut"],
                    "pred": node["pred"], "score": node["score"],
                })
        trajectories = pd.DataFrame(traj_rows)
        if trajectories.empty:
            trajectories = pd.DataFrame(columns=["trajectory_id", "step", "gen",
                                                 "sequence", "num_mut", "pred", "score"])

        operator_stats = pd.DataFrame(self.operator_stats_per_gen)

        # run-level realised-rate summaries (offspring generations only, so gen-0
        # padding does not dilute the crossover figures)
        off = [r for r in self.operator_stats_per_gen
               if r.get("phase") == "offspring" and r.get("n_children", 0) > 0]

        def _mean_over_gens(key):
            vals = [r[key] for r in off if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        metrics = {
            "n_records": len(self.records),
            "n_scored": self._n_scored,             # scoring events this arm (vs max_evals budget)
            "n_unique_seqs": int(genealogy["sequence"].nunique()),
            "n_qualifiers": len(self._qual),
            "n_shortlist": len(shortlist),
            "stop_reason": self.stop_reason,
            "best_novel_score": float(self._best_novel_seen) if np.isfinite(self._best_novel_seen) else None,
            "best_novel_pred": float(self._best_novel_pred_seen) if np.isfinite(self._best_novel_pred_seen) else None,
            "n_generations_run": len(self.best_score_per_gen) - 1,
            "best_score_per_gen": self.best_score_per_gen,
            "best_pred_per_gen": self.best_pred_per_gen,
            "mean_score_per_gen": self.mean_score_per_gen,
            "n_qualifiers_per_gen": self.n_qualifiers_per_gen,
            "shortlist_ref_per_gen": self.shortlist_ref_per_gen,
            "shortlist_num_mut": shortlist["num_mut"].tolist() if len(shortlist) else [],
            "shortlist_mean_num_mut": float(shortlist["num_mut"].mean()) if len(shortlist) else None,
            "explored_num_mut_mean": float(genealogy["num_mut"].mean()) if len(genealogy) else None,
            "explored_num_mut_hist": np.bincount(
                genealogy["num_mut"].to_numpy(), minlength=self.L + 1).tolist() if len(genealogy) else [],
            # realised operator rates, averaged over offspring generations
            "realised_crossover_rate_mean": _mean_over_gens("realised_crossover_rate"),
            "realised_mut_rate_mean": _mean_over_gens("realised_mut_rate"),
            "repair_frac_mean": _mean_over_gens("repair_frac"),
            "mean_repair_edits_mean": _mean_over_gens("mean_repair_edits"),
            "recomb_effectiveness_mean": _mean_over_gens("recomb_effectiveness"),
        }

        return {
            "shortlist": shortlist,
            "genealogy": genealogy,
            "trajectories": trajectories,
            "operator_stats": operator_stats,
            "metrics": metrics,
        }
