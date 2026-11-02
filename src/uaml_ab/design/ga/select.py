"""Pool GA search shortlists into final candidates, without retrospective analysis."""
import argparse
from pathlib import Path
import re
import pandas as pd

def _hamming(s, wt):
    return sum(c != w for c, w in zip(s, wt))

def _setting_of(arm):
    """Arm id with the trailing _seed<N> stripped → the 'setting'
    (acquisition × sigma-type × weight × seeding), i.e. everything but the seed."""
    return re.sub(r"_seed\d+$", "", arm)

def _parse_setting(s):
    """Split a setting string into its (acquisition, sigma_type, weight_mode, seeding)."""
    seeding, body = "unknown", s
    for sd in ("top_binders", "random_wt"):
        if s.endswith("_" + sd):
            body, seeding = s[: -len(sd) - 1], sd
            break
    if body == "greedy":
        return dict(acquisition="greedy", sigma_type="none", weight_mode="none", seeding=seeding)
    for acq in ("lcb", "ei"):
        if body.startswith(acq + "_"):
            st, _, wm = body[len(acq) + 1:].partition("_")
            return dict(acquisition=acq, sigma_type=st, weight_mode=wm or "none", seeding=seeding)
    return dict(acquisition=body, sigma_type="none", weight_mode="none", seeding=seeding)

def consolidate_by_acquisition(shortlists, n_per_acq, sigma_type="epistemic",
                               weight_mode="calibrated", min_hamming=2,
                               label_map=None, hit_threshold=None):
    """THE wet-lab order: one shortlist of `n_per_acq` sequences PER ACQUISITION
    (greedy / EI / LCB), so you can ask directly whether EI- or LCB-proposed
    candidates out-bind the greedy ones on average.

    To keep the acquisitions comparable, the σ-aware ones are taken at a single
    fixed σ-setting (`sigma_type` × `weight_mode`, default epistemic × calibrated);
    greedy uses no σ. For each acquisition the per-seed / per-seeding shortlists
    are pooled, deduped, ranked by predicted fitness, diversity-filtered on Hamming
    and truncated to `n_per_acq`. `n_arms` = how many arms (seed × seeding)
    proposed the sequence = consensus. Returns a tidy DataFrame."""
    # Robustness: if the requested σ-type isn't present among the arms (e.g. a
    # run that used only --sigma_types total), fall back to whatever σ-aware
    # arms exist, so the per-acquisition comparison isn't silently empty.
    avail = {_parse_setting(_setting_of(a))["sigma_type"]
             for a, sl in shortlists.items() if sl is not None and len(sl)}
    avail.discard("none")
    if avail and sigma_type not in avail:
        fallback = sorted(avail)[0]
        print(f"  [consolidate] sigma_type '{sigma_type}' not in runs "
              f"(available: {sorted(avail)}); using '{fallback}'")
        sigma_type = fallback

    pooled = {}   # acquisition -> {seq -> row}
    for arm, sl in shortlists.items():
        if sl is None or not len(sl):
            continue
        meta = _parse_setting(_setting_of(arm))
        acq = meta["acquisition"]
        if acq != "greedy" and not (meta["sigma_type"] == sigma_type
                                    and meta["weight_mode"] == weight_mode):
            continue
        d = pooled.setdefault(acq, {})
        for _, r in sl.iterrows():
            seq = r["sequence"]
            cur = d.get(seq)
            if cur is None:
                base = {"sequence": seq, "num_mut": int(r["num_mut"]),
                        "pred": float(r["pred"]), "epistd": float(r["epistd"]),
                        "alestd": float(r["alestd"]), "totalstd": float(r["totalstd"]),
                        "best_acq_score": float(r["score"]), "n_arms": 1,
                        "example_arm": arm}
                for c in r.index:                          # per-antigen detail (multi-objective)
                    if c.startswith(("pred_", "epistd_", "alestd_", "totalstd_")):
                        base[c] = float(r[c])
                d[seq] = base
            else:
                cur["n_arms"] += 1
                cur["best_acq_score"] = max(cur["best_acq_score"], float(r["score"]))

    out = []
    for acq, d in pooled.items():
        ranked = sorted(d.values(), key=lambda x: (x["pred"], x["n_arms"]), reverse=True)
        selected = []
        for r in ranked:
            if len(selected) >= n_per_acq:
                break
            if all(_hamming(r["sequence"], s["sequence"]) >= min_hamming for s in selected):
                selected.append(r)
        for r in ranked:                                    # top up if diversity left us short
            if len(selected) >= n_per_acq:
                break
            if r not in selected:
                selected.append(r)
        for rank, r in enumerate(selected, 1):
            row = dict(r)
            row["acquisition"] = acq
            row["sigma_type"] = "none" if acq == "greedy" else sigma_type
            row["weight_mode"] = "none" if acq == "greedy" else weight_mode
            row["rank"] = rank
            if label_map is not None:
                lab = label_map.get(r["sequence"])
                row["true_label"] = float(lab) if lab is not None else None
                row["true_above_wt"] = (bool(lab > hit_threshold)
                                        if (lab is not None and hit_threshold is not None) else None)
            out.append(row)
    cols = ["acquisition", "sigma_type", "weight_mode", "rank", "sequence", "num_mut",
            "pred", "epistd", "alestd", "totalstd", "best_acq_score", "n_arms", "example_arm"]
    cols += sorted({c for row in out for c in row                       # per-antigen detail
                    if c.startswith(("pred_", "epistd_", "alestd_", "totalstd_"))})
    if label_map is not None:
        cols += ["true_label", "true_above_wt"]
    cols = [c for c in cols if any(c in row for row in out)] or cols
    return pd.DataFrame(out)[cols] if out else pd.DataFrame(columns=cols)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="uaml-ab shortlist", description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="Directory produced by run_ga.py")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n_per_acquisition", type=int, default=15)
    parser.add_argument("--n_experimental", type=int, default=6,
                        help="Also write the first N candidates per acquisition")
    parser.add_argument("--sigma_type", choices=["total", "epistemic"], default="total")
    parser.add_argument("--weight_mode", choices=["calibrated", "unweighted"], default="calibrated")
    parser.add_argument("--min_hamming", type=int, default=2)
    args = parser.parse_args(argv)
    if args.n_per_acquisition < 1 or args.n_experimental < 1 or args.min_hamming < 0:
        parser.error("Candidate counts must be positive and --min_hamming nonnegative.")
    shortlists = {}
    for path in sorted(args.run_dir.glob("*/shortlist.csv")):
        try:
            shortlists[path.parent.name] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
    selected = consolidate_by_acquisition(
        shortlists, args.n_per_acquisition, sigma_type=args.sigma_type,
        weight_mode=args.weight_mode, min_hamming=args.min_hamming,
    )
    output = args.out or args.run_dir / "candidates_by_acquisition.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    experimental = selected[selected["rank"] <= args.n_experimental]
    experimental_path = output.with_name(output.stem + f"_top{args.n_experimental}.csv")
    experimental.to_csv(experimental_path, index=False)
    print(f"Wrote {len(selected)} candidates to {output}")
    print(f"Wrote {len(experimental)} experimental candidates to {experimental_path}")


if __name__ == "__main__":
    main()
