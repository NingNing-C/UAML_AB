"""Run retrospective single- or two-objective Bayesian optimization."""
import argparse
import logging
import torch
from .loop import BOLoop
from .multi import MultiObjectiveBOLoop


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="uaml-ab bo", description=__doc__)
    configs = parser.add_mutually_exclusive_group(required=True)
    configs.add_argument("--configs", nargs="+", help="One or two onehot dataset YAML files.")
    configs.add_argument("--config", help="Single-objective dataset YAML (alias for one --configs value).")
    parser.add_argument("--antigen_names", nargs="+", help="Distinct short names, one per objective.")
    parser.add_argument("--combine", choices=["min", "mean"], default="min")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acquisition", choices=["greedy", "lcb", "ucb", "ei", "random"], default="ei")
    parser.add_argument("--unc_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=20, help="Measurements acquired per round.")
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--n_init", type=int, default=200)
    parser.add_argument("--init_max_mut", type=int)
    parser.add_argument("--held_out_frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, help="Override shared-runtime training epochs per round.")
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir")
    parser.add_argument("--out_group", default="_mo_bo_runs")
    parser.add_argument("--acq_sigma", choices=["epistd", "alestd", "totalstd"], default="totalstd")
    parser.add_argument("--acq_normalize", choices=["gate_margin", "zscore", "none"], default="gate_margin")
    parser.add_argument("--adaptive_weight", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    args.configs = args.configs or [args.config]
    if len(args.configs) not in (1, 2):
        parser.error("Supply one or two dataset configurations.")
    if args.antigen_names is not None and (len(args.antigen_names) != len(args.configs) or len(set(args.antigen_names)) != len(args.antigen_names)):
        parser.error("--antigen_names must provide one distinct name per objective.")
    if args.n_init < 6 or args.n_rounds < 1 or args.batch_size < 1:
        parser.error("--n_init must be at least 6; round and batch counts must be positive.")
    if not 0 < args.held_out_frac < 1 or args.unc_weight < 0:
        parser.error("held_out_frac must be in (0,1), and unc_weight must be nonnegative.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive.")
    return args


def run_bo(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    common = dict(acquisition=args.acquisition, unc_weight=args.unc_weight,
                  batch_size=args.batch_size, n_rounds=args.n_rounds, n_init=args.n_init,
                  held_out_frac=args.held_out_frac, epochs=args.epochs, seed=args.seed,
                  out_dir=args.out_dir, acq_sigma_col=args.acq_sigma,
                  acq_normalize=args.acq_normalize, adaptive_weight=args.adaptive_weight,
                  init_max_mut=args.init_max_mut, device=device)
    if len(args.configs) == 1:
        loop = BOLoop(args.configs[0], **common)
    else:
        loop = MultiObjectiveBOLoop(args.configs, antigen_names=args.antigen_names,
                                    combine=args.combine, out_group=args.out_group, **common)
    if not len(loop.pool.candidate_pool):
        raise ValueError("No candidate observations remain after initialization and holding out reference data.")
    results = loop.run()
    logging.info("BO results written to %s", loop.out_dir)
    return results


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_bo(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
