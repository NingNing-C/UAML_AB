"""Public command interface; numerical workflows are implemented in shared services."""
import argparse
import logging
import sys


def _option(parser, name, **kwargs):
    flags = ["--" + name.replace("_", "-")]
    if "_" in name:
        flags.append("--" + name)
    return parser.add_argument(*flags, dest=name, **kwargs)


def _training_options(parser):
    _option(parser, "encoding", choices=["onehot", "esm2"], help="Default from the case YAML.")
    _option(parser, "epochs", type=int, help="Default from the selected model configuration.")
    _option(parser, "n_train", type=int, help="Cap development observations before taking validation; -1 means all.")
    _option(parser, "output_dir", default="results")
    _option(parser, "device", default="auto")
    _option(parser, "initialization_seed", type=int, help="Override initialization; default preserves source policy.")
    _option(parser, "split_method", choices=["random", "mutation_number"], default="random")
    _option(parser, "mutation_split_setting", help="For example train_1_2_test_3+.")
    _option(parser, "test_on_rest", action="store_true")
    _option(parser, "hpo", action="store_true")
    _option(parser, "hpo_trials", type=int, default=20)
    _option(parser, "hpo_epochs", type=int, default=50)
    _option(parser, "hpo_bnn_epochs", type=int, default=100)


def parser():
    root = argparse.ArgumentParser(prog="uaml-ab", description="UAML-AB models, benchmarks and sequence design.")
    root.add_argument("--verbose", action="store_true")
    commands = root.add_subparsers(dest="command")
    training = commands.add_parser("train", help="Train any one of the eleven reported models.")
    _option(training, "config", required=True)
    _option(training, "method", default="ModelArcEnsemble")
    _option(training, "seed", type=int, default=42)
    _option(training, "noise_level", type=float, default=0.0)
    _training_options(training)
    benchmark = commands.add_parser("benchmark", help="Sweep models and datasets through the same train service.")
    _option(benchmark, "configs", nargs="+", required=True)
    _option(benchmark, "methods", nargs="+", help="Default all eleven; 'all' is also accepted.")
    _option(benchmark, "seeds", nargs="+", type=int, default=[42])
    _option(benchmark, "noise_levels", nargs="+", type=float, default=[0.0])
    _training_options(benchmark)
    for command, description in (("predict", "Predict unlabeled sequences using any reported model."),
                                 ("evaluate", "Evaluate a saved model on its held-out rows or explicit external data.")):
        selected = commands.add_parser(command, help=description)
        selected.add_argument("--checkpoint", "--model-path", "--model_path", dest="checkpoint", required=True)
        _option(selected, "config", help="Override checkpoint dataset configuration, e.g. after moving data.")
        _option(selected, "device", default="auto")
        _option(selected, "seed", type=int, help="Override inference seed; legacy raw weights also use it for split reconstruction.")
        _option(selected, "sequence_col")
        if command == "predict":
            _option(selected, "input", required=True, help="CSV or a compatible ESM2 HDF5 cache.")
            _option(selected, "output", default="results/predictions.csv")
        else:
            selected.add_argument("--external-data", "--external_data_path", dest="external_data")
            _option(selected, "label_col")
            destination = selected.add_mutually_exclusive_group()
            _option(destination, "output")
            _option(destination, "output_dir")
            _option(selected, "n_train", type=int, help="Legacy raw checkpoints only.")
            _option(selected, "split_method", choices=["random", "mutation_number"], default="random")
            _option(selected, "mutation_split_setting")
    commands.add_parser("models", help="List the reported methods and their capabilities.")
    for command, help_text in (("embed", "Generate optional frozen ESM2 features."),
                               ("design", "Run genetic search using frozen UAML-AB models."),
                               ("shortlist", "Pool and select genetic-search candidates."),
                               ("bo", "Run retrospective single- or two-objective Bayesian optimization.")):
        commands.add_parser(command, help=help_text, add_help=False)
    return root


def _delegate(command, arguments):
    if command == "embed":
        from .embeddings import main
    elif command == "design":
        from .design.ga.run import main
    elif command == "shortlist":
        from .design.ga.select import main
    else:
        from .design.bo.run import main
    result = main(arguments)
    return result if isinstance(result, int) else 0


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    verbose = "--verbose" in arguments
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    # Design/embedding parsers own their detailed scientific options.
    delegated = arguments[1:] if arguments and arguments[0] == "--verbose" else arguments
    try:
        if delegated and delegated[0] in {"embed", "design", "shortlist", "bo"}:
            return _delegate(delegated[0], delegated[1:])
        command_parser = parser()
        args = vars(command_parser.parse_args(arguments))
        command = args.pop("command")
        args.pop("verbose")
        if command is None:
            command_parser.print_help()
            return 0
        if command == "models":
            from .models.registry import METHODS, method_spec
            for name in METHODS:
                spec = method_spec(name)
                print(f"{name}\tuncertainty={spec.uncertainty}\tinputs={spec.representation}")
            return 0
        from . import experiments
        if command == "train":
            args["config_path"] = args.pop("config")
            experiments.train(**args)
        elif command == "benchmark":
            args["config_paths"] = args.pop("configs")
            if args["methods"] == ["all"]:
                args["methods"] = None
            experiments.benchmark(**args)
        elif command == "predict":
            args["config_path"] = args.pop("config")
            args["input_path"] = args.pop("input")
            experiments.predict(**args)
        else:
            args["config_path"] = args.pop("config")
            experiments.evaluate(**args)
    except Exception as error:
        if verbose:
            logging.exception("Command failed")
        else:
            logging.error("%s: %s", type(error).__name__, error)
        return 1
    return 0
