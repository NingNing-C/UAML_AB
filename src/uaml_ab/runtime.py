"""Shared runtime for the eleven reported methods and all UAML-AB workflows.

Importing this module needs only the core environment. scikit-learn, GPyTorch,
and torchbnn are imported only when the selected method requires them. Fitting
uses training and validation observations only. A measured WT is not required.
"""
from pathlib import Path
import numpy as np
import torch
from uaml_ab.data import get_data_loader
from uaml_ab.models.layers import SimpleNN, SequenceCNN
from uaml_ab.models.ensemble import ModelArcEnsembleModel
from uaml_ab.training.ensemble_trainer import ModelArcEnsembleTrainer
from uaml_ab.training.losses import NegativeLogLikelihoodLoss, evidential_loss, taper_hidden_sizes, compute_std_bias_init

from uaml_ab.models.registry import (METHODS, DETERMINISTIC_METHODS, STOCHASTIC_METHODS,
                                     default_settings, method_spec)


def _require(module):
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError("Install optional benchmark dependencies: pip install .[benchmarks]") from error


def _loader(df, batch_size, encoding, cnn=False, include_labels=True):
    if len(df) == 0:
        raise ValueError("A model partition must contain at least one observation.")
    if encoding not in ("onehot", "esm2"):
        raise ValueError("encoding must be 'onehot' or 'esm2'.")
    encoder = encoding + ("_cnn" if cnn else "")
    return get_data_loader(df, batch_size, encoder, include_labels=include_labels)


def _paired_frames(df, df_cnn, encoding, include_labels=True):
    if df_cnn is None:
        if encoding == "esm2":
            raise ValueError("Position-wise ESM2 inputs (df_cnn) are required for a CNN/architecture ensemble.")
        return df
    if len(df) != len(df_cnn) or not df.index.equals(df_cnn.index):
        raise ValueError("Flat and position-wise inputs must contain the same row indices and order.")
    for name in ("mut_loci_seq", "row_id", "source_index", "row_index"):
        if name in df or name in df_cnn:
            if name not in df or name not in df_cnn or not np.array_equal(df[name].to_numpy(), df_cnn[name].to_numpy()):
                raise ValueError(f"Flat and position-wise identities must agree in {name}.")
    if encoding == "onehot" and not np.array_equal(df.sequence.to_numpy(), df_cnn.sequence.to_numpy()):
        raise ValueError("Flat and position-wise one-hot inputs must use the same sequences.")
    if include_labels and "label" in df and "label" in df_cnn and not np.array_equal(df.label.to_numpy(), df_cnn.label.to_numpy()):
        raise ValueError("Flat and position-wise input labels must have identical row order.")
    return df_cnn


def _build(method, settings, spec, device, seed, train=False, gp_train_x=None, gp_train_y=None, initialization_seed=None):
    """Construct only the requested reported model and its original trainer."""
    from uaml_ab.training import trainers
    criterion = NegativeLogLikelihoodLoss()
    size, channels = spec["input_size"], spec.get("input_dim", 24)
    hidden = settings.get("hidden_size", 128)
    initialization_seed = seed if initialization_seed is None else initialization_seed
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    model = None
    if method == "Linreg":
        model = _require("sklearn.linear_model").LinearRegression()
        return model, None
    if method in ("Deterministic", "MVE"):
        model = SimpleNN(size, hidden, 1 if method == "Deterministic" else 2)
    elif method == "CNN":
        model = SequenceCNN(channels, settings["num_filters"], settings["filter_size"], 2)
    elif method == "BootstrapEnsemble":
        from uaml_ab.models.bootstrap import BootstrapEnsembleModel
        model = BootstrapEnsembleModel(size, hidden, 2, settings["num_models"],
                                       np.empty((spec["train_count"], 0)), seed=initialization_seed)
    elif method == "ModelArcEnsemble":
        model = ModelArcEnsembleModel(size, input_dim=channels, seed=initialization_seed)
    elif method == "BNN":
        _require("torchbnn")
        from uaml_ab.models.bnn import BayesianNetwork
        model = BayesianNetwork(size, hidden, 2)
    elif method == "NeuralLinear":
        _require("torchbnn")
        from uaml_ab.models.neural_linear import NeuralLinearModel
        model = NeuralLinearModel(size, hidden, output_size=1,
                                  hidden_sizes=taper_hidden_sizes(hidden, settings.get("n_layers")),
                                  std_bias_init=spec.get("std_bias_init", -2.0))
    elif method == "ConcreteDropout":
        from uaml_ab.models.concrete_dropout import ConcreteDropoutModel
        model = ConcreteDropoutModel(input_dim=size, hidden_units=hidden, output_dim=2,
                                      weight_regularizer=settings["weight_regularizer"],
                                      dropout_regularizer=settings["dropout_regularizer"])
    elif method == "Evidential":
        from uaml_ab.models.evidential import EvidentialModel
        model = EvidentialModel(size, hidden, 4)
    elif method == "GP":
        gpytorch = _require("gpytorch")
        from uaml_ab.models.gp import GPRegressionModel
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = GPRegressionModel(gp_train_x.to(device), gp_train_y.to(device), likelihood)
    else:
        raise ValueError(f"Unknown benchmark method: {method}")
    model = model.to(device)
    if method in ("BootstrapEnsemble", "ModelArcEnsemble"):
        optimizer = [torch.optim.Adam(member.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
                     for member in model.models] if train else None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]) if train else None
    patience = settings.get("patience", 20 if method == "GP" else 10 if method == "BootstrapEnsemble" else 50)
    if method == "Deterministic":
        trainer = trainers.Trainer(model, torch.nn.MSELoss(), optimizer, device, patience=patience)
    elif method in ("MVE", "CNN"):
        trainer = trainers.MVETrainer(model, criterion, optimizer, device, patience=patience, uncertainty=True)
    elif method == "BootstrapEnsemble":
        criterions = [NegativeLogLikelihoodLoss(weight=model.weights[i].to(device)) for i in range(settings["num_models"])]
        trainer = trainers.BootstrapEnsembleTrainer(model, criterions, optimizer, device, patience=patience, uncertainty=True)
    elif method == "ModelArcEnsemble":
        trainer = ModelArcEnsembleTrainer(model, criterion, optimizer, device, patience=patience, uncertainty=True)
    elif method == "BNN":
        trainer = trainers.BNNTrainer(model, criterion, optimizer, device, patience=patience,
                                      uncertainty=True, num_samples=settings["num_samples"], early_stop=False)
    elif method == "NeuralLinear":
        trainer = trainers.NeuralLinearTrainer(model, criterion, optimizer, device, patience=patience,
                                               num_samples=settings["num_samples"], num_train=spec["train_count"], early_stop=True)
    elif method == "ConcreteDropout":
        trainer = trainers.ConcreteDropoutTrainer(model, criterion, optimizer, device, patience=patience,
                                                  num_samples=settings["num_samples"], uncertainty=True)
    elif method == "Evidential":
        def criterion(mu, v, alpha, beta, targets, idx=None):
            return evidential_loss(mu, v, alpha, beta, targets, lam=1.0, idx=idx)
        trainer = trainers.EvidentialTrainer(model, criterion, optimizer, device, patience=patience, uncertainty=True)
    elif method == "GP":
        trainer = trainers.GPTrainer(model, model.likelihood, criterion, optimizer, device, patience=patience, uncertainty=True)
    return model, trainer


class FittedModel:
    """A fitted method with label-independent inference and central persistence."""
    def __init__(self, method, model, trainer, settings, spec, encoding, seed, device,
                 gp_train_x=None, gp_train_y=None, initialization_seed=None, metadata=None):
        self.method, self.model, self.trainer = method, model, trainer
        self.settings, self.spec, self.encoding = settings, spec, encoding
        self.seed, self.device = int(seed), torch.device(device)
        self.initialization_seed = int(seed if initialization_seed is None else initialization_seed)
        self.gp_train_x, self.gp_train_y = gp_train_x, gp_train_y
        self.metadata = dict(metadata or {})

    def predict(self, df, df_cnn=None, seed=None, batch_size=None):
        """Predict without reading labels, evaluating a loss, or computing metrics.

        A supplied inference seed uses an isolated RNG context. No seed preserves
        the source stochastic models' continuing PyTorch sampling convention.
        """
        batch = int(self.settings["batch_size"] if batch_size is None else batch_size)
        if batch < 1:
            raise ValueError("batch_size must be positive.")
        source = _paired_frames(df, df_cnn, self.encoding, include_labels=False) if self.method == "CNN" else df
        loader = _loader(source, batch, self.encoding, cnn=self.method == "CNN", include_labels=False)
        cnn_loader = None
        if self.method == "ModelArcEnsemble":
            cnn_df = _paired_frames(df, df_cnn, self.encoding, include_labels=False)
            cnn_loader = _loader(cnn_df, batch, self.encoding, cnn=True, include_labels=False)
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        if seed is None:
            return self._predict_loaders(loader, cnn_loader)
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(seed))
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(int(seed))
            return self._predict_loaders(loader, cnn_loader)

    def _predict_loaders(self, loader, cnn_loader=None):
        rows = []
        if self.method != "Linreg":
            self.model.eval()
        paired = iter(cnn_loader) if cnn_loader is not None else None
        with torch.no_grad():
            for indices, values in loader:
                if self.method == "Linreg":
                    rows.append({"pred": np.asarray(self.model.predict(values.numpy()))})
                    continue
                values = values.to(self.device)
                if self.method == "ModelArcEnsemble":
                    cnn_indices, cnn_values = next(paired)
                    if not torch.equal(indices, cnn_indices):
                        raise ValueError("Paired model inputs must have identical row order.")
                    outputs = self.model(values, cnn_values.to(self.device))
                elif self.method in STOCHASTIC_METHODS:
                    outputs = []
                    for _ in range(self.settings["num_samples"]):
                        sample = self.model(values)
                        outputs.append(sample[0] if self.method == "ConcreteDropout" else sample)
                    outputs = torch.stack(outputs)
                elif self.method == "GP":
                    latent = self.model(values)
                    observed = self.model.likelihood(latent)
                    rows.append({"pred": observed.mean.cpu().numpy(),
                                 "epistd": latent.stddev.cpu().numpy(),
                                 "alestd": self.model.likelihood.noise.sqrt().expand_as(latent.stddev).cpu().numpy(),
                                 "totalstd": observed.stddev.cpu().numpy()})
                    continue
                else:
                    outputs = self.model(values)
                result = self._aggregate(outputs)
                rows.append({name: value.cpu().numpy() for name, value in result.items() if value is not None})
        if not rows:
            raise ValueError("Prediction input is empty.")
        return {name: np.concatenate([row[name] for row in rows]) for name in rows[0]}

    def _aggregate(self, outputs):
        if self.method == "Deterministic":
            return {"pred": outputs.squeeze(-1)}
        if self.method in ("MVE", "CNN"):
            return {"pred": outputs[:, 0], "totalstd": torch.exp(outputs[:, 1])}
        if self.method == "Evidential":
            mu, v, alpha, beta = outputs.unbind(dim=1)
            ale = torch.sqrt(beta/(alpha-1))
            epi = torch.sqrt(beta/(v*(alpha-1)))
            return {"pred": mu, "epistd": epi, "alestd": ale, "totalstd": torch.sqrt(epi**2+ale**2)}
        member_means = outputs[:, :, 0]
        member_variances = torch.exp(outputs[:, :, 1])**2
        if self.method == "NeuralLinear":
            member_variances = member_variances.clamp(min=1e-6)
        pred = member_means.mean(0)
        epi = member_means.std(0)
        ale = member_variances.mean(0).sqrt()
        return {"pred": pred, "epistd": epi, "alestd": ale, "totalstd": torch.sqrt(epi**2+ale**2)}

    def save(self, path):
        from uaml_ab.checkpoints import save
        return save(self, path)

    @classmethod
    def load(cls, path, **kwargs):
        return load(path, **kwargs)


def fit(method, train_df, val_df, settings, device, seed, epochs, encoding="onehot",
        train_cnn=None, val_cnn=None, initialization_seed=None):
    """Fit a fresh method using only supplied training/validation partitions."""
    if method not in METHODS:
        raise ValueError(f"Unknown benchmark method {method!r}.")
    if epochs < 1 or min(len(train_df), len(val_df)) < 1:
        raise ValueError("Positive epochs and nonempty training/validation partitions are required.")
    settings = dict(default_settings(method), **(settings or {}))
    initialization_seed = (42 if method == "ModelArcEnsemble" else seed) if initialization_seed is None else initialization_seed
    settings["epochs"] = int(epochs)  # Record the explicit runtime limit in saved metadata.
    if method == "BootstrapEnsemble" and settings["num_models"] < 2:
        raise ValueError("Bootstrap epistemic uncertainty requires at least two ensemble members.")
    if settings["batch_size"] < 1:
        raise ValueError("batch_size must be positive.")
    if method in STOCHASTIC_METHODS and settings["num_samples"] < 2:
        raise ValueError("Stochastic uncertainty requires at least two prediction samples.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    batch = settings["batch_size"]
    train_source, val_source = train_df, val_df
    if method == "CNN":
        train_source = _paired_frames(train_df, train_cnn, encoding)
        val_source = _paired_frames(val_df, val_cnn, encoding)
    train_loader = _loader(train_source, len(train_source) if method in ("Linreg", "GP") else batch,
                           encoding, cnn=method == "CNN")
    val_loader = _loader(val_source, batch, encoding, cnn=method == "CNN")
    _, sample, _ = next(iter(train_loader))
    spec = {"input_size": int(sample.shape[1]), "train_count": len(train_df)}
    if method == "CNN":
        spec["input_dim"] = int(sample.shape[2])
    if method == "ModelArcEnsemble":
        train_cnn = _paired_frames(train_df, train_cnn, encoding)
        val_cnn = _paired_frames(val_df, val_cnn, encoding)
        train_loader_cnn = _loader(train_cnn, batch, encoding, cnn=True)
        val_loader_cnn = _loader(val_cnn, batch, encoding, cnn=True)
        _, sample_cnn, _ = next(iter(train_loader_cnn))
        spec["input_dim"] = int(sample_cnn.shape[2])
    if method == "NeuralLinear":
        spec["std_bias_init"] = compute_std_bias_init(train_df.label.to_numpy())
    gp_x = gp_y = None
    if method == "GP":
        _, gp_x, gp_y = next(iter(train_loader))
    model, trainer = _build(method, settings, spec, device, seed, train=True,
                            gp_train_x=gp_x, gp_train_y=gp_y, initialization_seed=initialization_seed)
    if method == "Linreg":
        _, x, y = next(iter(train_loader))
        model.fit(x.numpy(), y.numpy().ravel())
    elif method == "ModelArcEnsemble":
        trainer.train(train_loader, train_loader_cnn, val_loader, val_loader_cnn, epochs=epochs)
    else:
        trainer.train(train_loader, val_loader, epochs=epochs)
    return FittedModel(method, model, trainer, settings, spec, encoding, seed, device, gp_x, gp_y, initialization_seed=initialization_seed)


def load(path, device="cpu", config=None, reference_data=None, reference_cnn=None, payload=None):
    """Central checkpoint loader; supports canonical and legacy formats."""
    from uaml_ab.checkpoints import load as load_checkpoint
    return load_checkpoint(path, device=device, config=config, reference_data=reference_data,
                           reference_cnn=reference_cnn, payload=payload)
