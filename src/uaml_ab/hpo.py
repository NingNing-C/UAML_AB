"""One validation-only Optuna engine shared by single runs and benchmarks."""
import copy
import math

from .metrics import spearman
from .models.registry import suggest_settings


def cap_development(train, validation, train_cnn, validation_cnn, seed=42, cap=5000):
    if len(train) + len(validation) <= cap:
        return train, validation, train_cnn, validation_cnn
    count = max(2, round(cap * len(validation) / (len(train) + len(validation))))
    validation = validation.sample(n=min(count, len(validation)), random_state=seed)
    train = train.sample(n=min(cap - len(validation), len(train)), random_state=seed)
    return train, validation, train_cnn.loc[train.index], validation_cnn.loc[validation.index]


def tune(method, train, validation, settings, device, seed=42, *, encoding="onehot",
         train_cnn=None, validation_cnn=None, n_trials=20, epochs=50,
         bnn_epochs=100, initialization_seed=None):
    if method == "Linreg":
        return {}, {"status": "not_applicable", "trials": [],
                    "reason": "Ordinary least squares has no searched hyperparameters."}
    try:
        import optuna
    except ImportError as error:
        raise ImportError('Install HPO with pip install -e ".[hpo]".') from error
    from .runtime import fit
    if min(n_trials, epochs, bnn_epochs) < 1:
        raise ValueError("Trial and epoch limits must be positive.")
    train_cnn = train if train_cnn is None else train_cnn
    validation_cnn = validation if validation_cnn is None else validation_cnn
    train, validation, train_cnn, validation_cnn = cap_development(
        train, validation, train_cnn, validation_cnn, seed)
    if len(validation) < 2:
        raise ValueError("HPO requires at least two validation observations.")
    errors = {}
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial):
        candidate = copy.deepcopy(settings)
        candidate.update(suggest_settings(trial, method))
        candidate["patience"] = 10
        try:
            model = fit(method, train, validation, candidate, device, seed,
                        bnn_epochs if method == "BNN" else epochs, encoding=encoding,
                        train_cnn=train_cnn, val_cnn=validation_cnn,
                        initialization_seed=initialization_seed)
            result = model.predict(validation, df_cnn=validation_cnn, seed=seed)
            score = spearman(result["pred"], validation.label.to_numpy())
            if not math.isfinite(score):
                raise ValueError("Validation Spearman correlation is undefined.")
            return score
        except Exception as error:
            errors[trial.number] = f"{type(error).__name__}: {error}"
            raise

    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed:
        raise RuntimeError(f"All HPO trials failed for {method}: {errors}")
    report = {"status": "complete", "objective": "validation_spearman", "seed": seed,
              "n_train": len(train), "n_validation": len(validation),
              "epoch_limit": bnn_epochs if method == "BNN" else epochs,
              "best_value": study.best_value, "best_params": study.best_params,
              "trials": [{"number": trial.number, "state": trial.state.name,
                          "value": trial.value, "params": trial.params,
                          "error": errors.get(trial.number)} for trial in study.trials]}
    return dict(study.best_params), report
