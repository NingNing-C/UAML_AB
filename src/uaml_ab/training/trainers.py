"""Original reported-method trainer bodies with W&B/TensorBoard calls removed.

Legacy early stopping and evaluation loss conventions are retained for source
compatibility; see the release method notes before claiming exact reproduction.
ConcreteDropout's uninitialized regularization slots are fixed in its model.
The deterministic trainer's legacy validation loss compares [B,1] predictions
with [B] labels through PyTorch broadcasting. That stopping signal is retained
for source compatibility; prediction and reported metrics still use [B] means.
"""
import importlib
import logging
import numpy as np
import torch
import torch.nn as nn
from uaml_ab.metrics import spearman
from uaml_ab.training.losses import NegativeLogLikelihoodLoss

class _LazyModule:

    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        try:
            module = importlib.import_module(self.name)
        except ImportError as error:
            raise ImportError('Install optional benchmark dependencies: pip install .[benchmarks]') from error
        return getattr(module, name)
gpytorch = _LazyModule('gpytorch')
bnn = _LazyModule('torchbnn')

class Trainer:

    def __init__(self, model, criterion, optimizer, device, log_dir='logs/training/', patience=5, deterministic=False, likelihood=None, uncertainty=False, early_stop=True):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.early_stop_counter = 0
        self.best_loss = np.inf
        self.best_spearman = -np.inf
        self.deterministic = deterministic
        self.likelihood = likelihood
        self.uncertainty = uncertainty
        self.epoch = 0
        self.early_stop = early_stop

    def train_step(self, idx, data, labels):
        self.optimizer.zero_grad()
        outputs = self.model(data)
        if self.uncertainty:
            means = outputs[:, 0]
            log_stds = outputs[:, 1]
            stds = torch.exp(log_stds)
            vars = stds ** 2
            loss = self.criterion(means, vars, labels)
        else:
            loss = self.criterion(outputs.squeeze(-1), labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        if self.likelihood:
            self.likelihood.train()
        total_loss = 0
        for batch_idx, (idx, data, labels) in enumerate(train_loader):
            idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
            loss = self.train_step(idx, data, labels)
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def train(self, train_loader, val_loader=None, epochs=100):
        val_loss = None
        for epoch in range(epochs):
            self.epoch = epoch
            avg_loss = self.train_epoch(train_loader, epoch)
            logging.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}')
            if val_loader is not None:
                if self.uncertainty:
                    val_loss, _, _, _, _, _, spearman = self.evaluate(val_loader)
                else:
                    val_loss, _, _, spearman = self.evaluate(val_loader)
                if self.early_stop:
                    if val_loss < 0 or spearman < self.best_spearman:
                        self.early_stop_counter += 1
                        logging.info(f'Early stopping counter: {self.early_stop_counter}/{self.patience}')
                    elif val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.best_spearman = spearman
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        logging.info(f'Early stopping counter: {self.early_stop_counter}/{self.patience}')
                    if self.early_stop_counter >= self.patience:
                        logging.info('Early stopping triggered.')
                        break

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        total_preds = []
        total_labels = []
        if self.likelihood:
            self.likelihood.eval()
        with torch.no_grad():
            for idx, data, labels in test_loader:
                idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
                preds = self.model(data)
                loss = self.criterion(preds, labels)
                preds = preds.squeeze(-1)
                total_loss += loss.item()
                total_preds.append(preds.cpu().numpy())
                total_labels.append(labels.cpu().numpy())
        avg_loss = total_loss / len(test_loader)
        total_preds = np.concatenate(total_preds)
        total_labels = np.concatenate(total_labels)
        spearman_corr = spearman(total_preds, total_labels)
        logging.info(f'Evaluation Loss: {avg_loss:.4f}; Spearman correlation: {spearman_corr:.4f}')
        return (avg_loss, total_preds, total_labels, spearman_corr)

class GPTrainer(Trainer):

    def __init__(self, model, likelihood, criterion, optimizer, device, log_dir='logs/training/', patience=5, deterministic=False, uncertainty=False):
        super().__init__(model, criterion, optimizer, device, log_dir, patience, deterministic, likelihood, uncertainty)
        if likelihood is None:
            raise ValueError('Likelihood must be provided for GP models.')
        self.likelihood = likelihood

    def train_step(self, idx, data, labels):
        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = -gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, test_loader):
        self.model.eval()
        self.likelihood.eval()
        total_loss = 0
        total_means = []
        total_epistds = []
        total_alestds = []
        total_totalstds = []
        total_labels = []
        with torch.no_grad():
            for idx, data, labels in test_loader:
                idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
                preds = self.model(data)
                likelihood_preds = self.likelihood(preds)
                pred_mean = likelihood_preds.mean
                pred_var = likelihood_preds.variance
                loss = self.criterion(pred_mean, pred_var, labels, idx=idx)
                total_loss += loss.item()
                means = preds.mean.cpu().numpy()
                epistds = preds.stddev
                alestds = self.likelihood.noise.sqrt().expand_as(epistds)
                totalstds = likelihood_preds.stddev
                totalstds = totalstds.cpu().numpy()
                epistds = epistds.cpu().numpy()
                alestds = alestds.cpu().numpy()
                total_alestds.append(alestds)
                total_epistds.append(epistds)
                total_means.append(means)
                total_labels.append(labels.cpu().numpy())
                total_totalstds.append(totalstds)
        avg_loss = total_loss / len(test_loader)
        total_means = np.concatenate(total_means)
        total_epistds = np.concatenate(total_epistds)
        total_alestds = np.concatenate(total_alestds)
        total_labels = np.concatenate(total_labels)
        total_totalstds = np.concatenate(total_totalstds)
        spearman_corr = spearman(total_means, total_labels)
        logging.info(f'Evaluation Loss: {avg_loss:.4f}; Spearman correlation: {spearman_corr:.4f}')
        return (avg_loss, total_means, total_epistds, total_alestds, total_totalstds, total_labels, spearman_corr)

class MVETrainer(Trainer):

    def __init__(self, model, criterion, optimizer, device, log_dir='logs/training/', patience=5, deterministic=True, likelihood=None, uncertainty=True):
        super().__init__(model, criterion, optimizer, device, log_dir, patience, deterministic, likelihood, uncertainty, early_stop=False)
        self.uncertainty = uncertainty

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        total_means = []
        total_stds = []
        total_labels = []
        with torch.no_grad():
            for idx, data, labels in test_loader:
                idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
                preds = self.model(data)
                means = preds[:, 0]
                log_stds = preds[:, 1]
                stds = torch.exp(log_stds)
                vars = stds ** 2
                loss = self.criterion(means, vars, labels)
                total_loss += loss.item()
                means = means.cpu().numpy()
                stds = stds.cpu().numpy()
                total_means.append(means)
                total_stds.append(stds)
                total_labels.append(labels.cpu().numpy())
        avg_loss = total_loss / len(test_loader)
        total_means = np.concatenate(total_means)
        total_stds = np.concatenate(total_stds)
        total_labels = np.concatenate(total_labels)
        spearman_corr = spearman(total_means, total_labels)
        logging.info(f'Evaluation Loss: {avg_loss:.4f}; Spearman correlation: {spearman_corr:.4f}')
        return (avg_loss, total_means, None, None, total_stds, total_labels, spearman_corr)

class EnsembleTrainer(Trainer):

    def __init__(self, model, criterion, optimizer, device, log_dir='logs/training/', patience=5, deterministic=True, likelihood=None, uncertainty=True):
        super().__init__(model, criterion, optimizer, device, log_dir, patience, deterministic, likelihood, uncertainty)
        self.uncertainty = uncertainty
        self.models = model.models
        self.num_models = len(model.models)
        self.optimizer = optimizer

    def train_step(self, idx, data, label):
        total_loss = 0
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizer)):
            optimizer.zero_grad()
            preds = model(data)
            means = preds[:, 0]
            log_stds = preds[:, 1]
            stds = torch.exp(log_stds)
            vars = stds ** 2
            loss = self.criterion(means, vars, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / self.num_models

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        if self.likelihood:
            self.likelihood.train()
        total_loss = 0
        for batch_idx, (idx, data, labels) in enumerate(train_loader):
            idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
            loss = self.train_step(idx, data, labels)
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def evaluate(self, test_loader):
        self.model.eval()
        total_means = []
        total_labels = []
        total_epistds = []
        total_alestds = []
        total_totalstds = []
        total_loss = 0
        with torch.no_grad():
            for idx, data, labels in test_loader:
                idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
                preds = self.model(data)
                means = preds[:, :, 0]
                log_stds = preds[:, :, 1]
                stds = torch.exp(log_stds)
                vars = stds ** 2
                mean_preds = means.mean(0)
                epistemic_stds = means.std(0)
                aleatoric_stds = torch.sqrt(vars.mean(0))
                total_stds = torch.sqrt(epistemic_stds ** 2 + aleatoric_stds ** 2)
                loss = self.criterion(mean_preds, vars.mean(0), labels)
                total_loss += loss.item()
                total_means.append(mean_preds.cpu().numpy())
                total_epistds.append(epistemic_stds.cpu().numpy())
                total_alestds.append(aleatoric_stds.cpu().numpy())
                total_totalstds.append(total_stds.cpu().numpy())
                total_labels.append(labels.cpu().numpy())
        total_means = np.concatenate(total_means)
        total_labels = np.concatenate(total_labels)
        total_epistds = np.concatenate(total_epistds)
        total_alestds = np.concatenate(total_alestds)
        total_totalstds = np.concatenate(total_totalstds)
        avg_loss = total_loss / len(test_loader)
        spearman_corr = spearman(total_means, total_labels)
        logging.info(f'Evaluation Loss: {avg_loss:.4f}; Spearman correlation: {spearman_corr:.4f}')
        return (avg_loss, total_means, total_epistds, total_alestds, total_totalstds, total_labels, spearman_corr)

class BootstrapEnsembleTrainer(EnsembleTrainer):

    def __init__(self, model, criterion, optimizer, device, log_dir='logs/training/', patience=5, deterministic=True, likelihood=None, uncertainty=True):
        super().__init__(model, criterion, optimizer, device, log_dir, patience, deterministic, likelihood, uncertainty)
        self.uncertainty = uncertainty
        self.models = model.models
        self.num_models = len(model.models)
        self.optimizer = optimizer
        self.weights = model.weights
        self.criterions = criterion
        self.criterion = NegativeLogLikelihoodLoss()

    def train_step(self, idx, data, label):
        total_loss = 0
        for i, (model, optimizer, criterion) in enumerate(zip(self.models, self.optimizer, self.criterions)):
            optimizer.zero_grad()
            preds = model(data)
            means = preds[:, 0]
            log_stds = preds[:, 1]
            stds = torch.exp(log_stds)
            vars = stds ** 2
            loss = criterion(means, vars, label, idx=idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / self.num_models

class BNNTrainer(Trainer):

    def __init__(self, model, criterion, optimizer, device, log_dir='logs/training/', patience=5, deterministic=False, likelihood=None, uncertainty=True, num_samples=10, early_stop=False):
        super().__init__(model, criterion, optimizer, device, log_dir, patience, deterministic, likelihood, uncertainty, early_stop=False)
        self.num_samples = num_samples
        self.epoch = 0
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

    def train_step(self, idx, data, labels):
        self.optimizer.zero_grad()
        total_nll = 0
        for _ in range(self.num_samples):
            preds = self.model(data)
            means = preds[:, 0]
            log_stds = preds[:, 1]
            stds = torch.exp(log_stds)
            vars = stds ** 2
            vars = torch.clamp(vars, min=1e-06)
            if torch.isnan(vars).any() or torch.isnan(means).any():
                print('NaNs detected in model output')
                print('pred_mean:', means)
                print('pred_var:', vars)
            nll = self.criterion(means, vars, labels, idx=idx)
            total_nll += nll
        kl_loss = self.kl_loss(self.model)
        avg_nll = total_nll / self.num_samples
        beta = 10
        loss = avg_nll + beta * kl_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, test_loader):
        self.model.eval()
        total_means = []
        total_epistds = []
        total_alestds = []
        total_totalstds = []
        total_labels = []
        total_nll = 0
        total_loss = 0
        with torch.no_grad():
            for idx, data, labels in test_loader:
                idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
                preds_list = []
                vars_list = []
                for _ in range(self.num_samples):
                    preds = self.model(data)
                    means = preds[:, 0]
                    log_stds = preds[:, 1]
                    stds = torch.exp(log_stds)
                    vars = stds ** 2
                    total_nll += self.criterion(means, vars, labels, idx=idx)
                    preds_list.append(means)
                    vars_list.append(vars)
                kl_loss = self.kl_loss(self.model)
                beta = 10
                preds_tensor = torch.stack(preds_list)
                vars_tensor = torch.stack(vars_list)
                mean_preds = preds_tensor.mean(0)
                epistemic_stds = preds_tensor.std(0)
                aleatoric_stds = torch.sqrt(vars_tensor.mean(0))
                total_stds = torch.sqrt(epistemic_stds ** 2 + aleatoric_stds ** 2)
                total_means.append(mean_preds.cpu().numpy())
                total_epistds.append(epistemic_stds.cpu().numpy())
                total_alestds.append(aleatoric_stds.cpu().numpy())
                total_totalstds.append(total_stds.cpu().numpy())
                total_labels.append(labels.cpu().numpy())
                loss = total_nll / self.num_samples + kl_loss * beta
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        total_means = np.concatenate(total_means)
        total_epistds = np.concatenate(total_epistds)
        total_alestds = np.concatenate(total_alestds)
        total_totalstds = np.concatenate(total_totalstds)
        total_labels = np.concatenate(total_labels)
        spearman_corr = spearman(total_means, total_labels)
        logging.info(f'Evaluation Loss: {avg_loss:.4f}; Spearman correlation: {spearman_corr:.4f}')
        return (avg_loss, total_means, total_epistds, total_alestds, total_totalstds, total_labels, spearman_corr)

    def kl_annealing_beta(self):
        return min(1.0, self.epoch / self.kl_annealing_epochs)

class ConcreteDropoutTrainer(Trainer):

    def __init__(self, model, criterion, optimizer, device, log_dir='logs/training/', patience=5, num_samples=10, early_stop=True, uncertainty=True):
        super().__init__(model, criterion, optimizer, device, log_dir, patience, deterministic=False, uncertainty=True, early_stop=early_stop)
        self.num_samples = num_samples

    def train_step(self, idx, data, labels):
        self.optimizer.zero_grad()
        total_nll = 0
        total_reg = 0
        outputs, regularization = self.model(data)
        means = outputs[:, 0]
        log_stds = outputs[:, 1]
        stds = torch.exp(log_stds)
        vars = stds ** 2
        vars = torch.clamp(vars, min=1e-06)
        nll = self.criterion(means, vars, labels, idx=idx)
        total_nll += nll
        total_reg += regularization
        loss = total_nll + total_reg
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, test_loader):
        self.model.eval()
        total_means, total_epistds, total_alestds, total_totalstds, total_labels = ([], [], [], [], [])
        total_loss = 0
        with torch.no_grad():
            for idx, data, labels in test_loader:
                idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
                total_nll = 0
                total_reg = 0
                mean_samples = []
                var_samples = []
                for _ in range(self.num_samples):
                    preds, regularization = self.model(data)
                    means = preds[:, 0]
                    log_stds = preds[:, 1]
                    stds = torch.exp(log_stds)
                    vars = stds ** 2
                    nll = self.criterion(means, vars, labels, idx=idx)
                    total_nll += nll
                    total_reg += regularization
                    mean_samples.append(means)
                    var_samples.append(vars)
                loss = (total_nll + total_reg) / self.num_samples
                mean_samples = torch.stack(mean_samples)
                var_samples = torch.stack(var_samples)
                mean_pred = mean_samples.mean(0)
                epistemic_std = mean_samples.std(0)
                aleatoric_std = torch.sqrt(var_samples.mean(0))
                total_std = torch.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)
                total_means.append(mean_pred.cpu().numpy())
                total_epistds.append(epistemic_std.cpu().numpy())
                total_alestds.append(aleatoric_std.cpu().numpy())
                total_totalstds.append(total_std.cpu().numpy())
                total_labels.append(labels.cpu().numpy())
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        total_means = np.concatenate(total_means)
        total_epistds = np.concatenate(total_epistds)
        total_alestds = np.concatenate(total_alestds)
        total_totalstds = np.concatenate(total_totalstds)
        total_labels = np.concatenate(total_labels)
        spearman_corr = spearman(total_means, total_labels)
        logging.info(f'Evaluation Loss: {avg_loss:.4f}; Spearman correlation: {spearman_corr:.4f}')
        return (avg_loss, total_means, total_epistds, total_alestds, total_totalstds, total_labels, spearman_corr)

class EvidentialTrainer(Trainer):

    def __init__(self, model, criterion, optimizer, device, log_dir='logs/training/', patience=5, num_samples=1, early_stop=True, uncertainty=True):
        super().__init__(model, criterion, optimizer, device, log_dir, patience, deterministic=True, uncertainty=uncertainty, early_stop=early_stop)
        self.num_samples = num_samples

    def train_step(self, idx, data, labels):
        self.optimizer.zero_grad()
        preds = self.model(data)
        mu, v, alpha, beta = (preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3])
        loss = self.criterion(mu, v, alpha, beta, labels, idx=idx)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        total_means = []
        total_epistds = []
        total_alestds = []
        total_totalstds = []
        total_labels = []
        with torch.no_grad():
            for idx, data, labels in test_loader:
                idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
                preds = self.model(data)
                mu, v, alpha, beta = (preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3])
                loss = self.criterion(mu, v, alpha, beta, labels, idx=idx)
                ale_std = torch.sqrt(beta / (alpha - 1))
                epi_std = torch.sqrt(beta / (v * (alpha - 1)))
                total_std = torch.sqrt(ale_std ** 2 + epi_std ** 2)
                total_means.append(mu.cpu().numpy())
                total_epistds.append(epi_std.cpu().numpy())
                total_alestds.append(ale_std.cpu().numpy())
                total_totalstds.append(total_std.cpu().numpy())
                total_labels.append(labels.cpu().numpy())
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        total_means = np.concatenate(total_means)
        total_epistds = np.concatenate(total_epistds)
        total_alestds = np.concatenate(total_alestds)
        total_totalstds = np.concatenate(total_totalstds)
        total_labels = np.concatenate(total_labels)
        spearman_corr = spearman(total_means, total_labels)
        logging.info(f'Evaluation Loss: {avg_loss:.4f}; Spearman correlation: {spearman_corr:.4f}')
        return (avg_loss, total_means, total_epistds, total_alestds, total_totalstds, total_labels, spearman_corr)

class NeuralLinearTrainer(Trainer):
    """
    Variational trainer for NeuralLinearModel and BayesianNeuralLinearModel.
    ELBO = E_q[NLL] + kl_weight * KL(q||p), with proper minibatch scaling
    kl_weight = 1 / N_train (pass num_train). Epistemic uncertainty comes from
    sampling the Bayesian layers; aleatoric from the predicted (log_)std head.
    """

    def __init__(self, model, criterion, optimizer, device, log_dir='logs/training/', patience=5, uncertainty=True, num_samples=20, kl_weight=None, num_train=None, early_stop=True):
        super().__init__(model, criterion, optimizer, device, log_dir, patience, deterministic=False, likelihood=None, uncertainty=True, early_stop=early_stop)
        self.num_samples = num_samples
        self.kl_loss = bnn.BKLLoss(reduction='sum', last_layer_only=False)
        if kl_weight is None:
            kl_weight = 1.0 / num_train if num_train else 1.0
        self.kl_weight = kl_weight

    def train_step(self, idx, data, labels):
        self.optimizer.zero_grad()
        preds = self.model(data)
        means = preds[:, 0]
        stds = torch.exp(preds[:, 1])
        vars = torch.clamp(stds ** 2, min=1e-06)
        nll = self.criterion(means, vars, labels)
        kl = self.kl_loss(self.model)
        loss = nll + self.kl_weight * kl
        loss.backward()
        self.optimizer.step()
        self._last_nll = nll.item()
        self._last_kl_weighted = (self.kl_weight * kl).item()
        return loss.item()

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss, total_nll, total_kl_weighted = (0, 0, 0)
        for batch_idx, (idx, data, labels) in enumerate(train_loader):
            idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
            loss = self.train_step(idx, data, labels)
            total_loss += loss
            total_nll += self._last_nll
            total_kl_weighted += self._last_kl_weighted
        avg_loss = total_loss / len(train_loader)
        avg_nll = total_nll / len(train_loader)
        avg_kl_weighted = total_kl_weighted / len(train_loader)
        return avg_loss

    def evaluate(self, test_loader):
        self.model.eval()
        total_means, total_epistds, total_alestds = ([], [], [])
        total_totalstds, total_labels = ([], [])
        total_loss = 0
        total_nll = 0
        with torch.no_grad():
            for idx, data, labels in test_loader:
                idx, data, labels = (idx.to(self.device), data.to(self.device), labels.to(self.device))
                mean_samples, var_samples = ([], [])
                nll_accum = 0.0
                for _ in range(self.num_samples):
                    preds = self.model(data)
                    means = preds[:, 0]
                    vars = torch.clamp(torch.exp(preds[:, 1]) ** 2, min=1e-06)
                    nll_accum += self.criterion(means, vars, labels)
                    mean_samples.append(means)
                    var_samples.append(vars)
                mean_samples = torch.stack(mean_samples)
                var_samples = torch.stack(var_samples)
                mean_pred = mean_samples.mean(0)
                epistemic_std = mean_samples.std(0)
                aleatoric_std = torch.sqrt(var_samples.mean(0))
                total_std = torch.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)
                kl = self.kl_loss(self.model)
                avg_nll_batch = nll_accum / self.num_samples
                loss = avg_nll_batch + self.kl_weight * kl
                total_loss += loss.item()
                total_nll += avg_nll_batch.item()
                total_means.append(mean_pred.cpu().numpy())
                total_epistds.append(epistemic_std.cpu().numpy())
                total_alestds.append(aleatoric_std.cpu().numpy())
                total_totalstds.append(total_std.cpu().numpy())
                total_labels.append(labels.cpu().numpy())
        avg_loss = total_loss / len(test_loader)
        avg_nll = total_nll / len(test_loader)
        kl_weighted = (self.kl_weight * self.kl_loss(self.model)).item()
        total_means = np.concatenate(total_means)
        total_epistds = np.concatenate(total_epistds)
        total_alestds = np.concatenate(total_alestds)
        total_totalstds = np.concatenate(total_totalstds)
        total_labels = np.concatenate(total_labels)
        spearman_corr = spearman(total_means, total_labels)
        logging.info(f'Evaluation Loss: {avg_loss:.4f}; Spearman correlation: {spearman_corr:.4f}')
        return (avg_loss, total_means, total_epistds, total_alestds, total_totalstds, total_labels, spearman_corr)
