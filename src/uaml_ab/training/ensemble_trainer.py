"""Focused UAML-AB ensemble training and uncertainty prediction.

The original per-member NLL, gradient clipping, sample standard deviation, and
legacy stopping rule are retained. For manuscript compatibility the validation
loss also retains the source's cumulative division by member count on each
batch. That value is a legacy stopping signal, not a conventional mean NLL.
The stopping rule counts negative validation losses as non-improvements even
though a Gaussian density can have negative NLL. No best-weight restoration is
performed. These choices are documented rather than silently changing training.
"""
import logging
from pathlib import Path
import numpy as np
import torch
from uaml_ab.metrics import spearman

class ModelArcEnsembleTrainer:

    def __init__(self, model, criterion, optimizer, device, log_dir='logs/training/', patience=50, deterministic=True, likelihood=None, uncertainty=True, early_stop=True):
        self.model = model.to(device)
        self.models = model.models
        self.num_models = len(model.models)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.uncertainty = uncertainty
        self.early_stop = early_stop
        self.early_stop_counter = 0
        self.best_loss = np.inf
        self.best_spearman = -np.inf
        self.history = []

    def train_step(self, data_nn, data_cnn, label):
        total_loss = 0.0
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizer)):
            optimizer.zero_grad()
            outputs = model(data_nn if i < 3 else data_cnn)
            means, log_stds = (outputs[:, 0], outputs[:, 1])
            variance = torch.exp(log_stds).clamp(max=1000000.0) ** 2
            loss = self.criterion(means, variance, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / self.num_models

    @staticmethod
    def _paired(nn_loader, cnn_loader):
        if len(nn_loader.dataset) != len(cnn_loader.dataset) or len(nn_loader) != len(cnn_loader):
            raise ValueError('MLP and CNN loaders must have the same sequences and batch sizes.')
        for (idx_nn, data_nn, labels_nn), (idx_cnn, data_cnn, labels_cnn) in zip(nn_loader, cnn_loader):
            if not torch.equal(idx_nn, idx_cnn) or not torch.equal(labels_nn, labels_cnn):
                raise ValueError('MLP and CNN loaders must use identical row order.')
            yield (data_nn, data_cnn, labels_nn)

    def train_epoch(self, train_loader_nn, train_loader_cnn, epoch):
        self.model.train()
        losses = [self.train_step(x.to(self.device), c.to(self.device), y.to(self.device)) for x, c, y in self._paired(train_loader_nn, train_loader_cnn)]
        if not losses:
            raise ValueError('Training partition is empty.')
        return float(np.mean(losses))

    def train(self, train_loader_nn, train_loader_cnn, val_loader_nn=None, val_loader_cnn=None, epochs=100):
        if epochs < 1:
            raise ValueError('epochs must be at least 1.')
        if not self.optimizer:
            raise ValueError('Training requires one optimizer per ensemble member.')
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader_nn, train_loader_cnn, epoch)
            record = {'epoch': epoch + 1, 'train_nll': loss}
            if val_loader_nn is not None and val_loader_cnn is not None:
                val_loss, _, _, _, _, _, correlation = self.evaluate(val_loader_nn, val_loader_cnn)
                record.update(val_loss_legacy=val_loss, val_spearman=correlation)
                if self.early_stop:
                    if val_loss < 0 or correlation < self.best_spearman:
                        self.early_stop_counter += 1
                    elif val_loss < self.best_loss:
                        self.best_loss, self.best_spearman = (val_loss, correlation)
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
            self.history.append(record)
            logging.info('Epoch %d/%d: train NLL %.6f', epoch + 1, epochs, loss)
            if self.early_stop and self.early_stop_counter >= self.patience:
                break
        return self.history

    def evaluate(self, test_loader_nn, test_loader_cnn):
        self.model.eval()
        rows = []
        legacy_loss = 0.0
        with torch.no_grad():
            for data_nn, data_cnn, labels in self._paired(test_loader_nn, test_loader_cnn):
                data_nn, data_cnn, labels = (data_nn.to(self.device), data_cnn.to(self.device), labels.to(self.device))
                outputs = self.model(data_nn, data_cnn)
                member_means = outputs[:, :, 0]
                member_variances = torch.exp(outputs[:, :, 1]) ** 2
                means = member_means.mean(0)
                epistemic = member_means.std(0)
                aleatoric = member_variances.mean(0).sqrt()
                total = (epistemic ** 2 + aleatoric ** 2).sqrt()
                legacy_loss = (legacy_loss + self.criterion(means, member_variances.mean(0), labels).item()) / self.num_models
                rows.append(tuple((t.cpu().numpy() for t in (means, epistemic, aleatoric, total, labels))))
        if not rows:
            raise ValueError('Evaluation partition is empty.')
        means, epistemic, aleatoric, total, labels = [np.concatenate(values) for values in zip(*rows)]
        return (legacy_loss / len(test_loader_nn), means, epistemic, aleatoric, total, labels, spearman(means, labels))
