"""Original reported losses; no experiment tracking or plotting dependencies."""
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

def evidential_loss(mu, v, alpha, beta, targets, lam=1, epsilon=0.0001, idx=None):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parameter for NIG
    :targets: Outputs to predict

    :return: Loss
    """
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) - alpha * torch.log(twoBlambda) + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    L_NLL = nll
    error = torch.abs(targets - mu)
    reg = error * (2 * v + alpha)
    L_REG = reg
    loss = L_NLL + lam * (L_REG - epsilon)
    return torch.mean(loss)

def taper_hidden_sizes(hidden_size, n_layers, floor=16):
    if n_layers is None:
        return None
    return [max(hidden_size // 2 ** i, floor) for i in range(n_layers)]

def compute_std_bias_init(train_labels, min_std=0.001):
    """
    Data-driven init for the NeuralLinear/BayesianNeuralLinear std head: returns
    log(std(train_labels)), so the predicted variance starts near Var(y) instead
    of a fixed (and often far too small) constant. This avoids the NLL gradient
    inflating the variance head before the mean head has learned anything.
    """
    label_std = max(float(np.std(train_labels)), min_std)
    return float(np.log(label_std))

class NegativeLogLikelihoodLoss(nn.Module):

    def __init__(self, min_var=1e-05, weight=None):
        """
        Negative Log-Likelihood loss assuming a Gaussian distribution.
        
        Args:
            min_var (float): Minimum variance to prevent numerical instability.
            weight (Tensor, optional): Per-sample weight tensor.
        """
        super().__init__()
        self.min_var = min_var
        self.weight = weight

    def forward(self, pred_mean, pred_var, target, idx=None):
        """
        Compute the NLL loss.
        
        Args:
            pred_mean (Tensor): Predicted mean (μ).
            pred_var (Tensor): Predicted variance (σ²).
            target (Tensor): Ground truth (y).
        
        Returns:
            Tensor: The computed NLL loss.
        """
        clamped_var = torch.clamp(pred_var, min=self.min_var)
        pred_std = torch.sqrt(clamped_var)
        distribution = dist.Normal(pred_mean, pred_std)
        log_likelihood = distribution.log_prob(target)
        if self.weight is not None:
            if idx is not None:
                loss = -self.weight[idx] * log_likelihood
            else:
                loss = -self.weight * log_likelihood
        else:
            loss = -log_likelihood
        return loss.mean()
