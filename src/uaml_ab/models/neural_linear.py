"""Reported benchmark architecture extracted from the source implementation."""
from uaml_ab.models.layers import BaseModel
import torch
import torch.nn as nn
import torchbnn as bnn

class _NeuralLinearBase(BaseModel):
    """
    Shared implementation. `probabilistic_backbone` toggles between the neural
    linear model (False) and the fully Bayesian network (True), mirroring
    ProbFlow's `probabilistic` keyword on DenseNetwork.
    """

    def __init__(self, input_size, hidden_size=128, output_size=1, hidden_sizes=None, prior_mu=0.0, prior_sigma=0.1, std_bias_init=-2.0, probabilistic_backbone=False):
        super().__init__()
        assert output_size == 1, f"NeuralLinear models expect output_size == 1 (scalar fitness). Got output_size={output_size}. Set 'output_size: 1' in the config for NeuralLinear / BayesianNeuralLinear."
        if hidden_sizes is None:
            hidden_sizes = [hidden_size, max(hidden_size // 2, 16)]

        def dense(in_f, out_f):
            if probabilistic_backbone:
                return bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=in_f, out_features=out_f)
            return nn.Linear(in_f, out_f)
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [dense(prev, h), nn.ReLU()]
            prev = h
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_dim = prev
        self.loc_head = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=prev, out_features=output_size)
        self.std_head = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=prev, out_features=output_size)
        self._init_small_std_head(std_bias_init)

    def _init_small_std_head(self, bias_init):
        """
        Initialise the std head to emit small, near-deterministic values early in
        training (ProbFlow's tip). With the log-std parameterisation used here,
        a negative bias -> small std = exp(bias), and a tight weight posterior
        keeps the noise from overwhelming the signal before the means are learned.
        """
        with torch.no_grad():
            if hasattr(self.std_head, 'weight_mu'):
                self.std_head.weight_mu.fill_(0.0)
            if hasattr(self.std_head, 'weight_log_sigma'):
                self.std_head.weight_log_sigma.fill_(-5.0)
            if getattr(self.std_head, 'bias_mu', None) is not None:
                self.std_head.bias_mu.fill_(bias_init)
            if getattr(self.std_head, 'bias_log_sigma', None) is not None:
                self.std_head.bias_log_sigma.fill_(-5.0)

    def forward(self, x):
        h = self.feature_extractor(x)
        loc = self.loc_head(h)
        log_std = self.std_head(h)
        return torch.cat([loc, log_std], dim=-1)

    def get_latent_rep(self, x):
        """Last-hidden-layer features, for parity with the other models here."""
        return self.feature_extractor(x)

class NeuralLinearModel(_NeuralLinearBase):
    """Deterministic backbone + Bayesian last-layer heads."""

    def __init__(self, input_size, hidden_size=128, output_size=1, **kwargs):
        super().__init__(input_size, hidden_size, output_size, probabilistic_backbone=False, **kwargs)
