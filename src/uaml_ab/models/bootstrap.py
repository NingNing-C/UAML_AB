"""Reported benchmark architecture extracted from the source implementation."""
from uaml_ab.models.layers import BaseModel, SimpleNN
import torch
import torch.nn as nn
from torch.distributions import Poisson
import numpy as np

class BootstrapEnsembleModel(BaseModel):

    def __init__(self, input_size, hidden_size, output_size, num_models, train_x, seed=42):
        super(BootstrapEnsembleModel, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.models = nn.ModuleList([SimpleNN(input_size, hidden_size, output_size) for _ in range(num_models)])
        self.weights = Poisson(rate=1).sample([num_models, train_x.shape[0]])

    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models], dim=0)
        return outputs
