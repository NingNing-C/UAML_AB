"""Reported benchmark architecture extracted from the source implementation."""
from uaml_ab.models.layers import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialModel(BaseModel):

    def __init__(self, input_size, hidden_size, output_size):
        super(EvidentialModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        min_val = 1e-06
        mu, logv, logalpha, logbeta = (x[:, 0], x[:, 1], x[:, 2], x[:, 3])
        v = F.softplus(logv) + min_val
        alpha = F.softplus(logalpha) + min_val + 1
        beta = F.softplus(logbeta) + min_val
        return torch.stack([mu, v, alpha, beta], dim=1)
