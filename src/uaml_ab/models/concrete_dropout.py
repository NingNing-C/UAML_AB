"""Reported benchmark architecture extracted from the source implementation."""
from uaml_ab.models.layers import BaseModel
import torch
from torch import nn
import numpy as np

class ConcreteDropout(nn.Module):

    def __init__(self, weight_regularizer=1e-06, dropout_regularizer=1e-05, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)
        out = layer(self._concrete_dropout(x, p))
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1.0 - p) * torch.log(1.0 - p)
        input_dimensionality = x[0].numel()
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        regularization = weights_regularizer + dropout_regularizer
        return (out, regularization)

    def _concrete_dropout(self, x, p):
        eps = 1e-07
        temp = 0.1
        unif_noise = torch.rand_like(x)
        drop_prob = torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps)
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        x = torch.mul(x, random_tensor)
        x /= retain_prob
        return x

class ConcreteDropoutModel(BaseModel):

    def __init__(self, input_dim, output_dim, hidden_units, weight_regularizer, dropout_regularizer):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_dim)
        self.conc_drop1 = ConcreteDropout(weight_regularizer, dropout_regularizer)
        self.conc_drop2 = ConcreteDropout(weight_regularizer, dropout_regularizer)
        self.conc_drop3 = ConcreteDropout(weight_regularizer, dropout_regularizer)
        self.relu = nn.ReLU()

    def forward(self, x):
        regularization = torch.empty(3, device=x.device)
        x, regularization[0] = self.conc_drop1(x, nn.Sequential(self.fc1, self.relu))
        x, regularization[1] = self.conc_drop2(x, nn.Sequential(self.fc2, self.relu))
        x, regularization[2] = self.conc_drop3(x, self.fc3)
        return (x, regularization.sum())
