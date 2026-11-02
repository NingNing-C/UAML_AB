"""Reported benchmark architecture extracted from the source implementation."""
from uaml_ab.models.layers import BaseModel
import torch.nn.functional as F
import torchbnn as bnn

class BayesianNetwork(BaseModel):

    def __init__(self, input_dim, hidden_units, output_dim):
        super().__init__()
        self.bnn1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_dim, out_features=hidden_units)
        self.output = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_units, out_features=2)

    def forward(self, x):
        x = F.relu(self.bnn1(x))
        x = self.output(x)
        return x
