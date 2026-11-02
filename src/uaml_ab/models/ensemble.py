"""Six-member UAML-AB ensemble; checkpoint keys match the source release."""
from uaml_ab.models.layers import BaseModel, SimpleNN, SequenceCNN, DeepCNN, DeepNN
import torch
import torch.nn as nn
import numpy as np

class ModelArcEnsembleModel(BaseModel):

    def __init__(self, input_size, input_dim=24, seed=42, hidden_size: int=128, num_filters: int=32, output_size: int=2):
        super(ModelArcEnsembleModel, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        nn1 = SimpleNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        nn2 = DeepNN(input_size=input_size, hidden_sizes=[hidden_size, hidden_size // 2], output_size=output_size)
        nn3 = DeepNN(input_size=input_size, hidden_sizes=[hidden_size, hidden_size // 2, hidden_size // 4], output_size=output_size)
        cnn1 = SequenceCNN(input_dim=input_dim, num_filters=num_filters, filter_size=3, output_dim=output_size)
        cnn2 = DeepCNN(input_dim=input_dim, num_filters=num_filters, filter_sizes=[3, 5], output_dim=output_size)
        cnn3 = DeepCNN(input_dim=input_dim, num_filters=num_filters, filter_sizes=[3, 5, 7], output_dim=output_size)
        self.models = nn.ModuleList([nn1, nn2, nn3, cnn1, cnn2, cnn3])

    def forward(self, x_nn, x_cnn):
        outputs = torch.stack([model(x_nn) if i < 3 else model(x_cnn) for i, model in enumerate(self.models)], dim=0)
        return outputs
