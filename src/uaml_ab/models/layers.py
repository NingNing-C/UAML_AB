"""Original UAML-AB MLP and CNN architectures."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

class SimpleNN(BaseModel):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_latent_rep(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

class SequenceCNN(BaseModel):

    def __init__(self, input_dim, num_filters, filter_size, output_dim):
        super(SequenceCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=filter_size, padding=0)
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.max(x, dim=2)[0]
        x = self.fc(x)
        return x

    def get_latent_rep(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.max(x, dim=2)[0]
        return x

class DeepNN(BaseModel):

    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)])
        self.fc2 = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.fc2(x)
        return x

    def get_latent_rep(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return x

class DeepCNN(BaseModel):

    def __init__(self, input_dim, num_filters, filter_sizes, output_dim):
        super(DeepCNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=fs, padding=0) for fs in filter_sizes])
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
        x = torch.cat(x, dim=1)
        x = self.fc(x)
        return x

    def get_latent_rep(self, x):
        x = x.permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
        x = torch.cat(x, dim=1)
        return x
