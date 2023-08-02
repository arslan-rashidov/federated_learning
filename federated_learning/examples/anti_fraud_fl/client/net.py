from collections import OrderedDict

import torch
import torch.nn as nn

from federated_learning.core.utils import NDArrays


class Net(nn.Module):
    def __init__(self, n_features=30, hidden_dim=32):
        super().__init__()
        self.relu_activation = nn.ReLU()
        self.sigmoid_activation = nn.Sigmoid()

        self.first_conv_layer = nn.Conv1d(1, hidden_dim, 2)
        self.first_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.first_dropout = nn.Dropout(0.2)

        self.second_conv_layer = nn.Conv1d(hidden_dim, hidden_dim * 2, 2)
        self.second_batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.second_dropout = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        self.first_linear_layer = nn.Linear((n_features - 2) * (hidden_dim * 2), hidden_dim * 2)
        self.third_dropout = nn.Dropout(0.5)

        self.second_linear_layer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.first_conv_layer(x)
        x = self.first_dropout(self.first_batch_norm(self.relu_activation(x)))

        x = self.second_conv_layer(x)
        x = self.second_dropout(self.second_batch_norm(self.relu_activation(x)))
        x = self.flatten(x)

        x = self.third_dropout(self.relu_activation(self.first_linear_layer(x)))
        x = self.sigmoid_activation(self.second_linear_layer(x))

        return x

    def set_weights(self, weights: NDArrays):  # TODO: Specify type of weights and return type
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

    def get_weights(self) -> NDArrays:  # TODO: Specify return type
        return [val.cpu().numpy() for _, val in self.state_dict().items()]


def load_model(n_features=30, hidden_dim=32) -> Net:
    model = Net(n_features, hidden_dim)
    return model
