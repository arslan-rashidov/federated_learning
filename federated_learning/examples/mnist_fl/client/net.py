from collections import OrderedDict

import torch
from torch import nn

from federated_learning.core.utils.typing import NDArrays


class SimpleModel(nn.Module):
    def __init__(self, input_shape, classes):
        super().__init__()
        self.first_linear_layer = nn.Linear(input_shape, 200)
        self.third_linear_layer = nn.Linear(200, classes)

        self.relu_layer = nn.ReLU()

    def forward(self, x):
        x = self.first_linear_layer(x)
        x = self.relu_layer(x)
        x = self.third_linear_layer(x)
        return x

    def set_weights(self, weights: NDArrays):  # TODO: Specify type of weights and return type
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

    def get_weights(self) -> NDArrays:  # TODO: Specify return type
        return [val.cpu().numpy() for _, val in self.state_dict().items()]


def load_model(input_shape, classes):
    model = SimpleModel(input_shape=input_shape, classes=classes)
    return model
