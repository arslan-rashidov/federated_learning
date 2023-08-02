from abc import ABC, abstractmethod
from typing import Any
from collections import OrderedDict

import torch
from torch import nn

from federated_learning.core.utils import weights_to_ndarrays
from federated_learning.core.utils.typing import TrainInstructions, TrainResult, EvaluateInstructions, EvaluateResult, GetWeightsInstructions, \
    GetWeightsResult, Weights


class Client(ABC):
    def __init__(self, id: str, model: Any, global_weights_save_folder_path: str = None, local_weights_save_folder_path: str = None):
        self.id = id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert issubclass(model.__class__, nn.Module)
        self.model = model
        self.model.to(self.device)

        self.save_global_weights = False
        if global_weights_save_folder_path:
            self.save_global_weights = True
            self.global_weights_save_folder_path = global_weights_save_folder_path

        self.save_local_weights = False
        if local_weights_save_folder_path:
            self.save_local_weights = True
            self.local_weights_save_folder_path = local_weights_save_folder_path

    def get_weights_execute(self, get_weights_ins: GetWeightsInstructions) -> GetWeightsResult:
        get_weights_result = self.get_weights(get_weights_ins=get_weights_ins)
        assert type(get_weights_result) == GetWeightsResult

        return get_weights_result

    def train_execute(self, train_ins: TrainInstructions) -> TrainResult:
        train_result = self.train(train_ins=train_ins)
        assert train_result.weights is not None

        if self.save_local_weights:
            self.save_weights(train_result.weights, self.local_weights_save_folder_path, server_round=train_ins.config['server_round'])

        return train_result

    def evaluate_execute(self, eval_ins: EvaluateInstructions) -> EvaluateResult:
        if self.save_global_weights:
            self.save_weights(eval_ins.weights, self.global_weights_save_folder_path, server_round=eval_ins.config['server_round'], is_global=True)

        evaluate_result = self.evaluate(eval_ins=eval_ins)

        return evaluate_result

    @abstractmethod
    def get_weights(self, get_weights_ins: GetWeightsInstructions) -> GetWeightsResult:
        pass

    @abstractmethod
    def train(self, train_ins: TrainInstructions) -> TrainResult:
        pass

    @abstractmethod
    def evaluate(self, eval_ins: EvaluateInstructions) -> EvaluateResult:
        pass

    def save_weights(self, weights: Weights, path: str, server_round: int, is_global: bool = False):
        weights = weights_to_ndarrays(weights)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        save_path = path + f"/{'global' if is_global else 'local'}_weights_{server_round}.pt"
        torch.save(state_dict, save_path)

