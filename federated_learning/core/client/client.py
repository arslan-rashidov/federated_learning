from abc import ABC, abstractmethod

from federated_learning.core.utils.typing import TrainInstructions, TrainResult, EvaluateInstructions, EvaluateResult, GetWeightsInstructions, \
    GetWeightsResult


class Client(ABC):
    def __init__(self, id: str):
        self.id = id

    @abstractmethod
    def get_weights(self, get_weights_ins: GetWeightsInstructions) -> GetWeightsResult:
        pass

    @abstractmethod
    def train(self, train_ins: TrainInstructions) -> TrainResult:
        pass

    @abstractmethod
    def evaluate(self, eval_ins: EvaluateInstructions) -> EvaluateResult:
        pass
