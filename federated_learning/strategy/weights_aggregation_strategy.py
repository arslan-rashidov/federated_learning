from abc import ABC, abstractmethod
from typing import List, Tuple

from federated_learning.core.client import Client
from federated_learning.core.client.client_manager import ClientManager
from federated_learning.core.utils.typing import TrainResult, Weights, EvaluateResult, TrainInstructions, Metrics, EvaluateInstructions


class WeightsAggregationStrategy(ABC):
    @abstractmethod
    def configure_train(self, server_round: int, weights: Weights, client_manager: ClientManager) -> List[Tuple[Client, TrainInstructions]]:
        pass

    @abstractmethod
    def aggregate_train(self, results: List[Tuple[Client, TrainResult]]) -> Tuple[Weights, Metrics]:
        pass

    @abstractmethod
    def configure_evaluate(self, server_round: int, weights: Weights, client_manager: ClientManager) -> List[Tuple[Client, EvaluateInstructions]]:
        pass

    @abstractmethod
    def aggregate_evaluate(self, results: List[Tuple[Client, EvaluateResult]]) -> Metrics:
        pass
