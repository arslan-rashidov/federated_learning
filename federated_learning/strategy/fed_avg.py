from functools import reduce
from typing import List, Tuple, overload

import numpy as np

from federated_learning.core.client import ClientManager, Client
from federated_learning.core.utils.typing import Weights, TrainInstructions, TrainResult, Metrics, EvaluateResult, EvaluateInstructions, \
    NDArrays, NDArray
from federated_learning.core.utils.weights_transformation import weights_to_ndarrays, ndarrays_to_weights
from federated_learning.strategy.weights_aggregation_strategy import WeightsAggregationStrategy


def calculate_weighted_metric(metrics: List[Tuple[int, float]], num_examples_total: int = None) -> float:
    if num_examples_total is None:
        num_examples_total = sum([num_examples for num_examples, _ in metrics])

    weighted_metrics = [num_examples * metric for num_examples, metric in metrics]
    return sum(weighted_metrics) / num_examples_total


class FedAvgStrategy(WeightsAggregationStrategy):
    def __init__(self, on_fit_config_fn, on_evaluate_config_fn):
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn

    def configure_train(self, server_round: int, weights: Weights, client_manager: ClientManager) -> List[Tuple[Client, TrainInstructions]]:
        config = {}
        if self.on_fit_config_fn:
            config = self.on_fit_config_fn(server_round)

        train_instructions = TrainInstructions(weights, config)

        clients = client_manager.get_num_clients(num_clients=len(client_manager))

        return [(client, train_instructions) for client in clients]

    def aggregate_train(self, results: List[Tuple[Client, TrainResult]]) -> Tuple[Weights, Metrics]:
        num_examples_total = sum([client_result[1].num_examples for client_result in results])

        weighted_weights = [
            [layer * client_result[1].num_examples for layer in weights_to_ndarrays(client_result[1].weights)] for client_result in
            results
        ]
        weights_list: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        weights_prime = ndarrays_to_weights(weights_list) # TODO: CHANGE DTYPE TO VALID

        metric_names: List[str] = list(results[0][1].metrics.keys())  # TODO: REFACTOR
        weighted_metrics = {}

        for metric_name in metric_names:
            metrics: List[Tuple[int, float]] = []
            for client_result in results:
                metrics.append((client_result[1].num_examples, client_result[1].metrics[metric_name]))
            weighted_metric = calculate_weighted_metric(metrics, num_examples_total)
            weighted_metrics[metric_name] = weighted_metric  # TODO: TILL HERE

        return weights_prime, weighted_metrics

    def configure_evaluate(self, server_round: int, weights: Weights, client_manager: ClientManager) -> List[Tuple[Client, EvaluateInstructions]]:
        config = {}
        config = {}
        if self.on_evaluate_config_fn:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateInstructions(weights, config)

        clients = client_manager.get_num_clients(num_clients=len(client_manager))

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, results: List[Tuple[Client, EvaluateResult]]) -> Metrics:
        metric_names: List[str] = list(results[0][1].metrics.keys())  # TODO: REFACTOR
        weighted_metrics = {}

        for metric_name in metric_names:
            metrics: List[Tuple[int, float]] = []
            for client_result in results:
                metrics.append((client_result[1].num_examples, client_result[1].metrics[metric_name]))
            weighted_metric = calculate_weighted_metric(metrics)
            weighted_metrics[metric_name] = weighted_metric  # TODO: TILL HERE

        return weighted_metrics
