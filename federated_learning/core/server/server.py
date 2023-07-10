import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List

from federated_learning.core.client import Client, ClientManager
from federated_learning.core.utils.typing import Weights, GetWeightsInstructions, GetWeightsResult, EvaluateInstructions, EvaluateResult, \
    Metrics, TrainInstructions, TrainResult
from federated_learning.strategy.weights_aggregation_strategy import WeightsAggregationStrategy


def evaluate_client(client: Client, eval_ins: EvaluateInstructions) -> Tuple[Client, EvaluateResult]:
    eval_result: EvaluateResult = client.evaluate(eval_ins=eval_ins)
    return client, eval_result


def evaluate_clients(clients_instructions: List[Tuple[Client, EvaluateInstructions]], max_workers: int) -> List[
    Tuple[Client, EvaluateResult]]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_client, client, eval_ins) for client, eval_ins in clients_instructions]
        completed_futures, uncompleted_futures = concurrent.futures.wait(futures)

    eval_results: List[Tuple[Client, EvaluateResult]] = []

    for completed_future in completed_futures:
        eval_results.append(completed_future.result())

    return eval_results


def train_client(client: Client, train_ins: TrainInstructions) -> Tuple[Client, TrainResult]:
    train_result: TrainResult = client.train(train_ins=train_ins)
    return client, train_result


def train_clients(clients_instructions: List[Tuple[Client, TrainInstructions]], max_workers: int) -> List[
    Tuple[Client, TrainResult]]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_client, client, train_ins) for client, train_ins in clients_instructions]
        completed_futures, uncompleted_futures = concurrent.futures.wait(futures)

    train_results: List[Tuple[Client, TrainResult]] = []

    for completed_future in completed_futures:
        train_results.append(completed_future.result())

    return train_results


class Server:
    def __init__(self, client_manager: ClientManager, strategy: WeightsAggregationStrategy):
        self.client_manager: ClientManager = client_manager
        self.strategy: WeightsAggregationStrategy = strategy

        self.weights: Weights = Weights(values=[], dtype='')

        self.client_manager.start_mlflow_tracking()

    def get_initial_weights(self):
        client: Client = self.client_manager.get_num_clients(1)[0]
        get_weights_ins: GetWeightsInstructions = GetWeightsInstructions(config={})
        get_weights_result: GetWeightsResult = client.get_weights(get_weights_ins=get_weights_ins)
        initial_weights: Weights = get_weights_result.weights
        return initial_weights

    def fit(self, num_rounds: int):
        self.weights = self.get_initial_weights()

        aggregated_evaluate_results, evaluate_results = self.evaluate_round(
            server_round=0)

        for result in evaluate_results:
            client_id = result[0].id
            metrics = result[1].metrics
            for metric in metrics:
                self.client_manager.log_client_metric(client_id, metric, metrics[metric])

        for current_round in range(1, num_rounds + 1):
            aggregated_weights, aggregated_metrics, train_results = self.train_round(
                server_round=current_round)

            self.weights = aggregated_weights

            aggregated_metrics, evaluate_results = self.evaluate_round(
                server_round=current_round)

            for result in evaluate_results:
                client_id = result[0].id
                metrics = result[1].metrics
                for metric in metrics:
                    self.client_manager.log_client_metric(client_id, metric, metrics[metric])

        return self.weights

    def train_round(self, server_round: int) -> Tuple[Weights, Metrics, List[Tuple[Client, TrainResult]]]:
        clients_train_instructions: List[Tuple[Client, TrainInstructions]] = self.strategy.configure_train(
            server_round=server_round, weights=self.weights, client_manager=self.client_manager)
        train_results: List[Tuple[Client, TrainResult]] = train_clients(clients_instructions=clients_train_instructions,
                                                                        max_workers=5)
        aggregated_weights, aggregated_metrics = self.strategy.aggregate_train(results=train_results)

        return aggregated_weights, aggregated_metrics, train_results

    def evaluate_round(self, server_round: int) -> Tuple[Metrics, List[Tuple[Client, EvaluateResult]]]:
        clients_evaluate_instructions: List[Tuple[Client, EvaluateInstructions]] = self.strategy.configure_evaluate(
            server_round=server_round, weights=self.weights, client_manager=self.client_manager)
        evaluate_results: List[Tuple[Client, EvaluateResult]] = evaluate_clients(
            clients_instructions=clients_evaluate_instructions, max_workers=5)
        aggregated_metrics: Metrics = self.strategy.aggregate_evaluate(results=evaluate_results)

        return aggregated_metrics, evaluate_results
