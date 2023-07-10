import random
from typing import Dict, List, Any

import mlflow

from federated_learning.core.client.client import Client


class ClientManager:
    def __init__(self):
        self.clients: Dict[str, Client] = {}

        self.mlflow_client = None
        self.clients_runs = None

    def __len__(self):
        return len(self.clients)

    def add_client(self, client: Client) -> bool:
        if client.id in self.clients.keys():
            return False

        self.clients[client.id] = client
        return True

    def remove_client(self, client_id: str) -> bool:
        if client_id not in self.clients.keys():
            return False

        del self.clients[client_id]
        return True

    def get_num_clients(self, num_clients: int) -> List[Client]:
        if num_clients > len(self):
            return list(self.clients.values())

        random_ids = random.sample(list(self.clients.keys()), num_clients)

        return [self.clients[random_id] for random_id in random_ids]

    def start_mlflow_tracking(self):
        self.mlflow_client = mlflow.tracking.MlflowClient()
        experiment_name = f"model_{len(self.mlflow_client.search_experiments())}"
        experiment_id = self.mlflow_client.create_experiment(experiment_name)
        self.clients_runs = {client: self.mlflow_client.create_run(experiment_id, run_name=client) for client in self.clients}

    def log_client_metric(self, client_id: str, key: str, value: float):
        self.mlflow_client.log_metric(run_id=self.clients_runs[client_id].info.run_id, key=key, value=value)

    def log_client_param(self, client_id: str, key: str, value: Any):
        self.mlflow_client.log_param(run_id=self.clients_runs[client_id].info.run_id, key=key, value=value)
