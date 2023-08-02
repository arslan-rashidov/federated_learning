import os

from federated_learning.core.client.client_manager import ClientManager
from federated_learning.core.server.server import Server
from federated_learning.core.utils.typing import Config

from federated_learning.examples.mnist_fl.client.client import create_client
from federated_learning.strategy.fed_avg import FedAvgStrategy

COMMUNICATION_ROUNDS = 50


def on_fit_config_fn(server_round) -> Config:
    config = {
        'batch_size': 16,
        'epochs': 1
    }
    return config


def on_evaluate_config_fn(server_round) -> Config:
    config = {
        'batch_size': 16
    }
    return config


def main():
    client_manager = ClientManager()
    weights_path = "weights"
    for client_i in range(1, 11):
        client_folder_path = weights_path + f"/client_{client_i}"
        os.mkdir(client_folder_path)
        os.mkdir(client_folder_path + f"/global_weights")
        os.mkdir(client_folder_path + f"/local_weights")
        client = create_client(id=f'client_{client_i}', dataset_path=f"../client/mnist_dataset/client_{client_i}", global_weights_save_folder_path=client_folder_path + f"/global_weights", local_weights_save_folder_path=client_folder_path + f"/local_weights")
        client_manager.add_client(client)

    strategy = FedAvgStrategy(on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn)

    server = Server(client_manager, strategy)

    server.fit(COMMUNICATION_ROUNDS)


main()
