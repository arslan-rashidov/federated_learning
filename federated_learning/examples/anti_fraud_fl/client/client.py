import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from federated_learning.core.client import Client
from federated_learning.core.utils.typing import GetWeightsInstructions, GetWeightsResult, TrainInstructions, \
    TrainResult, \
    EvaluateInstructions, EvaluateResult, Code, Status
from federated_learning.core.utils.weights_transformation import ndarrays_to_weights, weights_to_ndarrays
from federated_learning.examples.anti_fraud_fl.client.dataset import TransactionsDataset, get_train_test_datasets
from federated_learning.examples.anti_fraud_fl.client.net import load_model


class AntiFraudClient(Client):
    def __init__(self, id, model, global_weights_save_folder_path, local_weights_save_folder_path, train_set, test_set):
        super().__init__(id, model, global_weights_save_folder_path, local_weights_save_folder_path)

        self.train_set = train_set
        self.test_set = test_set

    def get_weights(self, get_weights_ins: GetWeightsInstructions) -> GetWeightsResult:
        weights = ndarrays_to_weights(self.model.get_weights())
        return GetWeightsResult(status=Status(code=Code.OK, message='OK'), weights=weights)

    def train(self, train_ins: TrainInstructions) -> TrainResult:
        if train_ins.weights:
            weights = weights_to_ndarrays(train_ins.weights)
            self.model.set_weights(weights)
        train_epoch_loss = 0.0
        config = train_ins.config
        batch_size = config['batch_size']
        train_dataloader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(params=self.model.parameters(), lr=0.0001)
        loss_fn = torch.nn.BCELoss()
        self.model.train()

        for i, data in enumerate(train_dataloader):
            transactions, labels = data['transaction'], data['label']
            transactions = transactions.to(self.device)
            labels = labels.to(self.device)
            transactions = transactions.reshape(transactions.shape[0], 1, transactions.shape[1])

            optimizer.zero_grad()

            output = self.model(transactions)

            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        train_epoch_loss /= len(train_dataloader)

        weights = ndarrays_to_weights(self.model.get_weights())
        return TrainResult(status=Status(code=Code.OK, message='OK'), weights=weights,
                           metrics={'train_loss': train_epoch_loss}, num_examples=len(train_dataloader.dataset))

    def evaluate(self, eval_ins: EvaluateInstructions) -> EvaluateResult:
        if eval_ins.weights:
            weights = weights_to_ndarrays(eval_ins.weights)
            self.model.set_weights(weights)

        test_loss = 0.0
        self.model.eval()
        loss_fn = torch.nn.BCELoss()

        test_dataloader = DataLoader(self.test_set)

        outputs = np.array([])
        labels = np.array([])

        for i, data in enumerate(test_dataloader):
            transactions, label = data['transaction'], data['label']
            transactions = transactions.to(self.device)
            label = label.to(self.device)
            transactions = transactions.reshape(transactions.shape[0], 1, transactions.shape[1])
            output = self.model(transactions)

            loss = loss_fn(output, label)

            test_loss += loss.item()

            outputs = np.hstack([outputs, output.cpu().detach().numpy().reshape(-1)])
            labels = np.hstack([labels, label.cpu().reshape(-1)])

        test_loss /= len(test_dataloader)
        test_roc_auc_score = roc_auc_score(labels, outputs)

        return EvaluateResult(status=Status(code=Code.OK, message='OK'), num_examples=len(test_dataloader.dataset),
                              metrics={'test_loss': test_loss, 'test_roc_auc_score': test_roc_auc_score})


def create_client(id, dataset_path, global_weights_save_folder_path: str, local_weights_save_folder_path: str,
                  init_weights=None) -> AntiFraudClient:
    train_set, test_set = get_train_test_datasets(dataset_path=dataset_path)

    model = load_model(n_features=30, hidden_dim=32)

    client = AntiFraudClient(id=id, model=model, global_weights_save_folder_path=global_weights_save_folder_path,
                             local_weights_save_folder_path=local_weights_save_folder_path, train_set=train_set,
                             test_set=test_set)

    if init_weights:
        client.model.set_weights(init_weights)

    return client
