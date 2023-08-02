import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    labels = df['Class'].to_numpy().reshape(-1, 1)
    transactions = df.drop(['Class'], axis=1).to_numpy()
    return transactions, labels

def preprocess_data(x_train, x_test):
    standard_scaler = StandardScaler()
    x_train = standard_scaler.fit_transform(x_train)
    x_test = standard_scaler.transform(x_test)
    return x_train, x_test

class TransactionsDataset(torch.utils.data.Dataset):
    def __init__(self, transactions, labels):
        self.transactions, self.labels = transactions, labels

    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, item):
        transaction = torch.Tensor(self.transactions[item])
        label = torch.FloatTensor(self.labels[item])
        return {'transaction': transaction, 'label': label}


def get_train_test_datasets(dataset_path):
    transactions, labels = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(transactions, labels, test_size=0.2, shuffle=True)
    x_train, x_test = preprocess_data(x_train, x_test)

    train_set = TransactionsDataset(x_train, y_train)
    test_set = TransactionsDataset(x_test, y_test)

    return (train_set, test_set)