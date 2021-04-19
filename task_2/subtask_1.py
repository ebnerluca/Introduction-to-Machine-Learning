from sklearn.model_selection import train_test_split
import numpy as np
# from numpy import genfromtxt
# import math
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()  # Number of input features is 5*35.

        n_inputs = 1*35
        n_layer1 = 40
        n_layer2 = 20
        n_outputs = 10

        self.layer_1 = nn.Linear(n_inputs, n_layer1)
        self.layer_2 = nn.Linear(n_layer1, n_layer2)
        self.layer_out = nn.Linear(n_layer2, n_outputs)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_layer1)
        self.batchnorm2 = nn.BatchNorm1d(n_layer2)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


# train data
class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


# test data
class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


"""def read_data(train_features_path, train_labels_path, test_features_path):
    print("Reading data...", end=" ", flush=True)

    train_features = np.genfromtxt(train_features_path, delimiter=",", skip_header=True)
    train_labels = np.genfromtxt(train_labels_path, delimiter=",", skip_header=True)
    test_features = np.genfromtxt(test_features_path, delimiter=",", skip_header=True)

    print("Done.")

    return train_features, train_labels, test_features"""

"""def save_preprocessed_data(preprocessed_train_features):
    print("Saving preprocessed data...", end=" ", flush=True)

    df = pd.DataFrame(preprocessed_train_features,
                      columns=["pid", "Time", "Age", "EtCO2", "PTT", "BUN", "Lactate", "Temp", "Hgb", "HCO3",
                               "BaseExcess", "RRate", "Fibrinogen", "Phosphate", "WBC", "Creatinine", "PaCO2",
                               "AST", "FiO2", "Platelets", "SaO2", "Glucose", "ABPm", "Magnesium", "Potassium",
                               "ABPd", "Calcium", "Alkalinephos", "SpO2", "Bilirubin_direct", "Chloride",
                               "Hct", "Heartrate", "Bilirubin_total", "TroponinI", "ABPs", "pH"])

    df.to_csv("data/preprocessed/preprocessed_test_features.csv", index=False,header=True)

    print("Done.")"""


if __name__ == '__main__':

    print("Reading data...", end=" ", flush=True)
    train_data = np.genfromtxt("data/preprocessed/train_features_preprocessed_task1.csv", delimiter=",",
                               skip_header=True)[:, 1:]
    train_labels = np.genfromtxt("data/train_labels.csv", delimiter=",", skip_header=True)[:,1:11]
    print("Done.")

    print(f"shape of train_data: {train_data.shape}")
    print(f"shape of train_labels: {train_labels.shape}")

    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    train_data = TrainData(torch.FloatTensor(train_data), torch.FloatTensor(train_labels))
    # test_data = TestData(torch.FloatTensor(X_test))

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BinaryClassification()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ## SUBTASK 1: Ordering of medical test

    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)

            y_pred = y_pred.reshape(-1,1)
            y_batch = y_batch.reshape(-1,1)

            #print(f"y_pred: {y_pred}")
            #print(f"y_batch: {y_batch}")
            loss = criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} 'f'| Acc: {epoch_acc / len(train_loader):.3f}')

    ## SUBTASK 2: Sepsis prediction

    ## SUBTASK 3: Key vital signs prediction

    # np.savetxt("results.csv", weights, delimiter="\n")
