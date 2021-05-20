# import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt


#### Config ####

# preprocessing
n_images = 10000
img_size = (224, 224)
image_loader_batch_size = 32
encoder_features = 1000  # dependant on output of classifier
compute_features = False  # features don't need to be recomputed at each run
features_path = "data/features.txt"

# prediction
train_mode = True
learning_rate = 0.015
epochs = 30
batch_size = 64

################


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()  # Number of input features is 4*5+1.

        n_inputs = 3 * encoder_features  # 3000
        n_layer1 = 2 * encoder_features
        n_layer2 = int(encoder_features / 1)
        n_layer3 = int(encoder_features / 2)
        n_layer4 = int(encoder_features / 8)
        n_layer5 = 1
        n_outputs = 1

        self.layer_1 = nn.Linear(n_inputs, n_layer1)
        self.layer_2 = nn.Linear(n_layer1, n_layer2)
        self.layer_3 = nn.Linear(n_layer2, n_layer3)
        self.layer_4 = nn.Linear(n_layer3, n_layer4)
        self.layer_5 = nn.Linear(n_layer4, n_layer5)

        # self.layer_out = nn.Linear(n_layer4, n_outputs)
        self.layer_out = nn.Sigmoid()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_layer1)
        self.batchnorm2 = nn.BatchNorm1d(n_layer2)
        self.batchnorm3 = nn.BatchNorm1d(n_layer3)
        self.batchnorm4 = nn.BatchNorm1d(n_layer4)
        self.batchnorm5 = nn.BatchNorm1d(n_layer5)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.layer_4(x))
        x = self.batchnorm4(x)
        x = self.relu(self.layer_5(x))
        x = self.batchnorm5(x)
        # x = self.dropout(x)
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


def preprocessing():
    """Computes features from images by using a pretrained classifier."""

    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
    images = datasets.ImageFolder("data", transform=transform)
    image_loader = DataLoader(images, batch_size=image_loader_batch_size, shuffle=False)

    # Model for feature predictions
    classifier_model = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)
    classifier_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier_model.to(device)
    print(f"Device: {device}")

    # Compute features by using pretrained model
    features = np.zeros((0, encoder_features))
    iter = 0
    for image_batch in image_loader:
        print(f"Computing features... {iter} / {n_images}", end="\r")  # , flush=True)
        image_batch = image_batch[0]  # get rid of target classes
        image_batch = image_batch.to(device)
        features_batch = classifier_model(image_batch)
        features = np.vstack((features, features_batch.cpu().detach().numpy()))
        iter = iter + image_loader_batch_size
    print(f"\nComputing features done. Saving features under {str(features_path)}")
    np.savetxt(features_path, features)

    return features


def binary_acc(predictions, labels):
    # map predictions to binary 0 or 1
    # predictions = torch.round(torch.sigmoid(predictions))
    predictions = torch.round(predictions)
    correct_results_sum = (predictions == labels).sum().float()
    binary_acc = correct_results_sum / labels.shape[0]
    binary_acc = torch.round(binary_acc * 100)

    return binary_acc


if __name__ == '__main__':

    if compute_features:
        features = preprocessing()
    else:
        print("Loading features...")
        features = np.loadtxt(features_path)
        print("Loading features done.")

    # Read Data
    train_triplets = np.loadtxt("data/train_triplets.txt", dtype=int)
    train_labels = np.ones(train_triplets.shape[0])
    print(f"train_triplets shape: {train_triplets.shape}")

    test_triplets = np.loadtxt("data/test_triplets.txt",dtype=int)
    print(f"test_triplets shape: {test_triplets.shape}")

    # Random shuffling second and third entry of train_triplets (--> ABC or ACB)
    # otherwise output label would always be 1
    for i in range(len(train_triplets)):
        shuffle = bool(np.random.randint(0, 2))  # random True or False
        if shuffle:
            train_triplets[i] = np.hstack((train_triplets[i, 0], train_triplets[i, 2], train_triplets[i, 1]))
            train_labels[i] = 0
    # split train_triplets in train_test_triplets and train_triplets

    train_triplets_features = np.zeros((train_triplets.shape[0], train_triplets.shape[1]*encoder_features))
    for i in range(train_triplets.shape[0]):
        train_triplets_features[i] = np.hstack((features[train_triplets[i, 0]],
                                                features[train_triplets[i, 1]],
                                                features[train_triplets[i, 2]]))
    print(f"train_triplets_features shape: {train_triplets_features.shape}")

    # Data Loader
    train_data = TrainData(torch.FloatTensor(train_triplets_features), torch.FloatTensor(train_labels))
    split = torch.utils.data.random_split(train_data, [int(0.8 * len(train_data)), len(train_data) - int(0.8 * len(train_data))])
    train_data = split[0]
    train_test_data = split[1]
    # test_data = TestData(torch.FloatTensor(test_triplets))
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    train_test_data_loader = DataLoader(train_test_data, batch_size=batch_size, shuffle=False)
    # test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BinaryClassification()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_predictions = np.zeros((len(train_triplets), 1))
    model.train()
    for e in range(1, epochs + 1):
        epoch_loss = 0
        epoch_acc = 0
        binary_acc_train = 0

        # training
        model.train()
        for X_batch, y_batch in train_data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)
            # print(f"y_pred: {y_pred}")

            y_pred = y_pred.reshape(-1, 1)
            y_batch = y_batch.reshape(-1, 1)

            loss = criterion(y_pred, y_batch)
            acc = abs(y_pred - y_batch).mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

            binary_acc_train += binary_acc(y_pred, y_batch)

        """print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_data_loader):.5f} | '
              f'Acc: {epoch_acc / len(train_data_loader):.3f} | '
              f'Binary Acc: {binary_acc_train / len(train_data_loader):.3f}%')"""

        # eval
        model.eval()
        binary_acc_train_test = 0

        for X_batch_test, y_batch_test in train_test_data_loader:
            X_batch_test, y_batch_test = X_batch_test.to(device), y_batch_test.to(device)

            y_pred_test = model(X_batch_test)

            y_pred_test = y_pred_test.reshape(-1, 1)
            y_batch_test = y_batch_test.reshape(-1, 1)

            binary_acc_train_test += binary_acc(y_pred_test, y_batch_test)

        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_data_loader):.5f} | '
              f'Acc: {epoch_acc / len(train_data_loader):.3f} | '
              f'Binary Acc: {binary_acc_train / len(train_data_loader):.2f}% | '
              f'Binary Acc (Train Test): {binary_acc_train_test / len(train_test_data_loader):.2f}%')

    print("All finished")






