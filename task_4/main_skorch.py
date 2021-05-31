# import numpy as np
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules.pooling import AvgPool1d
import torch.optim as optim
import torchvision.models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from skorch import NeuralNetClassifier, dataset
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer

# import matplotlib.pyplot as plt


#### Config ####

# preprocessing
n_images = 10000
img_size = (224, 224)
image_loader_batch_size = 32
encoder_features = 1000  # dependant on output of classifier
compute_features = False  # features don't need to be recomputed at each run
training_mode = True     # if true, output file is not generated
features_path = "data/features.txt"

# prediction
# train_mode = True
learning_rate = 0.007
epochs = 8
# batch_size = 128

# output
test_labels_path = "data/test_labels.txt"

################


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()  # Number of input features is 4*5+1.

        n_inputs = 1 * encoder_features  # 1000
        n_layer1 = 500

        self.feature0 = nn.Sequential(
            nn.Linear(n_inputs, n_layer1),
            nn.ReLU(),
            nn.BatchNorm1d(n_layer1),
        )
        self.feature1 = nn.Sequential(
            nn.Linear(n_inputs, n_layer1),
            nn.ReLU(),
            nn.BatchNorm1d(n_layer1),
        )
        self.feature2 = nn.Sequential(
            nn.Linear(n_inputs, n_layer1),
            nn.ReLU(),
            nn.BatchNorm1d(n_layer1),
        )

        merge_n1 = 3*n_layer1
        merge_n2 = 500
        merge_n3 = 250
        merge_n4 = 50

        self.merge_l1 = nn.Linear(merge_n1,merge_n2)
        self.merge_l2 = nn.Linear(merge_n2,merge_n3)
        self.merge_l3 = nn.Linear(merge_n3,merge_n4)
        self.merge_l4 = nn.Linear(merge_n4,1)
        self.relu = nn.ReLU()
        self.batchnorm_merge_l1 = nn.BatchNorm1d(merge_n2)
        self.batchnorm_merge_l2 = nn.BatchNorm1d(merge_n3)
        self.batchnorm_merge_l3 = nn.BatchNorm1d(merge_n4)
        self.batchnorm_merge_l4 = nn.BatchNorm1d(1)
        
        self.dropout_02 = nn.Dropout(p=0.2)
        self.dropout_05 = nn.Dropout(p=0.5)     

    
    def forward(self, inputs):
        #print(f"input size: {inputs.shape}")
        input_split = torch.split(inputs, 1000, dim=1)
        input_split = list(input_split)
        #print(f"input split size: {input_split[0].shape}") 

        x0 = self.feature0(input_split[0])
        x1 = self.feature1(input_split[1])
        x2 = self.feature2(input_split[2])

        x = torch.cat((x0,x1,x2), dim=1)

        x = self.relu(self.merge_l1(x))
        x = self.batchnorm_merge_l1(x)
        x = self.dropout_02(x)
        x = self.relu(self.merge_l2(x))
        x = self.batchnorm_merge_l2(x)
        x = self.dropout_05(x)
        x = self.relu(self.merge_l3(x))
        x = self.batchnorm_merge_l3(x)
        x = self.dropout_05(x)
        x = self.relu(self.merge_l4(x))
        x = self.batchnorm_merge_l4(x)

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
                                    # transforms.RandomHorizontalFlip(p=0.5),
                                    # transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
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


def binary_acc(labels, predictions):
    # map predictions to binary 0 or 1
    # predictions = np.round(torch.sigmoid(predictions))
    predictions = np.round_(predictions)
    correct_results_sum = (predictions == labels).sum()
    binary_acc = correct_results_sum / labels.shape[0]
    binary_acc = np.round_(binary_acc * 100)

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
    train_triplets_switched = np.loadtxt("data/train_triplets.txt", dtype=int)
    train_labels = np.ones(train_triplets.shape[0]).reshape((train_triplets.shape[0], 1))
    train_labels_switched = np.zeros(train_triplets_switched.shape[0]).reshape((train_triplets_switched.shape[0], 1))

    for i in range(len(train_triplets_switched)):
        train_triplets_switched[i] = np.asarray([train_triplets_switched[i, 0],
                                                 train_triplets_switched[i, 2],
                                                 train_triplets_switched[i, 1]])

    '''train_triplets = np.vstack((train_triplets, train_triplets_switched))
    train_labels = np.vstack((train_labels, train_labels_switched))'''

    print(f"train_triplets shape: {train_triplets.shape}")
    print(f"train_labels shape: {train_labels.shape}")

    test_triplets = np.loadtxt("data/test_triplets.txt", dtype=int)
    print(f"test_triplets shape: {test_triplets.shape}")

    # Random shuffling second and third entry of train_triplets (--> ABC or ACB)
    # otherwise output label would always be 1
    for i in range(len(train_triplets)):
        shuffle = bool(np.random.randint(0, 2))  # random True or False
        if shuffle:
            train_triplets[i] = np.hstack((train_triplets[i, 0], train_triplets[i, 2], train_triplets[i, 1]))
            train_labels[i] = 0

    train_triplets_features = np.zeros((train_triplets.shape[0], train_triplets.shape[1] * encoder_features))
    for i in range(train_triplets.shape[0]):
        train_triplets_features[i] = np.hstack((features[train_triplets[i, 0]],
                                                features[train_triplets[i, 1]],
                                                features[train_triplets[i, 2]]))
    print(f"train_triplets_features shape: {train_triplets_features.shape}")
    print(f"train_labels shape: {train_labels.shape}")
    train_triplets_features = np.float32(train_triplets_features)
    train_labels = np.float32(train_labels)
    #train_labels = np.expand_dims(train_labels, axis=1)

    test_triplets_features = np.zeros((test_triplets.shape[0], test_triplets.shape[1] * encoder_features))
    for i in range(test_triplets.shape[0]):
        test_triplets_features[i] = np.hstack((features[test_triplets[i, 0]],
                                               features[test_triplets[i, 1]],
                                               features[test_triplets[i, 2]]))
    print(f"test_triplets_features shape: {test_triplets_features.shape}")
    test_triplets_features = np.float32(test_triplets_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # skorch
    classifier = NeuralNetClassifier(
        BinaryClassification,
        train_split=dataset.CVSplit(5, stratified=False),
        criterion=nn.BCEWithLogitsLoss,
        optimizer=optim.Adam, 
        max_epochs=epochs,
        lr=learning_rate,
        optimizer__weight_decay=0.01,
        device=device
    )

    ### calculate cross validation score for training mode
    custom_scorer = make_scorer(binary_acc, greater_is_better=True)
    if training_mode:
        scores = cross_val_score(classifier, train_triplets_features, train_labels, cv=5, scoring=custom_scorer, verbose=True)
        print("Cross-Validation score {score:.3f},"
              " Standard Deviation {err:.3f}"
              .format(score = scores.mean(), err = scores.std()))

    ### generate output if not
    else:
        classifier = classifier.fit(train_triplets_features, train_labels)
        predictions = classifier.predict_proba(test_triplets_features)[:,1]
        predictions = np.around(predictions)
        np.savetxt(test_labels_path, predictions.astype(int), fmt="%i")


