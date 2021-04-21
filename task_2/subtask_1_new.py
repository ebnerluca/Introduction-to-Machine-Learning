import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier, dataset

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()  # Number of input features is 5*35.

        n_inputs = 35*5
        n_layer1 = 128
        n_layer2 = 128
        n_layer3 = 64
        n_layer4 = 64
        n_layer5 = 32
        n_outputs = 1

        self.layer_1 = nn.Linear(n_inputs, n_layer1)
        self.layer_2 = nn.Linear(n_layer1, n_layer2)
        self.layer_3 = nn.Linear(n_layer2, n_layer3)
        self.layer_4 = nn.Linear(n_layer3, n_layer4)
        self.layer_5 = nn.Linear(n_layer4, n_layer5)

        self.layer_out = nn.Linear(n_layer5, n_outputs)

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
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

if __name__ == '__main__':

    ### read the data
    print("Reading data...", end=" ", flush=True)
    data = np.genfromtxt("data/preprocessed/train_features_preprocessed_new.csv", delimiter=",",
                               skip_header=True)
    labels = np.genfromtxt("data/train_labels.csv", delimiter=",", skip_header=True)[:,0:11]
    test = np.float32(np.genfromtxt("data/preprocessed/test_features_preprocessed_new.csv", delimiter=",",
                                skip_header=True))
    pids = test[:,0]
    labels = labels[:,1:]
    print("Done.")

    training_mode = False

    ### trying neural net approach
    data = np.float32(data)
    labels = np.float32(labels)
    mean_score = 0

    output_array = pids.reshape((pids.shape[0], 1))

    for i in range(10):

        label = np.expand_dims(labels[:,i], axis=1)

        classifier = NeuralNetClassifier(
            BinaryClassification,
            train_split=dataset.CVSplit(5, stratified=False),
            criterion=nn.BCEWithLogitsLoss,
            optimizer=optim.Adam, 
            max_epochs=5,
            lr=0.002,
        )

        ### calculate cross validation score for training mode
        if training_mode:

            scores = cross_val_score(classifier, data, label, cv=5, scoring='roc_auc', verbose=True)
            print("Cross-Validation score {score:.3f},"
                  " Standard Deviation {err:.3f}"
                  .format(score = scores.mean(), err = scores.std()))
            mean_score += scores.mean()

        ### generate output if not
        else:
            classifier = classifier.fit(data, label)
            predictions = classifier.predict_proba(test)[:, 1]
            print("Training score:", metrics.roc_auc_score(label, classifier.predict_proba(data)[:, 1]))
            output_array = np.hstack((output_array, predictions))

    if training_mode:
        print(f"Mean Cross-Validation ROC-AUC score is: {mean_score/10}")
    else:
        output_path = "data/output/subtask_1_new_labels.csv"
        pd.DataFrame(output_array).to_csv(output_path,
                                        header=["pid", "LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos",
                                                "LABEL_Bilirubin_total","LABEL_Lactate","LABEL_TroponinI","LABEL_SaO2",
                                                "LABEL_Bilirubin_direct","LABEL_EtCO2"], index=None, float_format="%.3f")

        print(f"Predicted labels saved to {output_path}.")