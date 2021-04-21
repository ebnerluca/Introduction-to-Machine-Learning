import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()  # Number of input features is 4*5+1.

        n_inputs = 35*5
        n_layer1 = 128
        n_layer2 = 128
        n_layer3 = 128
        n_layer4 = 128
        #n_layer5 = 32
        n_outputs = 4

        self.layer_1 = nn.Linear(n_inputs, n_layer1)
        self.layer_2 = nn.Linear(n_layer1, n_layer2)
        self.layer_3 = nn.Linear(n_layer2, n_layer3)
        self.layer_4 = nn.Linear(n_layer3, n_layer4)
        #self.layer_5 = nn.Linear(n_layer4, n_layer5)

        self.layer_out = nn.Linear(n_layer4, n_outputs)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_layer1)
        self.batchnorm2 = nn.BatchNorm1d(n_layer2)
        self.batchnorm3 = nn.BatchNorm1d(n_layer3)
        self.batchnorm4 = nn.BatchNorm1d(n_layer4)
        #self.batchnorm5 = nn.BatchNorm1d(n_layer5)


    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.layer_4(x))
        x = self.batchnorm4(x)
        #x = self.relu(self.layer_5(x))
        #x = self.batchnorm5(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

if __name__ == '__main__':

    ### read the data
    print("Reading data...", end=" ", flush=True)
    data = np.genfromtxt("data/preprocessed/train_features_preprocessed_new.csv", delimiter=",",
                               skip_header=True)
    labels = np.genfromtxt("data/train_labels.csv", delimiter=",", skip_header=True)[:,11]
    test = np.genfromtxt("data/preprocessed/test_features_preprocessed_new.csv", delimiter=",",
                                skip_header=True)
    print("Done.")

    training_mode = True

    ### cross validation of the data
    '''classifier = HistGradientBoostingClassifier()

    scores = cross_val_score(classifier, data, labels, cv=5, scoring='roc_auc', verbose=True)
    print("Cross-Validation score {score:.3f},"
          " Standard Deviation {err:.3f}"
          .format(score = scores.mean(), err = scores.std()))'''

    ### trying neural net approach
    classifier = NeuralNetClassifier(
        BinaryClassification,
        criterion=nn.BCEWithLogitsLoss,
        optimizer=optim.Adam, 
        max_epochs=5,
        lr=0.0015,
    )
    scores = cross_val_score(classifier, np.float32(data), np.expand_dims(np.float32(labels), axis=1),
                                 cv=5, scoring='roc_auc', verbose=True)
    print("Cross-Validation score {score:.3f},"
          " Standard Deviation {err:.3f}"
          .format(score = scores.mean(), err = scores.std()))

    ### fit with the training set and generate test set output
    if training_mode == False:

        classifier = classifier.fit(data, labels)
        predictions = classifier.predict_proba(test)[:, 1]
        print("Training score:", metrics.roc_auc_score(labels, classifier.predict_proba(data)[:, 1]))

        output_array = predictions
        output_path = "data/output/subtask_2_new_labels.csv"
        pd.DataFrame(output_array).to_csv(output_path,
                                            header=["LABEL_Sepsis"], index=None, float_format="%.3f")

        print(f"Predicted labels saved to {output_path}.")