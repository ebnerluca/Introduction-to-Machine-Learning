import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor, dataset

class RegressionNetwork(nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()  # Number of input features is 4*5+1.

        n_inputs = 35*5
        n_layer1 = 128
        n_layer2 = 128
        n_layer3 = 128
        n_layer4 = 128
        #n_layer5 = 32
        n_outputs = 1

        self.layer_1 = nn.Linear(n_inputs, n_layer1)
        self.layer_2 = nn.Linear(n_layer1, n_layer2)
        self.layer_3 = nn.Linear(n_layer2, n_layer3)
        self.layer_4 = nn.Linear(n_layer3, n_layer4)
        #self.layer_5 = nn.Linear(n_layer4, n_layer5)

        self.layer_out = nn.Linear(n_layer4, n_outputs)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
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
    labels = np.genfromtxt("data/train_labels.csv", delimiter=",", skip_header=True)[:,12:16]
    test = np.genfromtxt("data/preprocessed/test_features_preprocessed_new.csv", delimiter=",",
                                skip_header=True)
    print("Done.")

    training_mode = False

    ### trying neural net approach
    data = np.float32(data)
    test = np.float32(test)
    labels = np.float32(labels)
    mean_score = 0

    test_sample_size = test.shape[0]
    output_array = np.empty((test_sample_size,0), float)

    for i in range(4):

        # NN approach
        '''label = np.expand_dims(labels[:,i], axis=1)
        regressor = NeuralNetRegressor(
            RegressionNetwork,
            criterion=nn.MSELoss,
            optimizer=optim.Adam,
            train_split=dataset.CVSplit(5, stratified=False), 
            max_epochs=5,
            lr=0.015,
        )'''
        # Histo approach
        label = labels[:,i]
        regressor = HistGradientBoostingRegressor(max_depth=3)

        ### calculate cross validation score for training mode
        if training_mode:

            scores = cross_val_score(regressor, data, label, cv=5, scoring='r2', verbose=True)
            hand_in_metric = 0.5 + 0.5 * np.maximum(0, scores.mean())
            print("Cross-Validation score {score:.3f},"
                  .format(score = hand_in_metric))

            mean_score += hand_in_metric

        ### generate output if not
        else:
            regressor = regressor.fit(data, label)
            predictions = regressor.predict(test)
            training_score = 0.5 + 0.5 * np.maximum(0, metrics.r2_score(label, regressor.predict(data)))
            print("Training score:", training_score)
            #print(predictions.reshape(-1,1).shape)
            #print(output_array.shape)
            output_array = np.hstack((output_array, predictions.reshape((-1,1)) ))

    if training_mode:
        print(f"Hand in metric score is: {mean_score/4}")
    else:
        output_path = "data/output/subtask_3_new_labels.csv"
        pd.DataFrame(output_array).to_csv(output_path,
                                          header=["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"],
                                          index=None, float_format="%.3f")

        print(f"Predicted labels saved to {output_path}.")
