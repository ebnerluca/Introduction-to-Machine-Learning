# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from numpy import genfromtxt

# Press the green button in the gutter to run the script.

def read_data(path):
    """Input: .csv filename string. Output: ndarray"""

    data = genfromtxt(path, delimiter=',')
    X = data[1:,1:]
    y = data[1:,0]
    return X,y


if __name__ == '__main__':

    datapath = "data/train.csv"
    X,y = read_data(datapath)

    # Debug
    print(np.shape(X))
    print(X)
    print(np.shape(y))
    print(y)

    #print (data_dict)

    training_model = RidgeCV(alphas=[0.1, 1, 10.0, 100.0, 200.0], cv=10).fit(X, y)
    training_model.score(X, y)



