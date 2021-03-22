# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from numpy import genfromtxt
import math

# Press the green button in the gutter to run the script.

def read_data(path):
    """Input: .csv filename string. Output: ndarray"""

    csv = np.recfromcsv("train.csv")
    X = np.column_stack((csv.x1, csv.x2, csv.x3, csv.x4, csv.x5))
    y = np.column_stack((csv.y,))
    
    return X,y

def compute_RMSE(y, y_hat):
    
    return math.sqrt(np.square(np.subtract(y,y_hat)).mean())


if __name__ == '__main__':

    datapath = "data/train.csv"
    X,y = read_data(datapath)

    print("Data file path: " + datapath + ". Number of data points: ")
    print(len(X[:,1]))

    alphas = np.array([0.1, 1., 10., 100., 200.])
    # X = np.arange(50).reshape(10,5) #debug

    fold_size = int(len(X[:,1])/10.0)
    error = np.array([])


    #print (data_dict)

    #training_model = RidgeCV(alphas=[0.1, 1, 10.0, 100.0, 200.0], cv=10).fit(X, y)
    #training_model.score(X, y)



