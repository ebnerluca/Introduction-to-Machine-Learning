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

    data = genfromtxt(path, delimiter=',')
    X = data[1:,1:]
    y = data[1:,0]
    return X,y

def compute_RMSE(y, y_hat):
    
    return math.sqrt(np.square(np.subtract(y,y_hat)).mean())


if __name__ == '__main__':

    datapath = "data/train.csv"
    X,y = read_data(datapath)

    # Debug
    #print(np.shape(X))
    #print(X)
    #print(np.shape(y))
    #print(y)
    print(len(X[:,1]))

    alphas = np.array([0.1, 1., 10., 100., 200.])
    # X = np.arange(50).reshape(10,5) #debug

    fold_size = int(len(X[:,1])/10.0)
    #print(fold_size)

    for i in range (len(alphas)):
        for j in range(10):
            if j==0:
                x_j = X[ fold_size: ,:]
                y_j = y[ fold_size:]
            else:
                x_j1 = X[ 0:j*fold_size , : ]
                x_j2 = X[ (j+1)*fold_size:, : ]
                x_j = np.vstack( (x_j1, x_j2) )
                y_j1 = y[ 0:j*fold_size ]
                y_j2 = y[ (j+1)*fold_size: ]
                y_j = np.hstack( (y_j1, y_j2) )
            #print(y_j)
            training_model = Ridge(alpha = alphas[i])
            
           

    #print (data_dict)

    #training_model = RidgeCV(alphas=[0.1, 1, 10.0, 100.0, 200.0], cv=10).fit(X, y)
    #training_model.score(X, y)



