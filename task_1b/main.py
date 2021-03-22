# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import genfromtxt
import math

# Press the green button in the gutter to run the script.

def read_data(path):
    """Input: .csv filename string. Output: ndarray"""

    csv = np.recfromcsv(path)
    X = np.column_stack((csv.x1, csv.x2, csv.x3, csv.x4, csv.x5))
    y = np.column_stack((csv.y,))

    return X,y

def transform_x(X):
    return np.array((X[0],
                    X[1],
                    X[2],
                    X[3],
                    X[4],
                    X[0]**2,
                    X[1]**2,
                    X[2]**2,
                    X[3]**2,
                    X[4]**2,
                    np.exp(X[0]),
                    np.exp(X[1]),
                    np.exp(X[2]),
                    np.exp(X[3]),
                    np.exp(X[4]),
                    np.cos(X[0]),
                    np.cos(X[1]),
                    np.cos(X[2]),
                    np.cos(X[3]),
                    np.cos(X[4]),
                    1
                    ))

def compute_RMSE(y, y_hat):
    
    return math.sqrt(np.square(np.subtract(y,y_hat)).mean())


if __name__ == '__main__':

    datapath = "data/train.csv"
    X,y = read_data(datapath)

    # Transform X
    X = np.apply_along_axis(transform_x, axis=1, arr=X)
    
    #model = RidgeCV(alphas=[0.1, 0.420, 0.69, 1, 3, 4.20, 5, 6.90, 10, 42, 69], alpha_per_target = True).fit(X,y)

    lamda_vec = np.array([0.1, 0.420, 0.69, 1, 3, 4.20, 5, 6.90, 10, 42, 69])

    model = MultiTaskLassoCV(alphas = lamda_vec, cv = 10, fit_intercept=False)

    model.fit(X, y)
    
    weights = model.coef_

    np.savetxt("results.csv", weights, delimiter="\n")



