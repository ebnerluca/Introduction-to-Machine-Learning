# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':



    training_model = RidgeCV(alphas=[0.1, 1, 10.0, 100.0, 200.0], cv=10).fit(X, y)
    training_model.score(X, y)



