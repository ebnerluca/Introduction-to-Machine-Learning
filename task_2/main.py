from sklearn.model_selection import train_test_split
import numpy as np
from numpy import genfromtxt
import math


def read_data(train_features_path, train_labels_path, test_features_path):

    print("Reading data...", end=" ", flush=True)

    train_features = np.genfromtxt(train_features_path, delimiter=",", skip_header=True)
    train_labels = np.genfromtxt(train_labels_path, delimiter=",", skip_header=True)
    test_features = np.genfromtxt(test_features_path, delimiter=",", skip_header=True)

    print("Done.")

    return train_features, train_labels, test_features


if __name__ == '__main__':

    train_features, train_labels, test_features = read_data("data/train_features.csv",
                                                            "data/train_labels.csv",
                                                            "data/test_features.csv")
    
    print(f"shape of train_features: {train_features.shape}")
    print(f"shape of train_labels: {train_labels.shape}")
    print(f"shape of test_features: {test_features.shape}")

    ## SUBTASK 1: Ordering of medical test

    ## SUBTASK 2: Sepsis prediction

    ## SUBTASK 3: Key vital signs prediction

    # np.savetxt("results.csv", weights, delimiter="\n")
