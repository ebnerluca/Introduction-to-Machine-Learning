from sklearn.model_selection import train_test_split
import numpy as np
from numpy import genfromtxt
import math
import pandas as pd


def read_data(train_features_path, train_labels_path, test_features_path):
    print("Reading data...", end=" ", flush=True)

    train_features = np.genfromtxt(train_features_path, delimiter=",", skip_header=True)
    train_labels = np.genfromtxt(train_labels_path, delimiter=",", skip_header=True)
    test_features = np.genfromtxt(test_features_path, delimiter=",", skip_header=True)

    print("Done.")

    return train_features, train_labels, test_features


def save_preprocessed_data(preprocessed_train_features):
    print("Saving preprocessed data...", end=" ", flush=True)

    df = pd.DataFrame(preprocessed_train_features,
                      columns=["pid", "Time", "Age", "EtCO2", "PTT", "BUN", "Lactate", "Temp", "Hgb", "HCO3",
                               "BaseExcess", "RRate", "Fibrinogen", "Phosphate", "WBC", "Creatinine", "PaCO2",
                               "AST", "FiO2", "Platelets", "SaO2", "Glucose", "ABPm", "Magnesium", "Potassium",
                               "ABPd", "Calcium", "Alkalinephos", "SpO2", "Bilirubin_direct", "Chloride",
                               "Hct", "Heartrate", "Bilirubin_total", "TroponinI", "ABPs", "pH"])

    df.to_csv("data/preprocessed/preprocessed_test_features.csv", index=False,header=True)

    print("Done.")


if __name__ == '__main__':
    train_features, train_labels, test_features = read_data("data/train_features.csv",
                                                            "data/train_labels.csv",
                                                            "data/test_features.csv")

    print(f"shape of train_features: {train_features.shape}")
    print(f"shape of train_labels: {train_labels.shape}")
    print(f"shape of test_features: {test_features.shape}")

    ## Preprocessing
    # reshape in dimensions (time, patients, values)
    print("Preprocessing data...", end=" ", flush=True)
    n_hours = 12
    n_columns = train_features.shape[1] - 2  # minus 2 because pid is not relevant and we reshape in time dimension
    n_rows = int(train_features.shape[0] / n_hours)
    preprocessed_train_features = train_features[:, 2:].reshape((n_hours, n_rows, n_columns))
    print("Done.")
    print(f"shape of preprocessed_train_features: {preprocessed_train_features.shape}")

    # impute / delete missing information
    # save preprocessed data
    # save_preprocessed_data(preprocessed_train_features)

    ## SUBTASK 1: Ordering of medical test

    ## SUBTASK 2: Sepsis prediction

    ## SUBTASK 3: Key vital signs prediction

    # np.savetxt("results.csv", weights, delimiter="\n")
