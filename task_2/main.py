from sklearn.model_selection import train_test_split
import numpy as np
from numpy import genfromtxt
import math


def read_data(train_features_path, train_labels_path, test_features_path):
    csv = np.recfromcsv(train_features_path)
    train_features = np.column_stack((csv.pid, csv.Time, csv.Age, csv.EtCO2, csv.PTT, csv.BUN, csv.Lactate, csv.Temp,
                                      csv.Hgb, csv.HCO3, csv.BaseExcess, csv.RRate, csv.Fibrinogen, csv.Phosphate,
                                      csv.WBC, csv.Creatinine, csv.PaCO2, csv.AST, csv.FiO2, csv.Platelets, csv.SaO2,
                                      csv.Glucose, csv.ABPm, csv.Magnesium, csv.Potassium, csv.ABPd, csv.Calcium,
                                      csv.Alkalinephos, csv.SpO2, csv.Bilirubin_direct, csv.Chloride, csv.Hct,
                                      csv.Heartrate, csv.Bilirubin_total, csv.TroponinI, csv.ABPs, csv.pH))

    csv = np.recfromcsv(train_labels_path)
    train_labels = np.column_stack((csv.pid, csv.LABEL_BaseExcess, csv.LABEL_Fibrinogen, csv.LABEL_AST,
                                    csv.LABEL_Alkalinephos, csv.LABEL_Bilirubin_total, csv.LABEL_Lactate,
                                    csv.LABEL_TroponinI, csv.LABEL_SaO2, csv.LABEL_Bilirubin_direct, csv.LABEL_EtCO2,
                                    csv.LABEL_Sepsis, csv.LABEL_RRate, csv.LABEL_ABPm, csv.LABEL_SpO2,
                                    csv.LABEL_Heartrate))

    csv = np.recfromcsv(test_features_path)
    test_features = np.column_stack((csv.pid, csv.Time, csv.Age, csv.EtCO2, csv.PTT, csv.BUN, csv.Lactate, csv.Temp,
                                     csv.Hgb, csv.HCO3, csv.BaseExcess, csv.RRate, csv.Fibrinogen, csv.Phosphate,
                                     csv.WBC, csv.Creatinine, csv.PaCO2, csv.AST, csv.FiO2, csv.Platelets, csv.SaO2,
                                     csv.Glucose, csv.ABPm, csv.Magnesium, csv.Potassium, csv.ABPd, csv.Calcium,
                                     csv.Alkalinephos, csv.SpO2, csv.Bilirubin_direct, csv.Chloride, csv.Hct,
                                     csv.Heartrate, csv.Bilirubin_total, csv.TroponinI, csv.ABPs, csv.pH))

    return train_features, train_labels, test_features


if __name__ == '__main__':
    train_features, train_labels, test_features = read_data("data/train_features.csv",
                                                            "data/train_labels.csv",
                                                            "data/test_features.csv")

    ## SUBTASK 1: Ordering of medical test

    ## SUBTASK 2: Sepsis prediction

    ## SUBTASK 3: Key vital signs prediction

    # np.savetxt("results.csv", weights, delimiter="\n")
