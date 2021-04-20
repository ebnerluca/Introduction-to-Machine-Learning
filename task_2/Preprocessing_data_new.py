from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


########################################################################################################################
#### helper functions
########################################################################################################################

def calculate_time_features(data, n_samples):
    x = []
    features = [np.nanmedian, np.nanmean, np.nanvar, np.nanmin,
           np.nanmax]
    for index in range(int(data.shape[0] / n_samples)):
        assert data[n_samples * index, 0] == data[n_samples * (index + 1) - 1, 0], \
        'Ids are {}, {}'.format(data[n_samples * index, 0], data[n_samples * (index + 1) - 1, 0])
        patient_data = data[n_samples * index:n_samples * (index + 1), 2:]
        feature_values = np.empty((len(features), data[:, 2:].shape[1]))
        for i, feature in enumerate(features):
            feature_values[i] = feature(patient_data, axis=0)
        x.append(feature_values.ravel())
    return np.array(x)


def reduced_pid(pid):
    pid_reduced = np.zeros((int(np.shape(pid)[0] / 12), 1), dtype=int)
    for i in range(int(np.shape(pid)[0] / 12)):
        pid_reduced[i] = int(pid[i * 12])

    return pid_reduced

########################################################################################################################
#### Load data
########################################################################################################################

train_data = pd.read_csv('data/train_features.csv')
test_data = pd.read_csv('data/test_features.csv')

########################################################################################################################
#### New Preprocessing (same for all subtasks)
########################################################################################################################

x_train = calculate_time_features(train_data.to_numpy(), 12)
x_test = calculate_time_features(test_data.to_numpy(), 12)

train_scaler = StandardScaler()
train_scaler.fit(x_train)
x_train_scaled = train_scaler.transform(x_train)

test_scaler = StandardScaler()
test_scaler.fit(x_test)
x_test_scaled = train_scaler.transform(x_test)

imp_median_1 = SimpleImputer(strategy='median')
imp_median_1.fit(x_train_scaled)
x_train_scaled_imputed = imp_median_1.transform(x_train_scaled)

imp_median_2 = SimpleImputer(strategy='median')
imp_median_2.fit(x_test_scaled)
x_test_scaled_imputed = imp_median_2.transform(x_test_scaled)

pd.DataFrame(x_train_scaled_imputed).to_csv('data/preprocessed/train_features_preprocessed_new.csv', index=False, header=True)
pd.DataFrame(x_test_scaled_imputed).to_csv('data/preprocessed/test_features_preprocessed_new.csv', index=False, header=True)



print('All the necessary data files were created')
