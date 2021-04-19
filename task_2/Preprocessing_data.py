import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

########################################################################################################################
#### helper functions
########################################################################################################################

def reduced_pid(pid):
    pid_reduced = np.zeros((int(np.shape(pid)[0] / 12), 1), dtype=int)
    for i in range(int(np.shape(pid)[0] / 12)):
        pid_reduced[i] = int(pid[i * 12])

    return pid_reduced

def preprocessing_function_1(X_raw):
    n = int(np.shape(X_raw)[0] / 12)
    d = np.shape(X_raw)[1]
    X_average = np.zeros((n, d))

    # Loop over features
    for j in range(d):
        # loop over 12h data points
        for i in range(n):
            sum = 0
            number = 0
            # loop within those 12h
            for t in range(0, 12):
                if X_raw[12 * i + t][j] == X_raw[12 * i + t][j]:
                    sum += X_raw[12 * i + t][j]
                    number += 1

            if sum == 0:
                X_average[i][j] = 0
            else:
                X_average[i][j] = sum / number
    return X_average

def preprocessing_function_2(X_raw):
    n = int(np.shape(X_raw)[0] / 12)
    d = int(np.shape(X_raw)[1])
    X = np.zeros((n, N_STATISTIC_FEATURES * d))

    # loop over all features;
    for j in range(d):
        X_n_trend_change = np.zeros(n)
        X_mean_intensity_change = np.zeros(n)
        X_median_intensity_change = np.zeros(n)
        X_min_intensity_change = np.zeros(n)
        X_max_intensity_change = np.zeros(n)

        # loop over eatch t
        for i in range(n):
            Y = []
            non_nan_entries = []
            X_values = []
            # check how many nan elements
            for t in range(0, 12):
                if X_raw[12 * i + t][j] == X_raw[12 * i + t][j]:
                    X_values.append(X_raw[12 * i + t][j])

            if len(X_values) < 2:
                X_n_trend_change[i] = 0
                X_mean_intensity_change[i] = 0
                X_median_intensity_change[i] = 0
                X_min_intensity_change[i] = 0
                X_max_intensity_change[i] = 0

            else:
                Y.append(X_values[0])  # First value is always maxima or minima
                # Filling Y Vector
                for k in range(1, len(X_values) - 1):
                    # Check for local maxima
                    if X_values[k] > X_values[k - 1] and X_values[k] >= X_values[k + 1]:
                        Y.append(X_values[k])
                    # Check for local minima
                    elif X_values[k] < X_values[k - 1] and X_values[k] <= X_values[k + 1]:
                        Y.append(X_values[k])
                Y.append(X_values[len(X_values) - 1])  # Last value is always maxima or minima

                # Filling Vector z_i = abs(y(i+1) - y(i))
                intensity_change = np.zeros(len(Y) - 1)
                for k in range(0, len(Y) - 1):
                    intensity_change[k] = abs(Y[k + 1] - Y[k])

                X_n_trend_change[i] = len(Y)
                X_mean_intensity_change[i] = np.mean(intensity_change)
                X_median_intensity_change[i] = np.median(intensity_change)
                X_min_intensity_change[i] = np.min(intensity_change)
                X_max_intensity_change[i] = np.max(intensity_change)

        X[:, j * N_STATISTIC_FEATURES + 0] = X_n_trend_change
        X[:, j * N_STATISTIC_FEATURES + 1] = X_mean_intensity_change
        X[:, j * N_STATISTIC_FEATURES + 2] = X_median_intensity_change
        X[:, j * N_STATISTIC_FEATURES + 3] = X_min_intensity_change
        X[:, j * N_STATISTIC_FEATURES + 4] = X_max_intensity_change

    return X


def preprocessing_function_3(X_raw):
    n = int(np.shape(X_raw)[0] / 12)
    d = int(np.shape(X_raw)[1])
    X = np.zeros((n, N_STATISTIC_FEATURES * d))

    for j in range(d):
        X_12_mean = np.zeros(n)
        X_12_variance = np.zeros(n)
        X_12_development = np.zeros(n)
        X_12_min = np.zeros(n)
        X_12_max = np.zeros(n)

        for i in range(n):

            non_nan_entries = []
            # check how many nan elements
            for t in range(0, 12):
                if X_raw[12 * i + t][j] == X_raw[12 * i + t][j]:
                    non_nan_entries.append(t)

            # Only nan elements
            if len(non_nan_entries) == 0:
                X_12_mean[i] = 0
                X_12_variance[i] = 0
                X_12_min[i] = 0
                X_12_max[i] = 0
                X_12_development[i] = 0

            # 1 non nan element, compute mean and variance, no development
            elif len(non_nan_entries) == 1:
                X_12_mean[i] = X_raw[12 * i + non_nan_entries[0]][j]
                X_12_variance[i] = 0
                X_12_min[i] = 0
                X_12_max[i] = 0
                X_12_development[i] = 0
            else:
                X_12_mean[i] = np.nanmean(X_raw[12 * i: 12 * i + 12, j])
                X_12_variance[i] = np.nanvar(X_raw[12 * i: 12 * i + 12, j])
                X_12_min[i] = np.nanmin(X_raw[12 * i: 12 * i + 12, j])
                X_12_max[i] = np.nanmax(X_raw[12 * i: 12 * i + 12, j])

                for t in range(len(non_nan_entries) - 1):
                    X_12_development[i] += X_raw[12 * i + non_nan_entries[t + 1]][j] - \
                                           X_raw[12 * i + non_nan_entries[t]][j]

                X_12_development[i] /= len(non_nan_entries)

        X[:, j * N_STATISTIC_FEATURES + 0] = X_12_mean
        X[:, j * N_STATISTIC_FEATURES + 1] = X_12_variance
        X[:, j * N_STATISTIC_FEATURES + 2] = X_12_min
        X[:, j * N_STATISTIC_FEATURES + 3] = X_12_max
        X[:, j * N_STATISTIC_FEATURES + 4] = X_12_development

    return X

########################################################################################################################
#### Load data
########################################################################################################################

df = pd.read_csv('data/train_features.csv')
df_test = pd.read_csv('data/test_features.csv')

########################################################################################################################
#### Preprocessing Train Features Task 1
########################################################################################################################
X_raw = np.array(df.iloc[:, 2:37])
pid = np.array(df.iloc[:, 0])
pid_reduced = reduced_pid(pid)
X_average_12_hours = preprocessing_function_1(X_raw)

scaler = StandardScaler()
scaler.fit(X_average_12_hours)
X_scaled = scaler.transform(X_average_12_hours)

# Writing to file
X_complete = np.concatenate((pid_reduced, X_scaled), axis=1)

new_df = list(df.columns.values)
del new_df[1]  # deleting time column
pd.DataFrame(X_complete, columns=new_df).to_csv('data/preprocessed/train_features_preprocessed_task1.csv', index=False, header=True)

########################################################################################################################
#### Preprocessing Test Features Task 1
########################################################################################################################
X_raw = np.array(df_test.iloc[:, 2:37])
pid = np.array(df_test.iloc[:, 0])
pid_reduced = reduced_pid(pid)
X_average_12_hours = preprocessing_function_1(X_raw)

# Same standardizer as for train data set needs to be used
X_scaled = scaler.transform(X_average_12_hours)

# Writing to file
X_complete = np.concatenate((pid_reduced, X_scaled), axis=1)

new_df = list(df_test.columns.values)
del new_df[1]  # deleting time column
pd.DataFrame(X_complete, columns=new_df).to_csv('data/preprocessed/test_features_preprocessed_task1.csv', index=False, header=True)

print('Task 1 done')

########################################################################################################################
#### Preprocessing Task 2 Train
########################################################################################################################
# Inspired from https://www.hindawi.com/journals/jhe/2019/5930379/
N_STATISTIC_FEATURES = 5

features = ['Temp', 'RRate', 'Heartrate', 'ABPm']
X_raw = np.array(df.loc[:, features])
X_statistic_raw = preprocessing_function_2(X_raw)
age = np.array(df.iloc[::12, 2]).reshape(-1, 1)
X_statistic_raw = np.concatenate((age, X_statistic_raw), axis=1)

scaler2 = StandardScaler()
scaler2.fit(X_statistic_raw)
X_statistic = scaler2.transform(X_statistic_raw)

labels = ["Age"]

for i in range(0, int((np.shape(X_statistic)[1] - 1) / N_STATISTIC_FEATURES)):
    labels.append(features[i] + "_N")
    labels.append(features[i] + "_Mean")
    labels.append(features[i] + "_Median")
    labels.append(features[i] + "_Min")
    labels.append(features[i] + "_Max")

pd.DataFrame(X_statistic, columns=labels).to_csv('data/preprocessed/train_features_preprocessed_task2.csv', index=False, header=True)

########################################################################################################################
#### Preprocessing Task 2 Test
########################################################################################################################
# Inspired from https://www.hindawi.com/journals/jhe/2019/5930379/
N_STATISTIC_FEATURES = 5

features = ['Temp', 'RRate', 'Heartrate', 'ABPm']
X_raw = np.array(df_test.loc[:, features])
X_statistic_raw = preprocessing_function_2(X_raw)
age = np.array(df_test.iloc[::12, 2]).reshape(-1, 1)
X_statistic_raw = np.concatenate((age, X_statistic_raw), axis=1)

X_statistic = scaler2.transform(X_statistic_raw)
pd.DataFrame(X_statistic, columns=labels).to_csv('data/preprocessed/test_features_preprocessed_task2.csv', index=False, header=True)

print('Task 2 done')



print('All the necessary data files were created')
