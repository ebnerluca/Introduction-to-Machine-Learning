### test code taken from: https://github.com/yardenas/ethz-intro-ml/blob/master/project_2/project_2.ipynb


import numpy as np
import pandas as pd
train_data = pd.read_csv('train_features.csv')
labels = pd.read_csv('train_labels.csv')
test_data = pd.read_csv('test_features.csv')


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

x_train = calculate_time_features(train_data.to_numpy(), 12)
x_test = calculate_time_features(test_data.to_numpy(), 12)

subtask1_labels_ids = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
         'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
y_train = labels[subtask1_labels_ids].to_numpy()

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (GridSearchCV,
    cross_val_score, KFold)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

# TODO (yarden):
# feature selection.
# parameters tuning (subsample, learning rate).
for i, label in enumerate(subtask1_labels_ids):
    pipeline = make_pipeline(
                        SimpleImputer(strategy='median'),
                        StandardScaler(),
                        HistGradientBoostingClassifier())
    scores = cross_val_score(pipeline, x_train, y_train[:, i],
                                cv=5,
                                scoring='roc_auc',
                                verbose=True)
    print("Cross-validation score is {score:.3f},"
          " standard deviation is {err:.3f}"
          .format(score = scores.mean(), err = scores.std()))



df = pd.DataFrame({'pid': test_data.iloc[0::12, 0].values})
for i, label in enumerate(subtask1_labels_ids):
    pipeline = pipeline.fit(x_train, y_train[:, i].ravel())
    print("Training score:", metrics.roc_auc_score(y_train[:, i], pipeline.predict_proba(x_train)[:, 1]))
    predictions = pipeline.predict_proba(x_test)[:, 1]
    df[label] = predictions

subtask2_labels_ids = ['LABEL_Sepsis']
y_train = labels[subtask2_labels_ids].to_numpy().ravel()

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (GridSearchCV,
    cross_val_score, KFold)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

# TODO (yarden):
# feature selection.
# parameters tuning (subsample, learning rate).
pipeline = make_pipeline(
                    SimpleImputer(strategy='median'),
                    StandardScaler(),
                    HistGradientBoostingClassifier())

scores = cross_val_score(pipeline, x_train, y_train,
                            cv=5,
                            scoring='roc_auc',
                            verbose=True)
print("Cross-validation score is {score:.3f},"
      " standard deviation is {err:.3f}"
      .format(score = scores.mean(), err = scores.std()))

pipeline = pipeline.fit(x_train, y_train)
predictions = pipeline.predict_proba(x_test)[:, 1]
print("Training score:", metrics.roc_auc_score(y_train, pipeline.predict_proba(x_train)[:, 1]))
df[subtask2_labels_ids[0]] = predictions

subtask3_labels_ids = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2',
                      'LABEL_Heartrate']
y_train = labels[subtask3_labels_ids].to_numpy()

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (GridSearchCV,
    cross_val_score, KFold)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

for i, label in enumerate(subtask3_labels_ids):
    pipeline = make_pipeline(
                        SimpleImputer(strategy='median'),
                        HistGradientBoostingRegressor(max_depth=3))
    scores = cross_val_score(pipeline, x_train, y_train[:, i],
                            cv=5,
                            scoring='r2',
                            verbose=True)
    print("Cross-validation score is {score:.3f},"
          " standard deviation is {err:.3f}"
          .format(score = scores.mean(), err = scores.std()))



from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
for i, label in enumerate(subtask3_labels_ids):
    pipeline = make_pipeline(
                        SimpleImputer(strategy='median'),
                        StandardScaler(),
                        HistGradientBoostingRegressor(max_depth=3))
    pipeline = pipeline.fit(x_train, y_train[:, i])
    predictions = pipeline.predict(x_test)
    print("Training score:", metrics.r2_score(y_train[:, i], pipeline.predict(x_train)))
    df[label] = predictions

df.to_csv('prediction.csv', index=False, float_format='%.4f')

