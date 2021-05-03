# feature separation code taken from:
# https://github.com/yardenas/ethz-intro-ml

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# read data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


def feature_separation(data):
    features = []
    for sequence in data['Sequence'].tolist():
        features.append([letter for letter in sequence])
    return features


encoder = OneHotEncoder(sparse=False)
X_train = encoder.fit_transform(feature_separation(train_data))
X_test = encoder.transform(feature_separation(test_data))
y_train = train_data['Active'].to_numpy()

classifier = HistGradientBoostingClassifier(learning_rate=0.21, max_iter=200, max_leaf_nodes=100, min_samples_leaf=100, scoring='f1')
scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1', verbose=True)

print("Cross-Validation score {score:.3f},"
          " Standard Deviation {err:.3f}"
          .format(score=scores.mean(), err=scores.std()))

classifier = classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

output_array = np.around(predictions, decimals=0)
output_path = "data/predictions.csv"
pd.DataFrame(output_array).to_csv(output_path, header=False, index=False)
print(f"Predicted labels saved to {output_path}.")





