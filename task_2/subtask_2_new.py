import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn.metrics as metrics


if __name__ == '__main__':

    ### read the data
    print("Reading data...", end=" ", flush=True)
    data = np.genfromtxt("data/preprocessed/train_features_preprocessed_new.csv", delimiter=",",
                               skip_header=True)
    labels = np.genfromtxt("data/train_labels.csv", delimiter=",", skip_header=True)[:,11]
    test = np.genfromtxt("data/preprocessed/test_features_preprocessed_new.csv", delimiter=",",
                                skip_header=True)
    print("Done.")

    ### cross validation of the data
    classifier = HistGradientBoostingClassifier()

    scores = cross_val_score(classifier, data, labels, cv=5, scoring='roc_auc', verbose=True)
    print("Cross-Validation score {score:.3f},"
          " Standard Deviation {err:.3f}"
          .format(score = scores.mean(), err = scores.std()))

    ### fit with the training set and generate test set output
    classifier = classifier.fit(data, labels)
    predictions = classifier.predict_proba(test)[:, 1]
    print("Training score:", metrics.roc_auc_score(labels, classifier.predict_proba(data)[:, 1]))
    
    output_array = predictions
    output_path = "data/output/subtask_2_new_labels.csv"
    pd.DataFrame(output_array).to_csv(output_path,
                                        header=["LABEL_Sepsis"], index=None, float_format="%.3f")

    print(f"Predicted labels saved to {output_path}.")