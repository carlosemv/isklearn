#!/usr/bin/env python
import sys
sys.path.append('..')

import argparse
import numpy as np
import pandas as pd

from isklearn.isklearn import ISKLEARN
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


if __name__=="__main__":
    isk = ISKLEARN('classification')

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str)
    args = parser.parse_known_args()[0]
    stochastic = ("DecisionTree", "RandomForest",  "MLP",
        "AdaBoost", "LogisticRegression")
    reps = 5 if args.algorithm in stochastic else 1


    X_train = np.load('data/svhn/train_data.npy')
    y_train = np.load('data/svhn/train_labels.npy')
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train).values.ravel()

    X_test = np.load('data/svhn/test_data.npy')
    y_test = np.load('data/svhn/test_labels.npy')
    X_test = pd.DataFrame(X_test)
    y_test = pd.Series(y_test).values.ravel()

    accs = []
    for i in range(reps):
        Xi_train, yi_train, Xi_test, yi_test = isk.preprocess(
            X_train, y_train, X_test, y_test)

        model = isk.build_model()
        model.fit(Xi_train, yi_train)
        yi_pred = model.predict(Xi_test)
        yi_pred = isk.impute(yi_pred)
        accs.append(accuracy_score(yi_test, yi_pred))
    print(accs, np.mean(accs))
    print(" ".join(sys.argv[1:]))
    print()