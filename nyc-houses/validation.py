#!/usr/bin/env python
import sys, os
sys.path.append('..')

import time
import argparse
import numpy as np
import pandas as pd

from isklearn.isklearn import ISKLEARN
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score


if __name__=="__main__":
    isk = ISKLEARN('regression')

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str)
    args = parser.parse_known_args()[0]
    stochastic = ("DecisionTree", "RandomForest",  "MLP")
    reps = 3 if args.algorithm in stochastic else 1


    datapath = 'data/nyc-houses/'

    X = pd.read_csv(os.path.join(datapath,'nyc-houses_X.csv'), index_col=0)
    y = pd.read_csv(os.path.join(datapath,'nyc-houses_y.csv'), index_col=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    r2_sum = 0.
    for i in range(reps):
        Xi_train, yi_train, Xi_test, yi_test = isk.preprocess(
            X_train, y_train, X_test, y_test)

        model = isk.build_model()
        model.fit(Xi_train, yi_train)
        yi_pred = model.predict(Xi_test)
        yi_pred = isk.impute(yi_pred)
        r2_sum += r2_score(yi_test, yi_pred)
    print(r2_sum/reps)
    print(" ".join(sys.argv[1:]))
    print()

