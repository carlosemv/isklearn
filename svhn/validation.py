#!/usr/bin/env python
import sys
import argparse
import numpy as np

from isklearn import ISKLEARN
from sklearn.metrics import accuracy_score
from ingestion import ingestion

if __name__=="__main__":
    isk = ISKLEARN('classification')

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str)
    args = parser.parse_known_args()[0]
    stochastic = ("DecisionTree", "RandomForest",  "MLP",
        "AdaBoost", "LogisticRegression")
    reps = 1 if args.algorithm in stochastic else 1

    X_train, X_test, y_train, y_test = ingestion()

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
