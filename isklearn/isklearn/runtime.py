import sys
import time
import argparse
import numpy as np
import pandas as pd
from isklearn import ISKLEARN
from isklearn.utils import _str_to_bool, ArgumentException
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score

def metafold(X, y, fold, k=20):
    X = pd.DataFrame(X)
    y = pd.Series(y).to_frame()

    skf = StratifiedKFold(n_splits=k, shuffle=False)
    skf_idxs = list(skf.split(X, y))
    kfold_idxs = skf_idxs[fold][1]
    Xk, yk = X.iloc[kfold_idxs], y.iloc[kfold_idxs]

    return (Xk, yk)

def evaluate(ingest_func):
    print(sys.argv)
    t0 = time.time()

    X, _, y, _ = ingest_func()

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--metafolds', type=int)
    parser.add_argument('--sparse', type=_str_to_bool)
    args = parser.parse_known_args()[0]

    if args.metafolds == 1:
        print("using 1 metafold")
        fold = args.instance

        isk = ISKLEARN(args.task, args.sparse)
        X_fold, y_fold = metafold(X, y, int(fold))
        scores = isk.validation(X_fold, y_fold)

        print(-1*np.mean(scores), time.time()-t0)
    else:
        print("using {} metafolds".format(args.metafolds))
        folds = [x for x in args.instance.split(',')]

        all_scores = []
        isk = ISKLEARN(args.task, args.sparse)
        for fold in folds:
            X_fold, y_fold = metafold(X, y, int(fold))
            scores = isk.validation(X_fold, y_fold)
            all_scores.append(np.mean(scores))

        print(-1*np.mean(all_scores), time.time()-t0)


def validate(ingest_func):
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--task', type=str)
    args = parser.parse_known_args()[0]
    stochastic = ("DecisionTree", "RandomForest",  "MLP",
        "AdaBoost", "LogisticRegression")
    reps = 1 if args.algorithm in stochastic else 1

    if args.task == 'classification':
        scoring = accuracy_score
    elif args.task == 'regression':
        scoring = r2_score
    else:
        raise ArgumentException('--task', args.task)

    isk = ISKLEARN(args.task)

    X_train, X_test, y_train, y_test = ingest_func()

    accs = []
    for i in range(reps):
        Xi_train, yi_train, Xi_test, yi_test = isk.preprocess(
            X_train, y_train, X_test, y_test)

        model = isk.build_model()
        model.fit(Xi_train, yi_train)
        yi_pred = model.predict(Xi_test)
        yi_pred = isk.impute(yi_pred)
        scores.append(scoring(yi_test, yi_pred))
    print(scores, np.mean(accs))
    print(" ".join(sys.argv[1:]))
    print()
