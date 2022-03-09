#!/usr/bin/env python
import time
import argparse
import numpy as np
import pandas as pd
from isklearn import ISKLEARN
from sklearn.model_selection import StratifiedKFold
from ingestion import ingestion

def metafold(X, y, fold, k=20):
    X = pd.DataFrame(X)
    y = pd.Series(y).to_frame()

    skf = StratifiedKFold(n_splits=k, shuffle=False)
    skf_idxs = list(skf.split(X, y))
    kfold_idxs = skf_idxs[fold][1]
    Xk, yk = X.iloc[kfold_idxs], y.iloc[kfold_idxs]

    return (Xk, yk)
    
if __name__=="__main__":
    print(sys.argv)
    t0 = time.time()

    X, _, y, _ = ingestion()

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--metafolds', type=int)
    args = parser.parse_known_args()[0]

    if args.metafolds == 1:
        print("using 1 MF")
        fold = args.instance
        task = args.task

        isk = ISKLEARN(task)
        X_fold, y_fold = metafold(X, y, int(fold))
        acc_scores = isk.validation(X_fold, y_fold)
        result = np.mean(acc_scores)

        print(-1*np.mean(result), time.time()-t0)
    else:
        print("using {} metafolds".format(args.metafolds))
        folds = [x for x in args.instance.split(',')]
        task = args.task

        accs = []
        isk = ISKLEARN(task)
        for fold in folds:
            X_fold, y_fold = metafold(X, y, int(fold))
            acc_scores = isk.validation(X_fold, y_fold)
            result = np.mean(acc_scores)
            accs.append(result)

        print(-1*np.mean(accs), time.time()-t0)