#!/usr/bin/env python
import sys
sys.path.append('..')

import os
import time
import argparse
import numpy as np
import pandas as pd
from isklearn.isklearn import ISKLEARN

from sklearn.model_selection import KFold
def ingestion(data, fold):
    X, y = data
    X = pd.DataFrame(X)
    #y = pd.Series(y)

    skf = KFold(n_splits=10)
    skf_idxs = list(skf.split(X, y))
    kfold_idxs = skf_idxs[fold][1]
    Xk, yk = X.iloc[kfold_idxs], y.iloc[kfold_idxs]

    return (Xk, yk)


if __name__=="__main__":
    print(sys.argv)
    t0 = time.time()

    datapath = 'data/nyc-houses/'

    X = pd.read_csv(os.path.join(datapath,'nyc-houses_X.csv'), index_col=0)
    y = pd.read_csv(os.path.join(datapath,'nyc-houses_y.csv'), index_col=0)
    data = (X, y)

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str)
    parser.add_argument('--task', type=str)
    args = parser.parse_known_args()[0]

    folds = args.instance
    task = args.task

    print(folds, type(folds))
    
    r2s = []
    isk = ISKLEARN(task)
    for fold in folds.split(','):
        X, y = ingestion(data, int(fold))
        r2 = isk.validation(X, y, cv="KFold")
        result = np.mean(r2)
        r2s.append(result)

    print(-1*np.mean(r2s), time.time()-t0)

