#!/usr/bin/env python
import sys
sys.path.append('..')

import time
import argparse
import numpy as np
import pandas as pd
from isklearn.isklearn import ISKLEARN

from mnist import MNIST
from sklearn.model_selection import StratifiedKFold

def ingestion(data, fold):
    X, y = data.load_training()
    X = pd.DataFrame(X)
    y = pd.Series(y).to_frame()

    skf = StratifiedKFold(n_splits=20, shuffle=False)
    skf_idxs = list(skf.split(X, y))
    kfold_idxs = skf_idxs[fold][1]
    Xk, yk = X.iloc[kfold_idxs], y.iloc[kfold_idxs]

    return (Xk, yk)
    
if __name__=="__main__":
    print(sys.argv)
    t0 = time.time()

    mndata = MNIST('./data')

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str)
    parser.add_argument('--config_id', type=str)
    parser.add_argument('--task', type=str)
    args = parser.parse_known_args()[0]

    folds = [x for x in args.instance.split(',')]
    config_id = args.config_id
    task = args.task

    accs = []
    isk = ISKLEARN(task)
    for fold in folds:
        X, y = ingestion(mndata, int(fold))
        acc_scores = isk.validation(X, y)
        result = np.mean(acc_scores)
        accs.append(result)

    print(-1*np.mean(accs), time.time()-t0)