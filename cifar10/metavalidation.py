#!/usr/bin/env python
import sys
sys.path.append('..')

import os
import time
import argparse
import numpy as np
import pandas as pd
from isklearn.isklearn import ISKLEARN
from sklearn.model_selection import StratifiedKFold

def ingestion(data, fold):
    X, y = data
    X = pd.DataFrame(X)
    y = pd.Series(y).to_frame()

    skf = StratifiedKFold(n_splits=20, shuffle=True)
    skf_idxs = list(skf.split(X, y))
    kfold_idxs = skf_idxs[fold][1]
    Xk, yk = X.iloc[kfold_idxs], y.iloc[kfold_idxs]

    return (Xk, yk)

    
if __name__=="__main__":
    print(sys.argv)
    t0 = time.time()

    X = np.load('data/cifar10/train_data.npy')
    y = np.load('data/cifar10/train_labels.npy')
    cifar_data = (X, y)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--instance', type=str)
    # parser.add_argument('--task', type=str)
    # args = parser.parse_known_args()[0]

    # folds = [x for x in args.instance.split(',')]
    # task = args.task

    for fold in range(20):
        # t0 = time.time()

        # accs = []
        # isk = ISKLEARN(task)
        X, y = ingestion(cifar_data, int(fold))
        print(fold)
        print(y.iloc[:,0].value_counts())
        print()
        # acc_scores = isk.validation(X, y)
        # acc = np.mean(acc_scores)

        # print(fold, -1*acc, time.time()-t0)