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

    skf = StratifiedKFold(n_splits=20, shuffle=False)
    skf_idxs = list(skf.split(X, y))
    kfold_idxs = skf_idxs[fold][1]
    Xk, yk = X.iloc[kfold_idxs], y.iloc[kfold_idxs]

    return (Xk, yk)

    
if __name__=="__main__":
    print(sys.argv)
    t0 = time.time()

    X = np.load('data/cifar10/imagenet_ResNet56v1_13l_train_data.npy')
    y = np.load('data/cifar10/train_labels.npy')
    cifar_data = (X, y)

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
        X, y = ingestion(cifar_data, int(fold))
        acc_scores = isk.validation(X, y)
        result = np.mean(acc_scores)

        print(-1*np.mean(result), time.time()-t0)
    else:
        print("using {} metafolds".format(args.metafolds))
        folds = [x for x in args.instance.split(',')]
        task = args.task

        accs = []
        isk = ISKLEARN(task)
        for fold in folds:
            X, y = ingestion(cifar_data, int(fold))
            acc_scores = isk.validation(X, y)
            result = np.mean(acc_scores)
            accs.append(result)

        print(-1*np.mean(accs), time.time()-t0)