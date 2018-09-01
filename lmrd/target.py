#!/usr/bin/env python
import sys
sys.path.append('..')

import time
import argparse
import numpy as np
from isklearn import ISKLEARN

from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold

def ingestion(fold):

    #X, y = load_svmlight_file("./data/lmrd/train/labeledBow.feat")
    #X = TfidfTransformer().fit_transform(X)
    #y[y <= 4] = 0
    #y[y >= 7] = 1
    X = np.load("training_samples.npy")
    y = np.load("training_labels.npy")

    skf = StratifiedKFold(n_splits=10, shuffle=False)
    skf_idxs = list(skf.split(X, y))
    kfold_idxs = skf_idxs[fold][1]
    Xk, yk = X[kfold_idxs], y[kfold_idxs]

    return (Xk, yk)

if __name__=="__main__":
    print(sys.argv)
    t0 = time.time()

    # large movie review dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str)
    parser.add_argument('--config_id', type=str)
    parser.add_argument('--task', type=str)
    args = parser.parse_known_args()[0]

    folds = [x for x in args.instance.split(',')]
    config_id = args.config_id
    task = args.task

    accs = []
    isk = ISKLEARN(task=task, sparse=True)
    for fold in folds:
        X, y = ingestion(int(fold))
        acc_scores = isk.validation(X, y)
        result = np.mean(acc_scores)
        accs.append(result)

    print(-1*np.mean(accs), time.time()-t0)