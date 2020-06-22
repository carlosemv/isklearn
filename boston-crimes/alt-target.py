#!/usr/bin/env python
import sys
sys.path.append('..')

import os
import time
import argparse
import numpy as np
import pandas as pd
from isklearn.isklearn import ISKLEARN


def get_folds_from_places(n_folds, places):
    places = list(places)
    folds = {}
    for i in range(n_folds):
        folds[i] = []
    while len(places)>0:
        for i in range(n_folds):
            if len(places)>0:
                folds[i].append(places.pop())
    return folds

def ingestion(data, fold_of_places):
    X, y = data
    idx = pd.IndexSlice
    return (X.loc[idx[:, fold_of_places],:], y.loc[idx[:, fold_of_places],:])


if __name__=="__main__":
    print(sys.argv)
    t0 = time.time()

    datapath = 'data/crime_data/'

    X = pd.read_csv(os.path.join(datapath,'weekly-boston_X.csv'), index_col=[0,1])
    y = pd.read_csv(os.path.join(datapath,'weekly-boston_y.csv'), index_col=[0,1])
    data = (X, y)

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str)
    parser.add_argument('--task', type=str)
    args = parser.parse_known_args()[0]

    fold = args.instance
    task = args.task

    folds = get_folds_from_places(10, X.index.get_level_values('place').unique())

    r2s = []
    isk = ISKLEARN(task)
    for fold in folds:
        X, y = ingestion(data, folds[int(fold)])
        r2 = isk.validation(X, y)
        result = np.mean(r2)
        r2s.append(result)

    print(-1*np.mean(r2s), time.time()-t0)
