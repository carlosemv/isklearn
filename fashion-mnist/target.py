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

def load_fashion(path, kind='train'):
    labels_path = os.path.join(path,
        '{}-labels-idx1-ubyte'.format(kind))
    images_path = os.path.join(path,
        '{}-images-idx3-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
            offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
            offset=16).reshape(len(labels), 784)

    return images, labels
    
if __name__=="__main__":
    print(sys.argv)
    t0 = time.time()
    
    X = np.load('data/fashion-mnist/cifar10_ResNet56v1_train_data.npy')
    _, y = load_fashion('./data/fashion-mnist')
    fashion_data = (X, y)

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str)
    parser.add_argument('--task', type=str)
    args = parser.parse_known_args()[0]

    fold = args.instance
    task = args.task

    isk = ISKLEARN(task)
    X, y = ingestion(fashion_data, int(fold))
    acc_scores = isk.validation(X, y)
    result = np.mean(acc_scores)

    print(-1*np.mean(result), time.time()-t0)