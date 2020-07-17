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
from sklearn.metrics import accuracy_score

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
    isk = ISKLEARN('classification')
    data_path = "/home/cemvieira/isklearn/data/fashion-mnist"

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str)
    args = parser.parse_known_args()[0]
    stochastic = ("DecisionTree", "RandomForest",  "MLP",
        "AdaBoost", "LogisticRegression")
    reps = 5 if args.algorithm in stochastic else 1

    # X_train = np.load('data/fashion-mnist/imagenet_ResNet56v1_13l_train_data.npy')
    X_train, y_train = load_fashion(data_path)
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train).values.ravel()

    # X_test = np.load('data/fashion-mnist/imagenet_ResNet56v1_13l_test_data.npy')
    X_test, y_test = load_fashion(data_path, 't10k')
    X_test = pd.DataFrame(X_test)
    y_test = pd.Series(y_test).values.ravel()

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