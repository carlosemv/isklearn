#!/usr/bin/env python
import sys
sys.path.append('..')

import time
import argparse
import numpy as np
import pandas as pd
from isklearn.isklearn import ISKLEARN

from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

if __name__=="__main__":
    path = "/home/cemvieira/isklearn/data/lmrd"
    n_tokens = 89527
    X_train, y_train = load_svmlight_file(path+"/train/labeledBow.feat", n_features=n_tokens)
    X_train = TfidfTransformer().fit_transform(X_train)
    y_train[y_train <= 4] = 0
    y_train[y_train >= 7] = 1

    X_test, y_test = load_svmlight_file(path+"/test/labeledBow.feat", n_features=n_tokens)
    X_test = TfidfTransformer().fit_transform(X_test)
    y_test[y_test <= 4] = 0
    y_test[y_test >= 7] = 1

    isk = ISKLEARN(task='classification', sparse=True)
    X_train, y_train, X_test, y_test = isk.preprocess(X_train, y_train, X_test, y_test)

    model = isk.build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = isk.impute(y_pred)
    print(accuracy_score(y_test, y_pred))
    print(" ".join(sys.argv))
    print()