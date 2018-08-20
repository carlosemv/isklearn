#!/usr/bin/env python
import sys
sys.path.append('..')

import time
import argparse
import numpy as np
import pandas as pd
from isklearn.isklearn import ISKLEARN

from ingestion import Ingestion
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

if __name__=="__main__":
    ts_file = "data/crime_districts_daily.csv"
    ts_df = pd.read_csv(ts_file, index_col=0).sum(axis=1).to_frame()

    train_df, test_df = train_test_split(ts_df, test_size=0.2, shuffle=False)

    r2s = {}
    for lags in range(3,16):
        X_train = pd.DataFrame(Ingestion(lags=lags, freq='d').fit_transform(train_df)[0])
        y_train = train_df[0][(6+lags):]

        X_test = pd.DataFrame(Ingestion(lags=lags, freq='d').fit_transform(test_df)[0])
        y_test = test_df[0][(6+lags):]

        isk = ISKLEARN('regression')
        X_train, y_train, X_test, y_test = isk.preprocess(X_train, y_train, X_test, y_test)

        model = isk.build_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = isk.impute(y_pred)
        r2s[lags] = r2_score(y_test, y_pred)
    best_lag, best_r2 = sorted(r2s.items(), key=lambda x: x[1], reverse=True)[0]
    print(best_r2)
    print("--lags", best_lag, end=' ') 
    print(" ".join(sys.argv[1:]))
    print()