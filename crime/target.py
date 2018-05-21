#!/usr/bin/env python
import sys
sys.path.append('..')

import time
import argparse
import numpy as np
import pandas as pd
from isklearn.isklearn import ISKLEARN

from ingestion import Ingestion

def ingestion(df, series):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lags', type=int)
    args = parser.parse_known_args()[0]

    X = pd.DataFrame(Ingestion(lags=args.lags, freq='d').fit_transform(df)[series])
    y = df[series][(6+args.lags):].to_frame()

    return X, y

if __name__=="__main__":
    print(sys.argv)
    t0 = time.time()

    ts_file = "data/crime_districts_daily.csv"
    ts_df = pd.read_csv(ts_file, index_col=0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str)
    parser.add_argument('--config_id', type=str)
    parser.add_argument('--task', type=str)
    args = parser.parse_known_args()[0]

    spots = [x for x in args.instance.split(',')]
    config_id = args.config_id
    task = args.task

    r2s = []
    isk = ISKLEARN(task)
    for series in spots:
        X, y = ingestion(ts_df, series)
        r2_scores = isk.validation(X, y, cv="TimeSeriesSplit")
        result = np.mean(r2_scores)
        r2s.append(result)

    print(-1*np.mean(r2s), time.time()-t0)