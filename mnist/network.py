#!/usr/bin/env python
import sys
sys.path.append('..')

import time
import argparse
import numpy as np
import pandas as pd

from keras.applications.inception_v3 import InceptionV3
from mnist import MNIST
from sklearn.model_selection import StratifiedKFold

def ingest(data):
    X, y = data.load_training()
    X = pd.DataFrame(X)
    y = pd.Series(y).to_frame()
    
if __name__=="__main__":
    t0 = time.time()

    mndata = MNIST('./data')

    X, y = ingestion(mndata)
    model = InceptionV3(weights='imagenet', include_top=False)
    net_feats = model.predict(X.iloc[0])
    print(net_feats.shape)
    print(time.time() - t0)
