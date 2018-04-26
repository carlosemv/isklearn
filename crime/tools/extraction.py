import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

def ts_train_test_split(y, test_size):
    if test_size<1:
        test_size = int(np.floor(test_size*len(y)))
    return y[:-test_size], y[-test_size:]

def lag(y, L, slice_lags=None):
    y = np.ravel(y)
    X = np.matrix(np.empty(shape=(len(y)-L,L)))
    for l in range(0,L):
        X[:,l] = y[L-1-l:len(y)-l-1].reshape(-1,1)
    if slice_lags is not None:
        slice_lags = [x-1 for x in slice_lags]
        return X[:,slice_lags]
    else:
        return X

class AutoRegressives(BaseEstimator, TransformerMixin):
    def __init__(self, lags):
        self.lags = lags
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if type(self.lags) is list:
            return lag(X, L=self.lags[-1], slice_lags=self.lags)
        else:
            return lag(X, L=self.lags)

class Seasonals(BaseEstimator, TransformerMixin):
    def __init__(self, lags):
        self.lags = lags
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if type(self.lags) is list:
            return lag(seasonal_decompose(X, two_sided=False).seasonal[6:], L=self.lags[-1], slice_lags=self.lags)
        else:
            return lag(seasonal_decompose(X, two_sided=False).seasonal[6:], L=self.lags)

class Trends(BaseEstimator, TransformerMixin):
    def __init__(self, lags):
        self.lags = lags
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if type(self.lags) is list:
            return lag(seasonal_decompose(X, two_sided=False).trend[6:], L=self.lags[-1], slice_lags=self.lags)
        else:
            return lag(seasonal_decompose(X, two_sided=False).trend[6:], L=self.lags)
