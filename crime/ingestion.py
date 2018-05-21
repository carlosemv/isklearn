import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion


def ts_train_test_split(y, test_size):
    if test_size<1:
        test_size = int(np.floor(test_size*len(y)))
    return y[:-test_size], y[-test_size:]

def lag(y, L, slice_lags=None):
    if y.index.freq=='d':
        freq_offset = 6
    elif y.index.freq=='w':
        freq_offset = 52
    y = np.ravel(y)
    X = np.matrix(np.empty(shape=(len(y)-L,L)))
    for l in range(0,L):
        X[:,l] = y[L-1-l:len(y)-l-1].reshape(-1,1)
    if slice_lags is not None:
        slice_lags = [x-1 for x in slice_lags]
        return X[:,slice_lags]
    else:
        return X[freq_offset:]

class AutoRegressives(BaseEstimator, TransformerMixin):
    def __init__(self, lags, freq='d'):
        self.lags = lags
        self.freq = freq
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.asfreq('d').asfreq(self.freq)
        if type(self.lags) is list:
            return lag(X, L=self.lags[-1], slice_lags=self.lags+6)
        else:
            return lag(X, L=self.lags)

class Seasonals(BaseEstimator, TransformerMixin):
    def __init__(self, lags, freq='d'):
        self.lags = lags
        self.freq = freq
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.asfreq('d').asfreq(self.freq)
        if type(self.lags) is list:
            return lag(seasonal_decompose(X, two_sided=False).seasonal, L=self.lags[-1], slice_lags=self.lags+6)
        else:
            return lag(seasonal_decompose(X, two_sided=False).seasonal, L=self.lags)

class Trends(BaseEstimator, TransformerMixin):
    def __init__(self, lags, freq='d'):
        self.lags = lags
        self.freq = freq
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.asfreq('d').asfreq(self.freq)
        if type(self.lags) is list:
            return lag(seasonal_decompose(X, two_sided=False).trend, L=self.lags[-1], slice_lags=self.lags+6)
        else:
            return lag(seasonal_decompose(X, two_sided=False).trend, L=self.lags)


class Ingestion(BaseEstimator, TransformerMixin):
    """
    Input: df_ts (dataframe of time series indexed by location id)
    Output: s_feat (series of features matrix indexed by df_ts id)
    """
    def __init__(self, lags=5, freq='d'):
        self.lags = lags
        self.freq = freq
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        ext_series, ext_index = [], []
        for ts in X.columns:
            y = X[ts]
            ext_series.append(FeatureUnion([
                ('ar',AutoRegressives(lags=self.lags,freq=self.freq)),
                ('sz',Seasonals(lags=self.lags,freq=self.freq)),
                ('tr',Trends(lags=self.lags,freq=self.freq))
            ]).fit_transform(y))
            ext_index.append(ts)
        r = pd.Series(ext_series, index=ext_index)
        return r

# df_ts = pd.read_csv('df_ts.csv', index_col=0)
# # y = df_ts[df_ts.columns[3]]
#
# print(df_ts.shape,df_ts.columns)
#
# outi = Ingestion(lags=5).fit_transform(df_ts)
# print(outi.shape)
# print(outi.head())
