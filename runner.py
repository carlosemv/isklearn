#!/usr/bin/env python
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, Adaboost
from sklearn.neighbors import KNeighborsRegressor

class Selector(BaseEstimator, TransformerMixin)
class Extractor(BaseEstimator, TransformerMixin)

def ingestion(df, series):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--lags', type=int)
    # X, y = ext.AutoRegressives(lags=self.lags).fit_transform(y), y[self.lags:]
    # return X, y

def build_model(algorithm):
    model = None
    parser = argparse.ArgumentParser()

    # todo: n_jobs = -1 when possible
    if algorithm == "SVM":
        parser.add_argument('--C', type=float)
        parser.add_argument('--epsilon', type=float)
        parser.add_argument('--kernel', type=str)
        parser.add_argument('--gamma', type=float)
        args = parser.parse_known_args()[0]

        model = SVR(C=args.C, epsilon=args.epsilon, kernel=args.kernel, gamma=args.gamma)
    elif algorithm == "MLP":
        parser.add_argument('--solver', type=str)
        parser.add_argument('--learning_rate_init', type=float)
        parser.add_argument('--hidden_layers', type=int)
        parser.add_argument('--neurons1', type=int)
        parser.add_argument('--neurons2', type=int)
        parser.add_argument('--neurons3', type=int)
        parser.add_argument('--activation', type=str)
        args = parser.parse_known_args()[0]

        if args.hidden_layers == 1:
            hidden_layer_sizes = (args.neurons1,)
        elif args.hidden_layers == 2:
            hidden_layer_sizes = (args.neurons1,args.neurons2)
        else:
            hidden_layer_sizes = (args.neurons1,args.neurons2,args.neurons3)

        model = MLPRegressor(solver=args.solver, learning_rate_init=args.learning_rate_init,
            hidden_layer_sizes=hidden_layer_sizes, activation=args.activation)
    elif algorithm == "RandomForest":
        parser.add_argument('--max_features', type=float)
        parser.add_argument('--rf_estimators', type=int)
        parser.add_argument('--max_depth', type=str)
        parser.add_argument('--max_depth_value', type=int)
        parser.add_argument('--min_samples_leaf', type=int)
        args = parser.parse_known_args()[0]

        max_depth = args.max_depth_value if args.max_depth == 'value' else None

        model = RandomForestRegressor(max_features=args.max_features, n_estimators=args.rf_estimators,
            max_depth=max_depth, min_samples_leaf=args.min_samples_leaf)
    elif algorithm == "KNeighbors":
        parser.add_argument('--n_neighbors', type=int)
        parser.add_argument('--weights', type=str)
        args = parser.parse_known_args()[0]

        model = KNeighborsRegressor(n_neighbors=args.n_neighbors, weights=args.weights)
    elif algorithm == "Adaboost":
        parser.add_argument('--n_estimators', type=int)
        parser.add_argument('--learning_rate', type=float)
        parser.add_argument('--loss', type=str)
        args = parser.parse_known_args()[0]

        model = AdaBoostRegressor(n_estimators=args.n_estimators,
            learning_rate=args.learning_rate, loss=args.loss)

    return model

def validation(X, y):
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_eng1', type=str)
    parser.add_argument('--f_eng2', type=str)
    parser.add_argument('--pre_scaling', type=bool)
    parser.add_argument('--extraction', type=str)
    parser.add_argument('--selection', type=str)
    parser.add_argument('--scaling', type=bool)
    parser.add_argument('--algorithm', type=str)

    args = parser.parse_known_args()[0]

    r2_scores = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if args.pre_scaling:            
            scaler = MinMaxScaler(feature_range=(1, 2))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # todo: check for f_eng constraints (forbidden configurations)
        for f_eng in (args.f_eng1, args.f_eng2):
            if f_eng == "Selection":
                selector = Selector(args.selection)
                X_train = selector.fit_transform(X_train, y_train)
                X_test = selector.transform(X_test, y_test)
            elif f_eng == "Extraction":
                extractor = Extractor(args.extraction)
                X_train = extractor.fit_transform(X_train)
                X_test = extractor.transform(X_test)

        if args.scaling:
            scaler = MinMaxScaler(feature_range=(1, 2))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = build_model(args.algorithm)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
    return r2_scores

if __name__=="__main__":
    t0 = time.time()

    ts_file = "data/dfts_18_nocategory_daily.csv"
    ts_df = pd.read_csv(ts_file, index_col='i')

    # todo: check for memory file before loading
    mem_file = "data/results.csv"
    mem_types = {'spot':str,'config_id':str,'result':float}
    mem_df = pd.read_csv(mem_file, dtype=mem_types).set_index(['spot','config_id'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str)
    parser.add_argument('--config_id', type=str)

    args = parser.parse_known_args()[0]

    spots = [x for x in args.instance.split(',')]
    config_id = args.config_id

    r2s = []
    new_value = False
    for series in spots:
        if (series, config_id) in mem_df.index:
            r2s.append(mem_df.loc[series, config_id]['result'])
        else:
            X, y = ingestion(ts_df, series)
            r2_scores = validation(X, y)
            result = np.mean(r2_scores)
            r2s.append(result)
            mem_df.loc[(series, config_id), 'result'] = result
            new_value = True

    if new_value:
        mem_df.to_csv(mem_file)

    print(np.mean(r2s), time.time()-t0)