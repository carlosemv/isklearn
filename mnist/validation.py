#!/usr/bin/env python
import sys
import time
import argparse
import numpy as np
import pandas as pd
from mnist import MNIST

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectPercentile, \
    SelectFromModel, RFE, f_classif, mutual_info_classif, chi2
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class Selector(BaseEstimator, TransformerMixin):
    def __init__(self, conf):
        parser = argparse.ArgumentParser()
        parser.add_argument('--sel_score', type=str)
        parser.add_argument('--sel_percentile', type=int)
        parser.add_argument('--sel_threshold', type=str)

        args = parser.parse_known_args()[0]
        if (conf == "SelectPercentile"):
            if (args.sel_score == "f_classif"):
                score_func = f_classif
            elif (args.sel_score == "chi2"):
                score_func = chi2
            elif (args.sel_score == "mutual_info_classif"):
                score_func = mutual_info_classif
            else:
                raise ValueError("Invalid Selector.score_func argument")
            self.score_func = score_func

        if (conf == "SelectFromModel"):
            self.threshold = args.sel_threshold
        else:
            self.percentile = args.sel_percentile
        self.conf = conf

    def fit(self, X, y=None):
        if self.conf=="RFE":
            n_features = int(X.shape[1] * (self.percentile/100.0))
            sel = RFE(RandomForestClassifier(), n_features_to_select=n_features)
        elif self.conf=="SelectFromModel":
            sel = SelectFromModel(RandomForestClassifier(), threshold=self.threshold)
        elif self.conf=="SelectPercentile":
            sel = SelectPercentile(score_func=self.score_func,
                percentile=self.percentile)
        else:
            raise ValueError("Invalid Selector.conf argument: "+str(self.conf))

        self.selection_model = sel
        self.selection_model.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.selection_model.transform(X)


# class Extractor(BaseEstimator, TransformerMixin)
    

def build_model(algorithm):
    model = None
    parser = argparse.ArgumentParser()

    if algorithm == "SVM":
        parser.add_argument('--C', type=float)
        parser.add_argument('--kernel', type=str)
        parser.add_argument('--degree', type=int)
        parser.add_argument('--gamma', type=float)
        args = parser.parse_known_args()[0]

        degree = args.degree if args.kernel == "poly" else 0

        gamma = 10**args.gamma if args.gamma != None else 'auto'
        C = (10**5) - (10**args.C) + (10**-3)
        model = SVC(C=C, kernel=args.kernel, degree=degree,
            gamma=gamma, cache_size=600)
    elif algorithm == "MLP":
        parser.add_argument('--solver', type=str)
        parser.add_argument('--alpha', type=float)
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

        learning_rate_init = 10**args.learning_rate_init if args.learning_rate_init != None else 0.001
        model = MLPClassifier(solver=args.solver, learning_rate_init=learning_rate_init,
            alpha=10**args.alpha, hidden_layer_sizes=hidden_layer_sizes, activation=args.activation)
    elif algorithm == "RandomForest":
        parser.add_argument('--max_features', type=float)
        parser.add_argument('--rf_estimators', type=int)
        parser.add_argument('--max_depth', type=str)
        parser.add_argument('--max_depth_value', type=int)
        parser.add_argument('--min_samples_leaf', type=float)
        args = parser.parse_known_args()[0]

        max_depth = args.max_depth_value if args.max_depth == 'value' else None

        model = RandomForestClassifier(max_features=args.max_features, n_estimators=args.rf_estimators,
            max_depth=max_depth, min_samples_leaf=args.min_samples_leaf, n_jobs=-1)
    elif algorithm == "KNeighbors":
        parser.add_argument('--n_neighbors', type=int)
        parser.add_argument('--weights', type=str)
        args = parser.parse_known_args()[0]

        model = KNeighborsClassifier(n_neighbors=args.n_neighbors, weights=args.weights, n_jobs=-1)

    return model

if __name__=="__main__":
    mndata = MNIST('./data')

    X_train, y_train = mndata.load_training()
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train).to_frame()

    X_test, y_test = mndata.load_testing()
    X_test = pd.DataFrame(X_test)
    y_test = pd.Series(y_test).to_frame()

    parser = argparse.ArgumentParser()
    parser.add_argument('--f_eng1', type=str)
    parser.add_argument('--f_eng2', type=str)
    parser.add_argument('--pre_scaling', type=bool)
    parser.add_argument('--extraction', type=str)
    parser.add_argument('--selection', type=str)
    parser.add_argument('--scaling', type=bool)
    parser.add_argument('--algorithm', type=str)

    args = parser.parse_known_args()[0]

    y_train, y_test = y_train.values.ravel(), y_test.values.ravel()

    if args.pre_scaling:            
        scaler = MinMaxScaler(feature_range=(1, 2))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    f_eng1, f_eng2 = args.f_eng1, args.f_eng2
    if (f_eng1 == f_eng2) and (f_eng1 != "None"):
        err_details = "f_eng1 = " + f_eng1 + ", f_eng2 = " + f_eng2
        raise ValueError("Invalid feature engineering parameters: " + err_details)

    for f_eng in (f_eng1, f_eng2):
        if f_eng == "Selection":
            selector = Selector(args.selection)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test, y_test)
        # elif f_eng == "Extraction":
            # extractor = Extractor(args.extraction)
            # X_train = extractor.fit_transform(X_train)
            # X_test = extractor.transform(X_test)

    if args.scaling:
        scaler = MinMaxScaler(feature_range=(1, 2))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = build_model(args.algorithm)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(" ".join(sys.argv))