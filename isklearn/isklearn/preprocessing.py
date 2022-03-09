import argparse
import numpy as np

from isklearn.utils import _str_to_bool, _parse_args, ArgumentException
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA, PCA, DictionaryLearning, TruncatedSVD
from sklearn.feature_selection import SelectPercentile, SelectFromModel, RFE, \
    f_classif, f_regression, mutual_info_classif, mutual_info_regression

from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class Selector(BaseEstimator, TransformerMixin):
    def __init__(self, conf, task):
        args = _parse_args({
            'sel_score_classification': {'type': str},
            'sel_score_regression': {'type': str},
            'sel_model': {'type': str},
            'sel_percentile': {'type': int},
            'sel_threshold': {'type': str},
        })

        self.task = task

        if conf == "SelectPercentile":
            sel_score = args.sel_score_classification if task == 'classification' \
                else args.sel_score_regression
            selection_score_map = {'f_regression': f_regression, 
                'mutual_info_regression': mutual_info_regression,
                'mutual_info_classif': mutual_info_classif,
                'f_classif': f_classif}
            if sel_score in selection_score_map:
                self.score_func = selection_score_map[sel_score]
            else:
                raise ArgumentException("Selector.score_func", sel_score)

        if conf == "SelectFromModel":
            self.threshold = args.sel_threshold
        else:
            self.percentile = args.sel_percentile

        if conf in ("SelectFromModel", "RFE"):
            self.sel_model = args.sel_model

        self.conf = conf

    def fit(self, X, y=None):
        sel_model_map = {('RandomForest', 'classification'): RandomForestClassifier(),
                        ('RandomForest', 'regression'): RandomForestRegressor(),
                        ('SVM', 'classification'): SVC(kernel='linear'),
                        ('SVM', 'regression'): SVR(kernel='linear'),
                        ('DecisionTree', 'classification'): DecisionTreeClassifier(),
                        ('DecisionTree', 'regression'): DecisionTreeRegressor()}
        if self.conf=="RFE":
            n_features = max(1, int(X.shape[1] * (self.percentile/100.0)))
            selector_model = sel_model_map[(self.sel_model, self.task)]
            selection = RFE(selector_model, n_features_to_select=n_features)
        elif self.conf=="SelectFromModel":
            selector_model = sel_model_map[(self.sel_model, self.task)]
            selection = SelectFromModel(selector_model, threshold=self.threshold)
        elif self.conf=="SelectPercentile":
            if int(X.shape[1] * (self.percentile/100.0)) == 0:
                self.percentile = 100*max(int(1. / X.shape[1]), 1)
            selection = SelectPercentile(score_func=self.score_func,
                percentile=self.percentile)
        else:
            raise ArgumentException("Selector.conf", self.conf)

        self.selection_model = selection
        self.selection_model.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.selection_model.transform(X)


class Extractor(BaseEstimator, TransformerMixin):
    def __init__(self, conf, task):
        args = _parse_args({
            'ext_components': {'type': float},
            'whiten': {'type': _str_to_bool},
            'svd_solver': {'type': str},
            'ica_algorithm': {'type': str},
            'ica_fun': {'type': str},
            'dl_fit_algorithm': {'type': str},
            'dl_transform_algorithm': {'type': str},
        })

        self.task = task
        self.fitted = False

        self.components = args.ext_components
        if conf == "PCA":
            self.whiten = args.whiten
            self.svd_solver = args.svd_solver
        elif conf == "FastICA":
            self.ica_algorithm = args.ica_algorithm
            self.ica_fun = args.ica_fun
        elif conf == "DictionaryLearning":
            self.dl_fit_algorithm = args.dl_fit_algorithm
            self.dl_transform_algorithm = args.dl_transform_algorithm

        self.conf = conf

    def fit(self, X, y=None):
        n_components = int(self.components * X.shape[1])
        if not n_components:
            n_components = 1
        if self.conf == "PCA":
            n_components = min(X.shape[0]-1, n_components)
            extraction = PCA(n_components=n_components, whiten=self.whiten,
                svd_solver=self.svd_solver)
        elif self.conf == "FastICA":
            extraction = FastICA(n_components=n_components,
                algorithm=self.ica_algorithm, fun=self.ica_fun)
        elif self.conf == "DictionaryLearning":
            extraction = DictionaryLearning(n_components=n_components,
                fit_algorithm=self.dl_fit_algorithm,
                transform_algorithm=self.dl_transform_algorithm)
        elif self.conf == "TruncatedSVD":
            n_components = min(n_components, (X.shape[0]-1))
            extraction = TruncatedSVD(n_components=n_components, algorithm='arpack')
        else:
            raise ArgumentException("Extractor.conf", self.conf)

        self.extraction_model = extraction
        try:
            self.extraction_model.fit(X, y)
        except np.linalg.LinAlgError:
            self.fitted = False
        else:
            self.fitted = True
        return self

    def transform(self, X, y=None):
        return self.extraction_model.transform(X) if self.fitted else X