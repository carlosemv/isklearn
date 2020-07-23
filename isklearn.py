import argparse
import numpy as np
import pandas as pd
import scipy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.decomposition import FastICA, PCA, DictionaryLearning, TruncatedSVD
from sklearn.feature_selection import SelectPercentile, SelectFromModel, RFE, \
    f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def _str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Selector(BaseEstimator, TransformerMixin):
    def __init__(self, conf, task):
        parser = argparse.ArgumentParser()
        parser.add_argument('--sel_score_classification', type=str)
        parser.add_argument('--sel_score_regression', type=str)
        parser.add_argument('--sel_model', type=str)
        parser.add_argument('--sel_percentile', type=int)
        parser.add_argument('--sel_threshold', type=str)
        args = parser.parse_known_args()[0]

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
                error = "Invalid Selector.score_func argument "+str(sel_score)
                raise ValueError(error)

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
            raise ValueError("Invalid Selector.conf argument: "+str(self.conf))

        self.selection_model = selection
        self.selection_model.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.selection_model.transform(X)


class Extractor(BaseEstimator, TransformerMixin):
    def __init__(self, conf, task):
        parser = argparse.ArgumentParser()
        parser.add_argument("--ext_components", type=float)
        parser.add_argument("--whiten", type=_str_to_bool)
        parser.add_argument("--svd_solver", type=str)
        parser.add_argument("--ica_algorithm", type=str)
        parser.add_argument("--ica_fun", type=str)
        parser.add_argument("--dl_fit_algorithm", type=str)
        parser.add_argument("--dl_transform_algorithm", type=str)
        args = parser.parse_known_args()[0]

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
            raise ValueError("Invalid Extractor.conf argument: "+str(self.conf))

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

class ISKLEARN:
    def __init__(self, task, sparse=False):
        if not (task == "classification" or task == "regression"):
            raise ValueError("Invalid ISKLEARN.task argument: "+str(task))
        if type(sparse) != bool:
            raise ValueError("Invalid ISKLEARN.sparse argument: "+str(sparse))

        self.task = task
        self.sparse = sparse

    def impute(self, X):
        if self.sparse and X.ndim > 1:
            X = SimpleImputer().fit_transform(X)
            return X

        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        max32 = np.finfo(np.float32).max
        min32 = np.finfo(np.float32).min

        for col in X.T:
            mean = np.nanmean(col)
            col[np.isnan(col)] = mean if not np.isnan(mean) else max32

            col[np.logical_or(col > max32, col == np.inf)] = max32
            col[np.logical_or(col < min32, col == -np.inf)] = min32

        return X


    def build_model(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--algorithm', type=str)
        args = parser.parse_known_args()[0]
        algorithm = args.algorithm

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
            C = 10**args.C
            obj = SVC if self.task == 'classification' else SVR
            model = obj(C=C, kernel=args.kernel, degree=degree,
                gamma=gamma, cache_size=600)
        elif algorithm == "MLP":
            parser.add_argument('--solver', type=str)
            parser.add_argument('--alpha', type=float)
            parser.add_argument('--mlp_learning_rate', type=str)
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

            if args.learning_rate_init != None:
                learning_rate_init = 10**args.learning_rate_init
            else:
                learning_rate_init = 0.001

            if args.mlp_learning_rate:
                mlp_learning_rate = args.mlp_learning_rate
            else:
                mlp_learning_rate = 'constant'

            obj = MLPClassifier if self.task == 'classification' else MLPRegressor
            model = obj(solver=args.solver, learning_rate=mlp_learning_rate,
                learning_rate_init=learning_rate_init, alpha=10**args.alpha,
                hidden_layer_sizes=hidden_layer_sizes, activation=args.activation)
        elif algorithm == "RandomForest":
            parser.add_argument('--criterion_classification', type=str)
            parser.add_argument('--criterion_regression', type=str)
            parser.add_argument('--max_features', type=float)
            parser.add_argument('--rf_estimators', type=int)
            parser.add_argument('--max_depth', type=str)
            parser.add_argument('--max_depth_value', type=int)
            parser.add_argument('--min_samples_leaf', type=float)
            args = parser.parse_known_args()[0]

            criterion = args.criterion_classification if self.task == 'classification' else \
                args.criterion_regression
            max_depth = args.max_depth_value if args.max_depth == 'value' else None
            min_samples_leaf = args.min_samples_leaf if args.min_samples_leaf > 0 else 1
            obj = RandomForestClassifier if self.task == 'classification' else RandomForestRegressor
            model = obj(criterion=criterion, max_features=args.max_features,
                    n_estimators=args.rf_estimators, max_depth=max_depth, 
                    min_samples_leaf=min_samples_leaf)
        elif algorithm == "KNeighbors":
            parser.add_argument('--n_neighbors', type=int)
            parser.add_argument('--weights', type=str)
            args = parser.parse_known_args()[0]

            obj = KNeighborsClassifier if self.task == 'classification' else KNeighborsRegressor
            model = obj(n_neighbors=args.n_neighbors, 
                weights=args.weights)
        elif algorithm == "DecisionTree":
            parser.add_argument('--criterion_classification', type=str)
            parser.add_argument('--criterion_regression', type=str)
            parser.add_argument('--max_features', type=float)
            parser.add_argument('--max_depth', type=str)
            parser.add_argument('--max_depth_value', type=int)
            parser.add_argument('--min_samples_leaf', type=float)
            args = parser.parse_known_args()[0]

            criterion = args.criterion_classification if self.task == 'classification' else \
                args.criterion_regression
            max_depth = args.max_depth_value if args.max_depth == 'value' else None
            min_samples_leaf = args.min_samples_leaf if args.min_samples_leaf > 0 else 1
            obj = DecisionTreeClassifier if self.task == 'classification' else DecisionTreeRegressor
            model = obj(criterion=criterion, max_features=args.max_features, 
                max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        elif algorithm == "AdaBoost":
            parser.add_argument('--ab_estimators', type=int)
            parser.add_argument('--ab_learning_rate', type=float)
            parser.add_argument('--ab_loss', type=str)
            args = parser.parse_known_args()[0]

            if self.task == 'classification':
                model = AdaBoostClassifier(n_estimators=args.ab_estimators,
                    learning_rate=args.ab_learning_rate)
            else:
                model = AdaBoostRegressor(n_estimators=args.ab_estimators,
                    learning_rate=args.ab_learning_rate, loss=args.ab_loss)
        elif algorithm == "LinearRegression":
            model = LinearRegression()
        elif algorithm == "LogisticRegression":
            parser.add_argument('--lr_C', type=float)
            parser.add_argument('--lr_solver', type=str)
            parser.add_argument('--multi_class', type=str)
            parser.add_argument('--max_iter', type=int, default=100)
            parser.add_argument('--lr_penalty', type=str, default='l2')
            parser.add_argument('--lr_dual', type=_str_to_bool, default=False)
            args = parser.parse_known_args()[0]

            model = LogisticRegression(C=10**args.lr_C, solver=args.lr_solver,
                multi_class=args.multi_class, max_iter=args.max_iter,
                penalty=args.lr_penalty, dual=args.lr_dual)

        return model

    def preprocess(self, X_train, y_train, X_test, y_test):
        parser = argparse.ArgumentParser()
        parser.add_argument('--f_eng1', type=str)
        parser.add_argument('--f_eng2', type=str)
        parser.add_argument('--pre_scaling', type=_str_to_bool)
        parser.add_argument('--extraction', type=str)
        parser.add_argument('--selection', type=str)
        parser.add_argument('--scaling', type=_str_to_bool)
        args = parser.parse_known_args()[0]

        if args.pre_scaling:
            with_mean = not self.sparse
            scaler = StandardScaler(with_mean=with_mean)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train = self.impute(X_train)
            X_test = self.impute(X_test)

        f_eng1, f_eng2 = args.f_eng1, args.f_eng2
         if (f_eng1 == f_eng2) and (f_eng1 != "none"):
             err_details = "f_eng1 = {}, f_eng2 = {}".format(f_eng1, f_eng2)
            raise ValueError("Invalid feature engineering parameters: " + err_details)

        for f_eng in (f_eng1, f_eng2):
            if f_eng == "Selection" and X_train.shape[1] > 1:
                selector = Selector(args.selection, self.task)
                X_train = selector.fit_transform(X_train, y_train)
                X_test = selector.transform(X_test)
            elif f_eng == "Extraction" and X_train.shape[1] > 1:
                extractor = Extractor(args.extraction, self.task) if not self.sparse else \
                    Extractor("TruncatedSVD", self.task)
                try:
                    X_train = extractor.fit_transform(X_train, y_train)
                except ValueError as e:
                    if 'array must not contain infs or NaNs' in e.args[0]:
                        X_train.drop(X_train.columns, axis=1, inplace=True)
                        X_test.drop(X_test.columns, axis=1, inplace=True)
                        break
                        # raise ValueError("Bug in scikit-learn: "
                            # +"https://github.com/scikit-learn/scikit-learn/pull/2738")
                    else:
                        raise e

                X_test = extractor.transform(X_test)

                X_train = self.impute(X_train)
                X_test = self.impute(X_test)

        if args.scaling:
            with_mean = not self.sparse
            scaler = StandardScaler(with_mean=with_mean)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train = self.impute(X_train)
            X_test = self.impute(X_test)

        return X_train, y_train, X_test, y_test

    def validation(self, X, y, cv="StratifiedKFold"):
        if cv == "TimeSeriesSplit":
            cv = TimeSeriesSplit
        elif cv == "StratifiedKFold":
            cv = StratifiedKFold
        else:
            raise ValueError("Invalid validation.cv argument: "+str(cv))

        fail = -2**32
        scores = []
        for train_idx, test_idx in cv(n_splits=5).split(X, y):
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx].values.ravel(), y.iloc[test_idx].values.ravel()
            else:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

            X_train, y_train, X_test, y_test = self.preprocess(X_train, y_train, X_test, y_test)

            if X_train.shape[1] == 0:
                scores.append(fail)
                continue

            scorer = accuracy_score if self.task == 'classification' else r2_score
            model = self.build_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = self.impute(y_pred)
            scores.append(scorer(y_test, y_pred))
        return scores
