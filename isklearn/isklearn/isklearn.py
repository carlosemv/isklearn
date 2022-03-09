import argparse
import numpy as np
import pandas as pd

from isklearn.utils import _str_to_bool, _parse_args, ArgumentException
from isklearn.preprocessing import Selector, Extractor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class ISKLEARN:
    def __init__(self, task, sparse=False):
        if task not in ("classification", "regression"):
            raise ArgumentException("ISKLEARN.task", task)
        if type(sparse) != bool:
            raise ArgumentException("ISKLEARN.sparse", sparse)

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
        algorithm = _parse_args({'algorithm':
            {'type': str}}).algorithm
        model = None

        if algorithm == "SVM":
            args = _parse_args({
                'C': {'type': float},
                'kernel': {'type': str},
                'degree': {'type': int},
                'gamma': {'type': float},
            })

            degree = args.degree if args.kernel == "poly" else 0

            gamma = 10**args.gamma if args.gamma != None else 'auto'
            C = 10**args.C
            obj = SVC if self.task == 'classification' else SVR
            model = obj(C=C, kernel=args.kernel, degree=degree,
                gamma=gamma, cache_size=600)
        elif algorithm == "MLP":
            args = _parse_args({
                'solver': {'type': str},
                'alpha': {'type': float},
                'mlp_learning_rate': {'type': str},
                'learning_rate_init': {'type': float},
                'hidden_layers': {'type': int},
                'neurons1': {'type': int},
                'neurons2': {'type': int},
                'neurons3': {'type': int},
                'activation': {'type': str},
            })

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
            args = _parse_args({
                'criterion_classification': {'type': str},
                'criterion_regression': {'type': str},
                'max_features': {'type': float},
                'rf_estimators': {'type': int},
                'max_depth': {'type': str},
                'max_depth_value': {'type': int},
                'min_samples_leaf': {'type': float},
            })

            criterion = args.criterion_classification if self.task == 'classification' else \
                args.criterion_regression
            max_depth = args.max_depth_value if args.max_depth == 'value' else None
            min_samples_leaf = args.min_samples_leaf if args.min_samples_leaf > 0 else 1
            obj = RandomForestClassifier if self.task == 'classification' else RandomForestRegressor
            model = obj(criterion=criterion, max_features=args.max_features,
                    n_estimators=args.rf_estimators, max_depth=max_depth, 
                    min_samples_leaf=min_samples_leaf)
        elif algorithm == "KNeighbors":
            args = _parse_args({
                'n_neighbors': {'type': int},
                'weights': {'type': str}
            })

            obj = KNeighborsClassifier if self.task == 'classification' else KNeighborsRegressor
            model = obj(n_neighbors=args.n_neighbors, 
                weights=args.weights)
        elif algorithm == "DecisionTree":
            args = _parse_args({
                'criterion_classification': {'type': str},
                'criterion_regression': {'type': str},
                'max_features': {'type': float},
                'max_depth': {'type': str},
                'max_depth_value': {'type': int},
                'min_samples_leaf': {'type': float},
            })

            criterion = args.criterion_classification if self.task == 'classification' else \
                args.criterion_regression
            max_depth = args.max_depth_value if args.max_depth == 'value' else None
            min_samples_leaf = args.min_samples_leaf if args.min_samples_leaf > 0 else 1
            obj = DecisionTreeClassifier if self.task == 'classification' else DecisionTreeRegressor
            model = obj(criterion=criterion, max_features=args.max_features, 
                max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        elif algorithm == "AdaBoost":
            args = _parse_args({
                'ab_estimators': {'type': int},
                'ab_learning_rate': {'type': float},
                'ab_loss': {'type': str},
            })

            if self.task == 'classification':
                model = AdaBoostClassifier(n_estimators=args.ab_estimators,
                    learning_rate=args.ab_learning_rate)
            else:
                model = AdaBoostRegressor(n_estimators=args.ab_estimators,
                    learning_rate=args.ab_learning_rate, loss=args.ab_loss)
        elif algorithm == "LinearRegression":
            model = LinearRegression()
        elif algorithm == "LogisticRegression":
            args = _parse_args({
                'lr_C': {'type': float},
                'lr_solver': {'type': str},
                'multi_class': {'type': str},
                'max_iter': {'type': int, 'default': 100},
                'lr_penalty': {'type': str, 'default': 'l2'},
                'lr_dual': {'type': _str_to_bool, 'default': False},
            })

            model = LogisticRegression(C=10**args.lr_C, solver=args.lr_solver,
                multi_class=args.multi_class, max_iter=args.max_iter,
                penalty=args.lr_penalty, dual=args.lr_dual)
        else:
            raise ArgumentException("--algorithm", algorithm)

        return model

    def preprocess(self, X_train, y_train, X_test, y_test):
        args = _parse_args({
            'f_eng1': {'type': str},
            'f_eng2': {'type': str},
            'pre_scaling': {'type': _str_to_bool},
            'extraction': {'type': str},
            'selection': {'type': str},
            'scaling': {'type': _str_to_bool},
        })

        if args.pre_scaling:
            with_mean = not self.sparse
            scaler = StandardScaler(with_mean=with_mean)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train = self.impute(X_train)
            X_test = self.impute(X_test)

        f_eng1, f_eng2 = args.f_eng1, args.f_eng2
        if f_eng1 == f_eng2 and f_eng1 not in ("None", "none", None):
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
            raise ArgumentException("ISKLEARN.validation.cv", cv)

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
