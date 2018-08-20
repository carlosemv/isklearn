import pandas as pd
from datetime import timedelta
from random import randint
from autosklearn.classification import AutoSklearnClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

path = "/home/cemvieira/isklearn/data/lmrd"
n_tokens = 89527
X_train, y_train = load_svmlight_file(path+"/train/labeledBow.feat", n_features=n_tokens)
X_train = TfidfTransformer().fit_transform(X_train)
y_train[y_train <= 4] = 0
y_train[y_train >= 7] = 1

X_test, y_test = load_svmlight_file(path+"/test/labeledBow.feat", n_features=n_tokens)
X_test = TfidfTransformer().fit_transform(X_test)
y_test[y_test <= 4] = 0
y_test[y_test >= 7] = 1

time_limit = 2*(9*60 + 10) * 60
print("time limit =", timedelta(seconds=time_limit))
tmp_folder = "./autosklearn_tmp_"+str(randint(0,10000))
askl = AutoSklearnClassifier(time_left_for_this_task=time_limit, per_run_time_limit=15*60,
	resampling_strategy='cv', resampling_strategy_arguments={'folds': 5},
	ml_memory_limit=8000, tmp_folder=tmp_folder)
askl.fit(X_train, y_train)
askl.refit(X_train, y_train)
y_pred = askl.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)