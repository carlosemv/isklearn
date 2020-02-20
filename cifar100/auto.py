import numpy as np
import pandas as pd

import sys
#sys.setrecursionlimit(40000)
from datetime import timedelta
from random import randint
from autosklearn.classification import AutoSklearnClassifier
from sklearn.metrics import accuracy_score

X_train = np.load('data/cifar100/train_data.npy')
y_train = np.load('data/cifar100/train_labels.npy')
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train).values.ravel()

X_test = np.load('data/cifar100/test_data.npy')
y_test = np.load('data/cifar100/test_labels.npy')
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test).values.ravel()

time_limit = timedelta(hours=18, minutes=3, seconds=15)
cutoff = timedelta(minutes=15)
mem_limit = 32*1000
print("time limit = {}".format(time_limit))
print("cutoff = {}".format(cutoff))
print("memory limit = {}".format(mem_limit))

tmp_folder = "./autosklearn_tmp_"+str(randint(0,10000))
askl = AutoSklearnClassifier(time_left_for_this_task=int(time_limit.total_seconds()),
	per_run_time_limit=int(cutoff.total_seconds()),
	resampling_strategy='cv',ensemble_memory_limit=16000,
	resampling_strategy_arguments={'folds': 5},
	ml_memory_limit=mem_limit, tmp_folder=tmp_folder)
askl.fit(X_train, y_train)
askl.refit(X_train, y_train)
y_pred = askl.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("accuracy: {}\n".format(acc))

print(askl.show_models())
print(askl.sprint_statistics())
# with open('./mnist_autoskl.txt', 'w') as f:
# 	print(askl.show_models(), file=f)
# 	print(acc, file=f)
