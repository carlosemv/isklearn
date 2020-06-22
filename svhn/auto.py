import numpy as np
import pandas as pd
import sys
sys.setrecursionlimit(40000)

from datetime import timedelta
from random import randint
from autosklearn.classification import AutoSklearnClassifier
from sklearn.metrics import accuracy_score

X_train = np.load('/home/addaraujo2/isklearn/data/svhn/train_data.npy', allow_pickle=True, encoding='latin1')
y_train = np.load('/home/addaraujo2/isklearn/data/svhn/train_labels.npy', allow_pickle=True, encoding='latin1')
#X_train = pd.DataFrame(X_train)
#y_train = pd.Series(y_train).values.ravel()

X_test = np.load('/home/addaraujo2/isklearn/data/svhn/test_data.npy', allow_pickle=True, encoding='latin1')
y_test = np.load('/home/addaraujo2/isklearn/data/svhn/test_labels.npy', allow_pickle=True, encoding='latin1')
#X_test = pd.DataFrame(X_test)
#y_test = pd.Series(y_test).values.ravel()

#X_train = pd.read_csv('/home/addaraujo2/isklearn/data/svhn/X_train.csv', index_col=0).values
#y_train = pd.read_csv('/home/addaraujo2/isklearn/data/svhn/y_train.csv', index_col=0).values.ravel()
#X_test = pd.read_csv('/home/addaraujo2/isklearn/data/svhn/X_test.csv', index_col=0).values
#y_test = pd.read_csv('/home/addaraujo2/isklearn/data/svhn/y_test.csv', index_col=0).values.ravel()

#time_limit = timedelta(hours=12, minutes=3, seconds=15)
#cutoff = timedelta(minutes=15)
#mem_limit = 32*1000
#print("time limit = {}".format(time_limit))
#print("cutoff = {}".format(cutoff))
#print("memory limit = {}".format(mem_limit))

#tmp_folder = "./autosklearn_tmp_"+str(randint(0,10000))
#askl = AutoSklearnClassifier(time_left_for_this_task=int(time_limit.total_seconds()),#
#	per_run_time_limit=int(cutoff.total_seconds()),
#	resampling_strategy='cv',
#	resampling_strategy_arguments={'folds': 5},
#	ml_memory_limit=mem_limit, tmp_folder=tmp_folder)
#askl.fit(X_train, y_train)
#askl.refit(X_train, y_train)
#y_pred = askl.predict(X_test)

time_limit = ( 18 * 60 * 60 ) + 200
print("time limit =", timedelta(seconds=time_limit))
tmp_folder = "./autosklearn_tmp_"+str(randint(0,10000))
askl = AutoSklearnClassifier(time_left_for_this_task=time_limit, per_run_time_limit=15*60,
        resampling_strategy='cv', resampling_strategy_arguments={'folds': 5},
        ml_memory_limit=32000, tmp_folder=tmp_folder, ensemble_memory_limit=16000)
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
