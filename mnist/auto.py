import pandas as pd
from datetime import timedelta
from random import randint
from autosklearn.classification import AutoSklearnClassifier
from mnist import MNIST
from sklearn.metrics import accuracy_score

mndata = MNIST('./data')
X_train, y_train = mndata.load_training()
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train).values.ravel()

X_test, y_test = mndata.load_testing()
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test).values.ravel()

time_limit = 2*(8*60 + 36) * 60
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
# with open('./mnist_autoskl.txt', 'w') as f:
# 	print(askl.show_models(), file=f)
# 	print(acc, file=f)