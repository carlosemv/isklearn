import os
import pandas as pd
from datetime import timedelta
from random import randint
from autosklearn.regression import AutoSklearnRegressor
#from ingestion import Ingestion
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datapath = 'data/crime_data/'

X = pd.read_csv(os.path.join(datapath,'natal_X.csv'), index_col=[0,1])
y = pd.read_csv(os.path.join(datapath,'natal_y.csv'), index_col=[0,1])

#  Scaling features
Xs = MinMaxScaler().fit_transform(X)
ys = MinMaxScaler().fit_transform(y)
#  Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.2, shuffle=False)


time_limit = 2*(5*60 + 56) * 60
print("time limit =", timedelta(seconds=time_limit))
tmp_folder = "./autosklearn_tmp_"+str(randint(0,10000))
askl = AutoSklearnRegressor(time_left_for_this_task=time_limit, per_run_time_limit=15*60,
	resampling_strategy='cv', resampling_strategy_arguments={'folds': 5},
	ml_memory_limit=16000, tmp_folder=tmp_folder, ensemble_memory_limit=16000)
askl.fit(X_train, y_train)
askl.refit(X_train, y_train)
y_pred = askl.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)

