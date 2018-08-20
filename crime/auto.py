import pandas as pd
from datetime import timedelta
from random import randint
from autosklearn.regression import AutoSklearnRegressor
from ingestion import Ingestion
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ts_file = "data/crime_districts_daily.csv"
ts_df = pd.read_csv(ts_file, index_col=0).sum(axis=1).to_frame()

train_df, test_df = train_test_split(ts_df, test_size=0.2, shuffle=False)

lags = 15
X_train = pd.DataFrame(Ingestion(lags=lags, freq='d').fit_transform(train_df)[0])
y_train = train_df[0][(6+lags):]

X_test = pd.DataFrame(Ingestion(lags=lags, freq='d').fit_transform(test_df)[0])
y_test = test_df[0][(6+lags):]

time_limit = 2*(5*60 + 56) * 60
print("time limit =", timedelta(seconds=time_limit))
tmp_folder = "./autosklearn_tmp_"+str(randint(0,10000))
askl = AutoSklearnRegressor(time_left_for_this_task=time_limit, per_run_time_limit=15*60,
	resampling_strategy='cv', resampling_strategy_arguments={'folds': 5},
	ml_memory_limit=8000, tmp_folder=tmp_folder)
askl.fit(X_train, y_train)
askl.refit(X_train, y_train)
y_pred = askl.predict(X_test)
acc = r2_score(y_test, y_pred)
print(acc)
# with open('./mnist_autoskl.txt', 'w') as f:
# 	print(askl.show_models(), file=f)
# 	print(acc, file=f)