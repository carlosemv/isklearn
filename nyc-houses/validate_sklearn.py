import os
import time
import pandas as pd
import fire

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
#
# import features


def run_default_sklearn(features_folder, label):
    print('Running...')
    t0 = time.time()

    # Part 1
    #  Load features
    X = pd.read_csv(os.path.join(features_folder, label+'_X.csv'), index_col=0)
    y = pd.read_csv(os.path.join(features_folder, label+'_y.csv'), index_col=0)
    print('\n\n\n',X.head(),'\n\n',y.head(),'\n\n')
    print(X.shape)
    print(time.time()-t0, ' seconds to generate features (X,y)')


    t0 = time.time()

    # Part 2
    # #  Scaling features
    # Xs = MinMaxScaler().fit_transform(X)
    # ys = MinMaxScaler().fit_transform(y)

    #  Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state =0)

    #  Train models with _train data (first 80%)
    rf = RandomForestRegressor().fit(X_train, y_train)
    mlp = MLPRegressor().fit(X_train, y_train)
    svm = SVR().fit(X_train, y_train)
    knn = KNeighborsRegressor().fit(X_train, y_train)
    ab = AdaBoostRegressor().fit(X_train, y_train)
    dt = DecisionTreeRegressor().fit(X_train, y_train)
    li = LinearRegression().fit(X_train, y_train)

    print(time.time()-t0, ' seconds to train default sklearn models')

    # Test models with _test data (last 20%)
    r2 = {}
    r2['rf'] = r2_score(y_test, rf.predict(X_test))
    r2['mlp'] = r2_score(y_test, mlp.predict(X_test))
    r2['svm'] = r2_score(y_test, svm.predict(X_test))
    r2['knn'] = r2_score(y_test, knn.predict(X_test))
    r2['ab'] = r2_score(y_test, ab.predict(X_test))
    r2['dt'] = r2_score(y_test, dt.predict(X_test))
    r2['li'] = r2_score(y_test, li.predict(X_test))

    print('\n\nR2 = ',r2)
    return 0


"""
Example:
    python3 validate_sklearn.py --features_folder=/path/to/features --label=boston
"""
if __name__=='__main__':
    fire.Fire(run_default_sklearn)

