import numpy as np
import pandas as pd

def ingestion():
    X_train = np.load('data/svhn/train_data.npy')
    y_train = np.load('data/svhn/train_labels.npy')
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train).values.ravel()

    X_test = np.load('data/svhn/test_data.npy')
    y_test = np.load('data/svhn/test_labels.npy')
    X_test = pd.DataFrame(X_test)
    y_test = pd.Series(y_test).values.ravel()

    return X_train, X_test, y_train, y_test