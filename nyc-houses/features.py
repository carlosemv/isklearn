import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from scipy import stats

#Reading Dataset
df = pd.read_csv('/Users/adelsondias/Documents/Repos/isklearn/nyc-houses/nyc-rolling-sales.csv',
            index_col=0)
df.head()

#Dropping column as it is empty
del df['EASE-MENT']
del df['SALE DATE']

df = df.drop_duplicates(df.columns, keep='last')

df['TAX CLASS AT TIME OF SALE'] = df['TAX CLASS AT TIME OF SALE'].astype('category')
df['TAX CLASS AT PRESENT'] = df['TAX CLASS AT PRESENT'].astype('category')
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
df['GROSS SQUARE FEET']= pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
#df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], errors='coerce')
df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
df['BOROUGH'] = df['BOROUGH'].astype('category')

df.isnull().sum()
df = df.dropna()
df.shape

data = df
numeric_data=data.select_dtypes(include=[np.number])
data = data[(data['SALE PRICE'] > 100000) & (data['SALE PRICE'] < 5000000)]
data = data[data['GROSS SQUARE FEET'] < 10000]
data = data[data['LAND SQUARE FEET'] < 10000]

data = data[(data['TOTAL UNITS'] > 0) & (data['TOTAL UNITS'] != 2261)]

cat_data=data.select_dtypes(exclude=[np.number])
numeric_data=data.select_dtypes(include=[np.number])
del data['ADDRESS']
del data['APARTMENT NUMBER']

#transform the numeric features using log(x + 1)
from scipy.stats import skew
skewed = data[numeric_data.columns].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
data[skewed] = np.log1p(data[skewed])
scaler = StandardScaler()
scaler.fit(data[numeric_data.columns])
scaled = scaler.transform(data[numeric_data.columns])

for i, col in enumerate(numeric_data.columns):
       data[col] = scaled[:,i]

data.head()

#Dropping few columns
del data['BUILDING CLASS AT PRESENT']
del data['BUILDING CLASS AT TIME OF SALE']
del data['NEIGHBORHOOD']

#Select the variables to be one-hot encoded
one_hot_features = ['BOROUGH', 'BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE']
one_hot_encoded = pd.get_dummies(data[one_hot_features])

# Replacing categorical columns with dummies
fdf = data.drop(one_hot_features,axis=1)
fdf = pd.concat([fdf, one_hot_encoded] ,axis=1)

Y_fdf = fdf[['SALE PRICE']]
X_fdf = fdf.drop('SALE PRICE', axis=1)

Y_fdf.to_csv('/Users/adelsondias/Documents/Repos/isklearn/data/nyc-houses/nyc-houses_y.csv')
X_fdf.to_csv('/Users/adelsondias/Documents/Repos/isklearn/data/nyc-houses/nyc-houses_X.csv')
