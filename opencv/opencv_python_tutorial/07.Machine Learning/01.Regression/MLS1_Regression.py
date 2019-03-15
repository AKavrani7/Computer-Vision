################# STEP 1. DATA MANIPULATION #################
## Libraries for Data Manipulation
import pandas as pd
import quandl
import math

## Data Features
df = quandl.get('WIKI/GOOGL')
print(df.head())
print('\n')

## Exsisting Features
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
print(df.head())
print('\n')

## New featurs
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())
print('\n')

## Forcast Column
forecast_col = 'Adj. Close'
print(forecast_col)
print('\n')

## Fill any NaN data with -99999
## Either drop all feature/label sets that contain missing data
## or fill them with a random value 
df.fillna(value=-99999, inplace=True)
print(df.head())
print('\n')


## here we wish to predict 1% of entire length of the dataset,
## thus if our data is 100 days of stock prices, we will predict
## 1 day out into the future.
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
print('\n')


## Shift function: In simple words, a  particular column is pulled in upwards direction
## by a particular number, leaving NAs in the bottom. Later those rows with NaN
## were dropped.
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head())
print('\n')


############### Step 2 Training and Testing ##########
## Libraries for training and testing ####
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

## X-Features 
X = np.array(df.drop(['label'], 1))
## Pre-Processing Step 
## We wish our features in ML to be in range of -1 to 1(Speeds up processing).
X = preprocessing.scale(X)
##Our X_lately variable contains the most recent features,
##which we're going to predict against
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)

## y-Label
y = np.array(df['label'])

## When you test your classifier on test_data then the sort of
## accuracy and reliability you get is called confidence score.

## cross validation or model selection shuffles the data 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

## svm.SVR classifier provide a paremeter "Kernel"
for k in ['linear','poly','rbf','sigmoid']:
    ## Support Vector Regression from Scikit-Learn's svm package:
    clf = svm.SVR(kernel=k)
    ## Training the train_dataSet
    clf.fit(X_train, y_train)
    ## Confidence Score
    confidence = clf.score(X_test, y_test)
    print(k,confidence)


## Linear Regression
print("Linear Regression with threads or n_jobs = 10")
## Linear Regression with 10 n_jobs 
clf = LinearRegression(n_jobs = 10)
## NOTE: if you use n_jobs = -1 it will use all threads
## Training the train_dataSet
clf.fit(X_train, y_train)
## Confidence Score
confidence = clf.score(X_test, y_test)
print(confidence)

print("Linear Regression with ALL threads or n_jobs = -1")
## Linear Regression with ALL threads    
clf = LinearRegression(n_jobs = -1)
## NOTE: if you use n_jobs = -1 it will use all threads
## Training the train_dataSet
clf.fit(X_train, y_train)
## Confidence Score
confidence = clf.score(X_test, y_test)
print(confidence)


#### Sstep 2.5 Intermediate step Pickling #######
## Convert Data into Serialized Data
## With pickle, you can save any Python object, like our classifier. 
import pickle

## Saving our classifier
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)
    
## After saving the "linearregression.pickle" you can use the classifier :-> 
## save the serialized data 
pickle_in = open('linearregression.pickle','rb')
## Load the serialized data and save it to clf
clf = pickle.load(pickle_in)

######## Step 3. Forecasting and Predicting ################
## Now we are trying to predict or checking is our classifier is working or not,
## you can consider that we have divided our train data set also.

print("Predicting Stock values of next 35 days")
forecast_set = clf.predict(X_lately)
print(forecast_set)
print(confidence, forecast_out)


########## Step 4 visualizing the above information #################
#### Libraries ####
## We import datetime to work with datetime objects
import datetime
## matplotlib's pyplot package for graphing
import matplotlib.pyplot as plt
## style to make our graphs look decent.
from matplotlib import style
style.use('ggplot')

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


