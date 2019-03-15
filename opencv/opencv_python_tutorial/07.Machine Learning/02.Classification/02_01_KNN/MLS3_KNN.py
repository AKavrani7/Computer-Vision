##### Classification algorithm #########
######## K Nearest Neighbors KNN #######

import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

## Data Features
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
## Replace NAN -> -99999
df.replace('?',-99999, inplace=True)
print(df.head())
print('\n')

df.drop(['id'], 1, inplace=True)
print(df.head())
print('\n')

## To print complete dataset
#print(df.to_string())

## X-Features
X = np.array(df.drop(['class'], 1))
## y-label or class
y = np.array(df['class'])

## Creoss Validation
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

## Type of classifier
clf = neighbors.KNeighborsClassifier()
## Training the classifier
clf.fit(X_train, y_train)

## Testing accuracy over test data  
accuracy = clf.score(X_test, y_test)
print(accuracy)

### Predicting the class
## Sample input
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
## Reshaping it
example_measures = example_measures.reshape(len(example_measures), -1)
## Predicition step
prediction = clf.predict(example_measures)
print(prediction)
