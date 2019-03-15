
## Testing our KNN Euclid algorithm on Breast Cancer ###

import numpy as np
## To avoid using a lower K value than we have groups
import warnings
from math import sqrt
## To get the most popular votes.
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=5):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:            
            ## 3. Fast, By using Euclidean Norm which calculate magnitude of vector
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            #print(euclidean_distance)
            distances.append([euclidean_distance,group])
                             
    ## First k elements are saved in votes where votes is a list
    votes = [i[1] for i in sorted(distances)[:k]]
    ## Suppose k = 5
    ## Counter(votes).most_common(2): [('r',3),('k',2)]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

## Data Features
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

full_data = df.values.tolist()
print(full_data[:10])
print(20*'#')
full_data = df.astype(float).values.tolist()
print(full_data[:10])
print(20*'#')

## Shuffle the data
random.shuffle(full_data)
print(full_data[:10])
print(20*'#')

test_size = 0.20

## Dictionaries for training and testing set
## 2:Benign Tumor
## 4:Malignant Tumor
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
## First 80%
train_data = full_data[:-int(test_size*len(full_data))]
## Last 20%
test_data = full_data[-int(test_size*len(full_data)):]


for i in train_data:
    ## i[-1]: Last column which is class of the data
    ## train_set[i[-1]]: key of the train set == class
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])


correct = 0.0
total = 0.0

for group in test_set:
    #print(group)
    #print(10*'$')
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        #print(vote)
        if(group == vote):
            correct = correct + 1
        total = total + 1
        #print(correct)
        
print('Accuracy:', correct/total)
