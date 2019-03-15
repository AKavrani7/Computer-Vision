
## KNN Euclid algorithm from scratch ###

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

## To avoid using a lower K value than we have groups
import warnings

from math import sqrt

## To get the most popular votes.
from collections import Counter

style.use('fivethirtyeight')


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            ## 1. very slow
            #euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )

            ## 2. NumPy linear algebra (slow)
            #euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            #print(euclidean_distance)
            
            ## 3. Fast, By using Euclidean Norm which calculate magnitude of vector
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            #print(euclidean_distance)

            distances.append([euclidean_distance,group])
                             
    ## First k elements are saved in votes where votes is a list
    votes = [i[1] for i in sorted(distances)[:k]]
    ## OR
    #votes = []
    #for i in sorted(distances)[:k]:
    #                        votes.append(i[1])

    ## Suppose k = 5
    ## Now in Counter(votes): ['r','r','r','k','k']
    ## Counter(votes).most_common(2): [('r',3),('k',2)]
    ## Counter(votes).most_common(1): [('r',3)]
    ## Counter(votes).most_common(1)[0]: ('r',3)
    ## Counter(votes).most_common(1)[0][0]: 'r'
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

## Dataset
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0], j[1], s=50, color=i)

plt.scatter(new_features[0],new_features[1],s=10)

result = k_nearest_neighbors(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=100, color = result)
plt.show()



