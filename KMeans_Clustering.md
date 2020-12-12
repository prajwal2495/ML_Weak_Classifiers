## K-Means Clustering

### Build your own K-Means Clustering Algorithm (with continuous input)

#### Hint: memorize the calculated distances to avoid redundant computations.

#### Implement my_KMeans.fit() function in [my_KMeans.py]
Inputs:
- X: pd.DataFrame, independent variables, each value is a continuous number of float type

#### Implement my_KMeans.predict() function in [my_KMeans.py]
Input:
- X: pd.DataFrame, independent variables, each value is a continuous number of float type

Output:
- Predicted categories of each input data point. List of str or int.

#### Implement my_KMeans.transform() function in [my_KMeans.py]
Transform to cluster-distance space.
Input:
- X: pd.DataFrame, independent variables, each value is a continuous number of float type

Output:
- dists = list of [dist to centroid 1, dist to centroid 2, ...]

### Test my_KMeans Algorithm with [A6.py]

 - It is expected to perform the same with [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) with input (algorithm = "full").
 
  
## Hint
 - If my_KMeans.py is too difficult to implement, you can try to complete [my_KMeans_hint.py].

