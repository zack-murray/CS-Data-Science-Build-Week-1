# K-Means Clustering Algorithm Implementation

## Required Packages:

    - numpy
    - scipy.spatial

## Usage:

### Imports

```py
from KMeans2 import Kmeans
from sklearn import preprocessing (for testing)
```

### Model:

```py
# Instantiate model
model = Kmeans(n_clusters=3, seed=7, max_iter=20)

# Fit
model.fit(data)

# Prediction
model.predict(data)
```

### Blogpost:

```py
https://medium.com/@zachery.murray/implementing-a-k-means-clustering-algorithm-from-scratch-214a417b7fee
```
