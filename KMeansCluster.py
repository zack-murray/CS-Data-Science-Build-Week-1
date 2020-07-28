import numpy as np


class K_Means:
    '''
    n_clusters = number of clusters
    The number of clusters to form as well as the number of centroids to generate.
    ------------
    tol = tolerance
    How much the centriod will move (by percent change). If our centroid is not moving
    more than the tolerance value, we know we're optimized. 
    ------------
    max_iter = max iterations
    Maximum number of iterations of the k-means algorithm for a single run.
    '''

    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):
        # Setting an empty dictionary to house centroids
        # Starting centroids wont change
        self.centroids = {}

        # Iterate through data
        for i in range(self.k):
            # Assign centroids as first two data samples in dataset
            # Shuffle data to make starting centroids random
            self.centroids[i] = data[i]

        # Iterate through max iterations
        for i in range(self.max_iter):
            # Empty dictionary to contain centroids and classifications
            # Clears out every iteration (changes everytime centroid changes)
            self.classifications = {}
            
            for i in range(self.k):
                # Create 2 dict keys 
                self.classifications[i] = []

            # Iterate through features
            for features in data:
                # Calculate distances of features to current centroids
                distances = [np.linalg.norm(features-self.centroids[centroid]) for centroid in self.centroids]
                # Index value at min of distances
                classification = distances.index(min(distances))
                # Features belongs to that centroid
                self.classifications[classification].append(features)

            
