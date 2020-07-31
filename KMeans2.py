import numpy as np
from scipy.spatial import distance

class Kmeans:
    def __init__(self, n_clusters, seed=None, max_iter=300):
        self.n_clusters = n_clusters
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter
        
    def initialize_centroids(self, data):
        '''
        Randomly initializes centroids in regards to 
        n_clusters.

        input
        -----
        data
            2D numpy array 
        '''
        # Assign n_clusters amount of centroids to random data points
        initial_centroids = np.random.permutation(data.shape[0])[:self.n_clusters]
        self.centroids = data[initial_centroids]
        return self.centroids
    
    def get_og_centroids(self, data):
        '''
        Instantiates a variable that keeps track of
        the centroids position before it's updated
        in it's iterations.

        input
        -----
        data
            2D numpy array
        '''
        # Instantiate centroids before going through re-calculating
        self.og_cents = self.centroids
        
        return self.og_cents

    def assign_clusters(self, data):
        '''
        Computes the distance of data points from clusters
        and assigns them accordingly to the nearest 
        centroid. If given a 1-dimensional array, converts it
        to 2-dimensions. 

        input
        -----
        data
            2D numpy array 
        '''
        # If data is one-dimension
        if data.ndim == 1:
            # Reshape to two
            data = data.reshape(-1,1)
        # Calculate the distances between each data point and each centroid
        distance_to_centroid = distance.cdist(data, self.centroids, 'euclidean')
        # Get nearest centroid to each point based on distance
        self.cluster_labels = np.argmin(distance_to_centroid, axis=1)

        return self.cluster_labels

    def update_centroids(self, data):
        '''
        Calculates the average of all the data points in
        a cluster and assigns new centroids to lessen
        variance. 

        input
        -----
        data
            2D numpy array 
        '''
        # Calculate the mean of each cluster and reassign centroids
        self.centroids = np.array([data[self.cluster_labels == i].mean(axis = 0) for i in range(self.n_clusters)])
        return self.centroids

    def fit(self, data):
        '''
        Fits n_clusters centroids to the data and clusters them.
        Contains the loop that will continuously assign clusters
        and update centroids until the distance between old centroids
        and new centroids in negligable. 

        input
        -----
        data
            2D numpy array
        '''
        self.centroids = self.initialize_centroids(data)
        for iter in range(self.max_iter):
            self.og_cents = self.get_og_centroids(data)
            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)
            # Convergence
            if sum(self.og_cents[0]) == sum(self.centroids[0]):
                break
            # print(self.og_cents[0])
            # print(self.centroids[0])
            # if iter % 100 == 0:
            # Will print for every iteration completed
            if iter % 1 == 0:
                print("Iterations run: %d" %iter)
        print("Iterations complete!")

        return self

    def predict(self, data):
         '''
        Evaluates data points after being fit, assigns
        them to a centroid/cluster and labels them 
        accordingly. 

        input
        -----
        data
            2D numpy array

        output
        ------
        labels
            ndarray of the labels each point is closest to
        '''
        return self.assign_clusters(data)
        