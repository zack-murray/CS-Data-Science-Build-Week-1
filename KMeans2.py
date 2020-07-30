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

        initial_centroids = np.random.permutation(data.shape[0])[:self.n_clusters]
        self.centroids = data[initial_centroids]
        return self.centroids
    
    # def get_og_centroids(self, data):

    #     self.og_centroids = self.centroids
    #     return self.og_centroids

    def assign_clusters(self, data):
        if data.ndim == 1:
            data = data.reshape(-1,1)

        dist_to_centroid = distance.cdist(data, self.centroids, 'euclidean')
        self.cluster_labels = np.argmin(dist_to_centroid, axis=1)

        return self.cluster_labels

    def update_centroids(self, data):
        self.centroids = np.array([data[self.cluster_labels == i].mean(axis = 0) for i in range(self.n_clusters)])
        return self.centroids

    def fit(self, data):
        self.centroids = self.initialize_centroids(data)
        # self.og_centroids = self.get_og_centroids(data)
        for iter in range(self.max_iter):
            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)
            distances = distance.cdist(self.initialize_centroids(data), self.centroids, 'euclidean')
            if sum(distances) == 0:
                break
            # if iter % 100 == 0:
            if iter % 1 == 0:
                print("Iterations run: %d" %iter)
        print("Iterations complete!")

        return self

    def predict(self, data):
        return self.assign_clusters(data)
        