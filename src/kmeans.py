import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances

def initialize_clusters(points, k):
    return points[np.random.randint(points.shape[0], size=k)]
    
def get_distances(centroid, points):
    return np.linalg.norm(points - centroid, axis=1)

def kmeans(X, k):
    maxiter = 50

    # Initialize our centroids by picking random data points
    centroids = initialize_clusters(X, k)

    # Initialize the vectors in which we will store the
    # assigned classes of each data point and the
    # calculated distances from each centroid
    classes = np.zeros(X.shape[0], dtype=np.float64)
    distances = np.zeros([X.shape[0], k], dtype=np.float64)

    # Loop for the maximum number of iterations
    for i in range(maxiter):

        # Assign all points to the nearest centroid
        for i, c in enumerate(centroids):
            distances[:, i] = get_distances(c, X)

        # Determine class membership of each point
        # by picking the closest centroid
        classes = np.argmin(distances, axis=1)

        # Update centroid location using the newly
        # assigned data point classes
        for c in range(k):
            centroids[c] = np.mean(X[classes == c], 0)

    return centroids, classes