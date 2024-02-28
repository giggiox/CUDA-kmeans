from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import os


centroids = [[2.39014695, -0.57684421,-10],[-7.53619806,  4.49955772,13.4]]


def generate_blob_dataset(num_samples, num_clusters):
    x, _ = make_blobs(n_samples=num_samples, n_features=3, centers=num_clusters)
    return x

def save_to_csv(x, filename):
    df = pd.DataFrame(x)
    df.to_csv(filename, header=False, index=False)




p = generate_blob_dataset(1000,2)
save_to_csv(p,"ciao.csv")
X = np.array(p)
initial_centroids = np.array(centroids)
k = initial_centroids.shape[0]
kmeans = KMeans(n_clusters=k, init=initial_centroids,max_iter=1)

kmeans.fit(X)

# Get the cluster centers
centroids = kmeans.cluster_centers_

# Get the cluster labels
labels = kmeans.labels_

print("Custom Initial Centroids:")
print(initial_centroids)
print("\nCluster centers:")
print(centroids)
print("\nCluster labels:")
print(labels)
