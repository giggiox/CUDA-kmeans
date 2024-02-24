import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import os

def generate_blob_dataset(num_samples, num_clusters):
    x, _ = make_blobs(n_samples=num_samples, n_features=3, centers=num_clusters)
    return x

def save_to_csv(x, filename):
    df = pd.DataFrame(x)
    df.to_csv(filename, header=False, index=False)

if __name__ == "__main__":
    num_samples = 1000
    num_clusters = 5
    show_graph = True

    x = generate_blob_dataset(num_samples, num_clusters)
    save_to_csv(x, f"{os.getcwd()}/dataset/{num_samples}_{num_clusters}.csv")