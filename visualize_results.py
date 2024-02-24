import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D

def show_dataset(dataset_path,is_result=False):
    df = pd.read_csv(dataset_path, header=None)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for _, row in df.iterrows():
        cluster_label = int(row.iloc[-1])
        coords = row.iloc[:3].values
        ax.scatter(coords[0], coords[1], coords[2], color=f'C{cluster_label}')
    plt.show()

if __name__ == "__main__":
    dataset = "1000_5.csv"
    show_dataset(f"{os.getcwd()}/result/{dataset}",is_result=True)
