import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D

dataset = "1000_3_5.csv"
df = pd.read_csv(f"{os.getcwd()}/result/{dataset}", header=None)
dimension = len(df.iloc[0].values) - 1
if dimension == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
for _, row in df.iterrows():
    clusterLabel = int(row.iloc[-1])
    coords = row.iloc[:-1].values
    num_coords = len(coords)
    if len(coords) == 2:
        plt.scatter(coords[0], coords[1], color=f'C{clusterLabel}')
    elif len(coords) == 3:
        ax.scatter(coords[0], coords[1], coords[2], color=f'C{clusterLabel}')

plt.show()
