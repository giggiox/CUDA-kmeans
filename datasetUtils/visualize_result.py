import matplotlib.pyplot as plt
import pandas as pd
import os

def show_dataset(dataset_path, is_result=False):
    df = pd.read_csv(dataset_path, header=None)
    plt.figure(figsize=(10, 8))
    
    for _, row in df.iterrows():
        cluster_label = int(row.iloc[-1])
        coords = row.iloc[:2].values
        plt.scatter(coords[0], coords[1], color=f'C{cluster_label}', alpha=0.6)
    plt.title('Visualizzazione Dataset 2D')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    dataset = "sequential_result.csv"
    show_dataset(f"../datasetUtils/result/{dataset}", is_result=True)