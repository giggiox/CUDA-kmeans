from create_dataset import *
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np



num_features = 3
num_clusters = 5
samples_test = [10,100,1000,10000,100000]
seq_times = []

for num_samples in samples_test:
    filename = f"{os.getcwd()}/dataset/{num_samples}_{num_features}_{num_clusters}.csv"
    x = generate_blob_dataset(num_samples, num_features, num_clusters)
    save_to_csv(x, filename)

    dt_seq = 0
    proc = subprocess.Popen(["./cmake-build-debug/ompkmeans", filename], stdout=subprocess.PIPE)
    output, _ = proc.communicate()
    output_str = output.decode("utf-8")
    lines = output_str.split("\n")
    dt_seq = float(lines[0])
    seq_times.append(dt_seq)

ypoints = np.array(seq_times)
plt.plot(ypoints, color = 'b')
plt.show()
