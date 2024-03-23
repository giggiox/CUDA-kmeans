from create_dataset import *
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np



num_clusters = 1000
samples_test = [1000,10000,100000,1000000,10000000]

cuda_times1024 = []
cuda_times512 = []
cuda_times256 = []
cuda_times128 = []

for i in range(len(samples_test)):
    filename = f"{os.getcwd()}/dataset/{samples_test[i]}_{num_clusters}.csv"
    filename_centroids = f"{os.getcwd()}/dataset/{samples_test[i]}_{num_clusters}_centroids.csv"

    x = generate_blob_dataset(samples_test[i], num_clusters)
    save_to_csv(x, filename)
    c = random_centroids(x,num_clusters)
    save_to_csv(c,filename_centroids)

    proc1 = subprocess.Popen(["./cuda/cuda1024", filename, filename_centroids], stdout=subprocess.PIPE)
    output, _ = proc1.communicate()
    output_str = output.decode("utf-8")
    lines = output_str.split("\n")
    cuda_times1024.append(float(lines[0]))

    proc1 = subprocess.Popen(["./cuda/cuda512", filename, filename_centroids], stdout=subprocess.PIPE)
    output, _ = proc1.communicate()
    output_str = output.decode("utf-8")
    lines = output_str.split("\n")
    cuda_times512.append(float(lines[0]))

    proc1 = subprocess.Popen(["./cuda/cuda256", filename, filename_centroids], stdout=subprocess.PIPE)
    output, _ = proc1.communicate()
    output_str = output.decode("utf-8")
    lines = output_str.split("\n")
    cuda_times256.append(float(lines[0]))

    proc1 = subprocess.Popen(["./cuda/cuda128", filename, filename_centroids], stdout=subprocess.PIPE)
    output, _ = proc1.communicate()
    output_str = output.decode("utf-8")
    lines = output_str.split("\n")
    cuda_times128.append(float(lines[0]))

    print(f"done test with {samples_test[i]} points {i+1}/{len(samples_test)}")


fig, ax = plt.subplots()
plt.title(f'k={num_clusters}')
plt.plot(np.arange(len(samples_test)),cuda_times1024,'-o',color = 'b',label="1024")
plt.plot(np.arange(len(samples_test)),cuda_times512,'-o',color = 'r',label="512")
plt.plot(np.arange(len(samples_test)),cuda_times256,'-o',color = 'y',label="256")
plt.plot(np.arange(len(samples_test)),cuda_times128,'-o',color = 'g',label="128")
ax.xaxis.set_ticks(np.arange(len(samples_test))) #set the ticks to be a
ax.xaxis.set_ticklabels(samples_test) # change the ticks' names to x
plt.legend(loc="upper left")
plt.xlabel("#points")
plt.ylabel("execution time (s)")
plt.show()