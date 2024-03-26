from create_dataset import *
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np



num_clusters = 100
samples_test = [1000,10000,100000,1000000]

times2 = []
times4 = []
times8 = []
times32 = []

for i in range(len(samples_test)):
    filename = f"{os.getcwd()}/dataset/{samples_test[i]}_{num_clusters}.csv"
    filename_centroids = f"{os.getcwd()}/dataset/{samples_test[i]}_{num_clusters}_centroids.csv"

    x = generate_blob_dataset(samples_test[i], num_clusters)
    save_to_csv(x, filename)
    c = random_centroids(x,num_clusters)
    save_to_csv(c,filename_centroids)



    proc1 = subprocess.Popen(["./build/kmeans", filename, filename_centroids,"2"], stdout=subprocess.PIPE)
    output, _ = proc1.communicate()
    output_str = output.decode("utf-8")
    lines = output_str.split("\n")
    times2.append(float(lines[0]))

    proc1 = subprocess.Popen(["./build/kmeans", filename, filename_centroids,"4"], stdout=subprocess.PIPE)
    output, _ = proc1.communicate()
    output_str = output.decode("utf-8")
    lines = output_str.split("\n")
    times4.append(float(lines[0]))

    proc1 = subprocess.Popen(["./build/kmeans", filename, filename_centroids,"8"], stdout=subprocess.PIPE)
    output, _ = proc1.communicate()
    output_str = output.decode("utf-8")
    lines = output_str.split("\n")
    times8.append(float(lines[0]))

    proc1 = subprocess.Popen(["./build/kmeans", filename, filename_centroids,"32"], stdout=subprocess.PIPE)
    output, _ = proc1.communicate()
    output_str = output.decode("utf-8")
    lines = output_str.split("\n")
    times32.append(float(lines[0]))

    print(f"done test with {samples_test[i]} points {i+1}/{len(samples_test)}")


fig, ax = plt.subplots()
plt.title(f'k={num_clusters}')
plt.plot(np.arange(len(samples_test)),times2,'-o',color = 'b',label="2")
plt.plot(np.arange(len(samples_test)),times4,'-o',color = 'r',label="4")
plt.plot(np.arange(len(samples_test)),times8,'-o',color = 'y',label="8")
plt.plot(np.arange(len(samples_test)),times32,'-o',color = 'g',label="32")
ax.xaxis.set_ticks(np.arange(len(samples_test))) #set the ticks to be a
ax.xaxis.set_ticklabels(samples_test) # change the ticks' names to x
plt.legend(loc="upper left")
plt.xlabel("#points")
plt.ylabel("execution time (s)")
plt.show()