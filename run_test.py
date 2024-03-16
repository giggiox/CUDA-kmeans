from create_dataset import *
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np



num_clusters = 5
samples_test = [10000,100000,1000000,10000000]
seq_times = []
par_times = []
cuda_times = []

for i in range(len(samples_test)):
    filename = f"{os.getcwd()}/dataset/{samples_test[i]}_{num_clusters}.csv"
    filename_centroids = f"{os.getcwd()}/dataset/{samples_test[i]}_{num_clusters}_centroids.csv"

    seq_time = 0
    par_time = 0
    cuda_time = 0
    num_test = 1
    for _ in range(num_test):
        x = generate_blob_dataset(samples_test[i], num_clusters)
        save_to_csv(x, filename)
        c = random_centroids(x,num_clusters)
        save_to_csv(c,filename_centroids)

        dt_seq = 0
        proc = subprocess.Popen(["./cmake-build-release/ompkmeans", filename, filename_centroids], stdout=subprocess.PIPE)

        output, _ = proc.communicate()
        output_str = output.decode("utf-8")
        lines = output_str.split("\n")

        seq_time += float(lines[0])
        par_time += float(lines[1])


        proc1 = subprocess.Popen(["./cuda/cuda", filename, filename_centroids], stdout=subprocess.PIPE)

        output, _ = proc1.communicate()
        output_str = output.decode("utf-8")
        lines = output_str.split("\n")

        cuda_time += float(lines[0])

    print(f"done test with {samples_test[i]} points {i+1}/{len(samples_test)}")





    seq_times.append(seq_time/num_test)
    par_times.append(par_time/num_test)
    cuda_times.append(cuda_time/num_test)




fig, ax = plt.subplots()
plt.title(f'k={num_clusters}')

plt.plot(np.arange(len(samples_test)),seq_times,'-o',color = 'b',label="sequential")
plt.plot(np.arange(len(samples_test)),par_times,'-o',color = 'r',label="openmp")
ax.xaxis.set_ticks(np.arange(len(samples_test))) #set the ticks to be a
ax.xaxis.set_ticklabels(samples_test) # change the ticks' names to x
plt.legend(loc="upper left")
plt.xlabel("#points")
plt.ylabel("execution time (s)")
plt.show()


fig, ax = plt.subplots()
speedup = []
for i in range(len(seq_times)):
    speedup.append(seq_times[i]/par_times[i])
plt.plot(np.arange(len(samples_test)),speedup,'-o',color = 'r')
ax.xaxis.set_ticks(np.arange(len(samples_test))) #set the ticks to be a
ax.xaxis.set_ticklabels(samples_test) # change the ticks' names to x
plt.xlabel("#points")
plt.ylabel("speedup")
plt.show()



fig, ax = plt.subplots()
plt.title(f'k={num_clusters}')

plt.plot(np.arange(len(samples_test)),seq_times,'-o',color = 'b',label="sequential")
plt.plot(np.arange(len(samples_test)),par_times,'-o',color = 'r',label="openmp")
plt.plot(np.arange(len(samples_test)),cuda_times,'-o',color = 'y',label="cuda")
ax.xaxis.set_ticks(np.arange(len(samples_test))) #set the ticks to be a
ax.xaxis.set_ticklabels(samples_test) # change the ticks' names to x
plt.legend(loc="upper left")
plt.xlabel("#points")
plt.ylabel("execution time (s)")
plt.show()


fig, ax = plt.subplots()
speedup = []
for i in range(len(seq_times)):
    speedup.append(seq_times[i]/cuda_times[i])
plt.plot(np.arange(len(samples_test)),speedup,'-o',color = 'r')
ax.xaxis.set_ticks(np.arange(len(samples_test))) #set the ticks to be a
ax.xaxis.set_ticklabels(samples_test) # change the ticks' names to x
plt.xlabel("#points")
plt.ylabel("speedup")
plt.show()








