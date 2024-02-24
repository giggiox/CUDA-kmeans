from create_dataset import *
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np



num_clusters = 5
samples_test = [100,1000,10000,100000]
seq_times = []
par_times = []

for num_samples in samples_test:
    filename = f"{os.getcwd()}/dataset/{num_samples}_{num_clusters}.csv"

    seq_time = 0
    par_time = 0
    for _ in range(10):
        x = generate_blob_dataset(num_samples, num_clusters)
        save_to_csv(x, filename)

        dt_seq = 0
        proc = subprocess.Popen(["./cmake-build-release/ompkmeans", filename], stdout=subprocess.PIPE)
        output, _ = proc.communicate()
        output_str = output.decode("utf-8")
        lines = output_str.split("\n")

        seq_time += float(lines[0])
        par_time += float(lines[1])


    seq_times.append(seq_time/10)
    par_times.append(par_time/10)





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







