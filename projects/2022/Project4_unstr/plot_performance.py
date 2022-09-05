import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir("build");

#nx, ny, time_finite, time_diff
times = pd.read_csv("benchmark_cache.csv", sep = " ").to_numpy(); 
#times = pd.read_csv("benchmark.csv", sep = " ").to_numpy(); 

num_nodes = np.multiply(times[:,0],times[:,1]);

print(num_nodes);

plt.plot(num_nodes, times[:,2], label = "grid stored as 1d array", marker = "o")
plt.plot(num_nodes, times[:,3], label = "usage of lookup table", marker = "o")

plt.xlabel("num nodes e^3");
plt.ylabel("time [s]");
plt.legend();
plt.grid(True)

x_ticks_labels = [str(int(i/10**3)) for i in num_nodes]
plt.xticks(num_nodes);
plt.xticks(num_nodes[0:-1:2], x_ticks_labels[0:-1:2]);


#plt.savefig("performanceplot.png")
plt.savefig("performanceplot_cache.png");


 
