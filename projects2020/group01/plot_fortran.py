import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = np.loadtxt('results.dat')
df = pd.DataFrame(data, columns=['iter', 'gridpoints', 'ice_fraction', 'time'])


gp = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
ii = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]

fig = plt.subplots(figsize=(8, 6))
plt.grid()
ys = []
for i in ii:
    x = df[df.ice_fraction == i].groupby(df['gridpoints']).median().gridpoints
    yy = df[df.ice_fraction == i].groupby(df['gridpoints']).median().time / 1e3
    ys.append(yy)
colors = cm.rainbow(np.linspace(0, 1, 10))
for y, c, i in zip(ys, colors, ii):
    plt.scatter(x, y, label=str(i), s=8, color=c)

plt.legend(loc=2, ncol=3, title='fraction of sea ice points')
plt.xlabel('number of grid points')
plt.ylabel('elapsed time [s]')
plt.title('Performance Overview with Optimized Fortran')
plt.ylim(5e-6, 5e0)
plt.xscale('log')
plt.yscale('log')
plt.savefig("fortran_performance_optimized.png")


