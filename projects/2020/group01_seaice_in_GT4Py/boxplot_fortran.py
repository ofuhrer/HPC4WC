import numpy as np
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# read in fortran results
data = np.loadtxt('results.dat')
df = pd.DataFrame(data, columns=['iter', 'gridpoints', 'ice_fraction', 'time'])

# define grid points and sea ice fraction
gp = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
fp = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]

# select 0.25 ice fraction 
df = df[df['ice_fraction'] == 0.25]
df.time = df.time / 1e3


# define position for boxplot
pos = np.array(gp)

# define width for boxplot 
width =  pos * 0.3

# boxplot
fig, ax = plt.subplots(figsize=(8,6))
df.boxplot(ax=ax, by =['gridpoints'],positions=pos,widths=width,column = ['time'])
plt.title('Boxplot for 25% sea ice fraction with Optimized Fortran')
plt.suptitle("")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('number of grid points')
ax.set_ylabel('elapsed time [s]')
ax.set_xlim(2e1, 2e6)
ax.set_ylim(5e-7, 5e0)
plt.grid()
plt.tight_layout()
fig.savefig("boxplot_fortran_opt.png")
    






