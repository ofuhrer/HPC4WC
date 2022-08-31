import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

os.chdir("build")

neighbours = pd.read_csv("lookup_table.csv", delim_whitespace = True).to_numpy();

num_neighbours_origin = 0; #number of neighbours that are not on the same cache line (cache line mac 64 Byte) -> 8 doubles on a cache line for finite difference method
num_neighbours_lookup = 0; #number of neighbours not on the same cache line using lookup_table. 

distance_of_neighbours = 0; 

#dictionary = np.zeros(neighbours.shape[0]);#store number of neighbours which indices are a given distance apart from each other (distance is a key) 

distances = np.empty(neighbours.size);

num_neighbours = neighbours.shape[1];

for i in range(0, neighbours.shape[0]):
   for j in range(0,num_neighbours):
      if(neighbours[i,j] == -1):
         dist = 0;
      else:
         dist = abs(i-neighbours[i,j]);
      distances[num_neighbours*i+j] = dist;
      #dictionary[dist] = dictionary[dist]+1;

#print(distances);
#non_zero_indices = np.nonzero(dictionary > 3)[0];

#compute distances of neighbouring indices for common finite element method
num_cells = neighbours.shape[0];
nx = math.floor(math.sqrt(num_cells)); #TODO: only if square

dist_1 = nx*np.ones(2*num_cells-2*nx); #top, bottom neighbour
dist_2 = np.ones(2*num_cells-2*nx); #left and right neighbour
dist_3 = np.zeros(4*nx); #boundaries
distances_finite = np.concatenate([dist_3, dist_2, dist_1]);

print(distances_finite.size); 
print(distances.size);
#############################################################

right_boundary = nx+10;

bins = np.arange(0,right_boundary,5);

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
fig.set_size_inches(10, 4.8)

n, bins_, patches = axs[0].hist(distances, rwidth=.8, label = "use of lookup table");
axs[0].hist(distances_finite, rwidth=.8, bins = bins_, alpha = 0.6, label = "grid stored as 1d array");
#plt.xticks(np.arange(distances.min(), distances.max()+1, 1.0))
 
axs[0].set_xlabel('distance of neighbour indices')
axs[0].set_ylabel('num neighbours')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.xlim(40, 160)
#plt.ylim(0, neighbours.size)
axs[0].grid(True)
axs[0].legend();

axs[1].hist(distances,rwidth=.8,bins=bins, label = "use of lookup table");
axs[1].hist(distances_finite, rwidth=.8, bins = bins, alpha = 0.6, label = "grid stored as 1d array" );

axs[1].set_xlabel('distance of neighbour indices')
axs[1].set_ylabel('num neighbours')
axs[1].grid(True)
axs[1].legend()


plt.savefig("histogram.png");

             
