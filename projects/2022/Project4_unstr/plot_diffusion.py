#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 18:12:51 2022

@author: andrin
"""

#%% Plotting the CSV-file representing the mesh and the field values

# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm
from matplotlib import colors
from matplotlib import collections
from matplotlib import path

os.chdir('build')

initial_field = True;

# reading in file
mesh = pd.read_csv( 'centers.csv', header=None, delim_whitespace=True )
nodes = pd.read_csv( 'nodes.csv', header=None, delim_whitespace=True )
lookup_table = pd.read_csv( 'lookup_table.csv', header=None, delim_whitespace=True )
element_vertices = pd.read_csv( 'element_vertices.csv', header=None, delim_whitespace=True)
if(initial_field == True):
   diffusion_output = pd.read_csv("initial_field.csv", header = None).to_numpy(dtype = float)
else:
   diffusion_output = pd.read_csv("output_diffusion.csv", header = None).to_numpy(dtype=float)

#normalize diffusion output
#diffusion_output = np.abs(diffusion_output)
diffusion_max = np.abs(diffusion_output).max();
diffusion_output /= diffusion_max;

print(diffusion_max);
print(np.abs(diffusion_output).min());


n_elements, numNodePerElement = lookup_table.shape
el_shape = ( n_elements, numNodePerElement, 2 )
el = np.zeros( el_shape )



for i in range( element_vertices.shape[0] ):
    for j in range(numNodePerElement):
        if np.mod(i, numNodePerElement) == j:
            el[int(i / numNodePerElement), j] = element_vertices[i : i + 1][0:1]


#%% Plotting
# preparing figure
#fig

fig = plt.figure( 1, [8, 8], 100, frameon = True );
ax1 = fig.add_subplot( 111 )
#ax2 = fig.add_subplot(112)
plt.xlim([-50, 50])
plt.ylim([-50, 50])
ax1.tick_params(axis='both', which='major', labelsize=20)
colormap = cm.get_cmap( 'GnBu', 1000 )


#plt.pcolor(el.flatten(), data[::-1],edgecolors='k', linewidths=1)


# plotting elements (cells) with respective index and nodes
"""
ax1.scatter( mesh[0], mesh[1], color = "grey", alpha = 0.5)
for i in range( mesh.shape[0] ):
    plt.text( mesh[0][i], mesh[1][i], str(i) )
"""

# plotting edges and filling cells with color corresponding to index
for i in range( el.shape[0] ):
    
    polygon = patches.Polygon(el[i,:,:], closed=True, edgecolor=colors.to_rgba("black",alpha=1), facecolor = colors.to_rgba("white",alpha=0))
    ax1.add_patch(polygon)
    
    #for j in range( el.shape[1] ):
    #ax1.plot( el[i, j:(j + 1) % 2, 0], el[i, j:(j + 1) % 2, 1], color = 'grey' )
    # ax1.plot( [el[i,2], el[i,4]], [el[i,3], el[i,5]], color = 'grey' )
    # ax1.plot( [el[i,4], el[i,0]], [el[i,5], el[i,1]], color = 'grey' )
   
    alpha_ = np.abs(diffusion_output[i,0]); 
    
    #print(alpha_); 

    ax1.fill_between( 
        [el[i, j % el.shape[1], 0] for j in range(el.shape[1] + 1)], 
        [el[i, j % el.shape[1], 1] for j in range(el.shape[1] + 1)], 
        facecolor = "mediumturquoise", alpha = alpha_);

       
	#colormap(output_diffusion[i]);
        #(1-(i/el.shape[0])/2,0.7,(i/el.shape[0])/1.5+0.3,0.5) )
#ax1.scatter( nodes[0], nodes[1], color="red" )


#plt.show()
if(initial_field == True):
    filename = "initial_field.png"
else:
    filename = "diffusion.png"

plt.savefig(filename)


