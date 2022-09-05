#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 18:12:51 2022

@author: andrin
"""

#%% Plotting the CSV-file representing the mesh and the field values

# imports
import os
from pydoc import doc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm
from matplotlib import colors
from matplotlib import collections
from matplotlib import path

os.chdir('build')

# reading in file
mesh = pd.read_csv( 'centers.csv', header=None, delim_whitespace=True ).to_numpy()
nodes = pd.read_csv( 'nodes.csv', header=None, delim_whitespace=True )
lookup_table = pd.read_csv( 'lookup_table.csv', header=None, delim_whitespace=True )
element_vertices = pd.read_csv( 'element_vertices.csv', header=None, delim_whitespace=True )

n_elements, numNodePerElement = lookup_table.shape
el_shape = ( n_elements, numNodePerElement, 2 )
el = np.zeros( el_shape )

for i in range( element_vertices.shape[0] ):
    for j in range(numNodePerElement):
        if np.mod(i, numNodePerElement) == j:
            el[int(i / numNodePerElement), j] = element_vertices[i : i + 1][0:1]

#%% Plotting
# preparing figure
fig = plt.figure( 1, [20, 20], 200, frameon = True, tight_layout=True ) #for the mesh only set frameon to False
ax1 = fig.add_subplot( 111 )
# plt.title( 'Localized mesh with element indices', fontname='DejaVu Sans', fontsize=50, fontweight='bold', pad=25 )
plt.xlim([-50, 50])
plt.ylim([-50, 50])
ax1.tick_params(axis='both', which='major', labelsize=20)
colormap = cm.get_cmap( 'GnBu', 1000 )

# plotting elements (cells) with respective index and nodes
for i in range( mesh.shape[0] ):
    plt.text( mesh[i, 0] + 0.5, mesh[i, 1] + 0.5, str(i), fontsize=20 )

# plotting edges and filling cells with color corresponding to index

for i in range( el.shape[0] ):
   polygon = patches.Polygon(el[i,:,:], closed=True, edgecolor=colors.to_rgba("black",alpha=1), facecolor = colors.to_rgba("white",alpha=0))
   ax1.add_patch(polygon)
#    if i < el.shape[0] - 1:
#     ax1.plot(mesh[i:i+2, 0], mesh[i:i+2, 1], color="k", lw=4)

    #path_ = path.Path(np.concatenate([el[i,:,:],np.expand_dims(el[i,0,:],axis=0)], axis=0), codes = [path.Path.MOVETO, path.Path.LINETO, path.Path.LINETO, path.Path.CLOSEPOLY])
    #patch = patches.PathPatch(path_, edgecolor = "black", facecolor = None);
    #ax1.add_patch(patch)

   
    #for j in (lookup_table.iloc[i,:]):
    #if(j != -1):
    #ax1.plot([mesh.iloc[i,0],mesh.iloc[j,0]],[mesh.iloc[i,1],mesh.iloc[j,1]],color="red", linestyle = "-");    
   
    
   #uncomment if one wants to plot indices by color
   for j in range( el.shape[1] ):
      ax1.fill_between([el[i, j % el.shape[1], 0] for j in range(el.shape[1] + 1)], [el[i, j % el.shape[1], 1] for j in range(el.shape[1] + 1)],  facecolor = colormap(i/el.shape[0]) ) 
 
         
# ax1.scatter( nodes[0], nodes[1], color="red" )  
ax1.scatter(mesh[:, 0], mesh[:, 1], color = "red", alpha = 0.5)
    
"""

# plotting elements (cells) with respective index and nodes
ax1.scatter( mesh[0], mesh[1], alpha=0.7, color='red' )
for i in range( mesh.shape[0] ):
    plt.text( mesh[0][i] + 0.4, mesh[1][i] + 0.4, str(i) )
ax1.scatter( nodes[0], nodes[1], 10, color="grey" )
"""


plt.savefig("mesh.png")


