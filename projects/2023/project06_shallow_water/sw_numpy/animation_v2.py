"""
Script to plot results coming from finite difference SWES solver.
The user may specify all the solver options and let the code look for
the correspondent dataset or directly give the path to the dataset.
In any case, the user should specify all the settings.
"""

import os
import pickle
import numpy as np
import math
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.animation import PillowWriter 
import cartopy.crs as ccrs
#-----------------------------------------------------------
#from mpl_toolkits.basemap import Basemap
#-----------------------------------------------------------


# --- SETTINGS --- #

# Choose what to plot:
# 	* h (fluid height)
#	* u (longitudinal velocity)
#	* v (latitudinal velocity)
# 	* quiver (velocity quiver-plot)
#	* vorticity (relative vorticity)
#	* mesh
what_to_plot = 'h'

# Choose projection for height plot:
#	* cyl (cylindrical)
#	* ortho (orthogonal)
projection = 'ortho'

# Choose if you want to save the movie and if yes which format do you prefer:
#	* mp4
#	* mpg
# and the frames per seconds
save_movie = True
movie_format = 'gif'
fps = 1


# --- LOAD DATA --- #

GRIDTOOLS_ROOT = '.' #os.environ.get('GRIDTOOLS_ROOT')
#baseName = GRIDTOOLS_ROOT + '/data/swes-numpy-0-M180-N90-T5-1-'
baseName = GRIDTOOLS_ROOT + '/data/swes-numpy-0-M180-N90-T4-1-'

# Load h
with open(baseName + 'h', 'rb') as f:
    M, N, t, phi, theta, h = pickle.load(f)

# Load u
with open(baseName + 'u', 'rb') as f:
    M, N, t, phi, theta, u = pickle.load(f)

# Load v
with open(baseName + 'v', 'rb') as f:
    M, N, t, phi, theta, v = pickle.load(f)

# Apply wrap-around boundary conditions on data coming from
# any original solver implementation
phi = np.concatenate((phi, phi[0:1,:]), axis = 0)
theta = np.concatenate((theta, theta[0:1,:]), axis = 0)
h = np.concatenate((h, h[0:1,:]), axis = 0)
u = np.concatenate((u, u[0:1,:]), axis = 0)
v = np.concatenate((v, v[0:1,:]), axis = 0)
Nt = h.shape[2]


#
## --- PLOT HEIGHT --- ###

if (what_to_plot == 'h'):
    fig1 = plt.figure(figsize=[15,8])
    
    if (projection == 'cyl'):
        proj=ccrs.PlateCarree(central_longitude=0.0, globe=None)

    elif (projection == 'ortho'):
        proj=ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0, globe=None)
    
    ax = plt.subplot(1,1,1, projection=proj)
    
    color_scale = np.linspace(h.min(), h.max(), 30, endpoint = True)
    x1, y1 = phi*180.0/math.pi, theta*180.0/math.pi
    
    def update(frame):
        # for each frame, update the data stored on each artist.
        ax.contourf(x1[1:-1,:], y1[1:-1,:], h[:,:,frame], color_scale, transform=ccrs.PlateCarree())
        ax.set_title('Fluid height [m]: time = %5.2f hours\n' % (t[frame] / 3600.0))
        return ax
    
    ani = animation.FuncAnimation(fig=fig1, func=update, frames=Nt, interval=1)
    ani.save(filename="test.gif", writer="pillow")


