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

def make_animation(
    baseName,
    what_to_plot='h',
    make_gif=True,
    make_pngs=True,
):

    # --- SETTINGS --- #

    # Choose what to plot: what_to_plot
    # 	* h (fluid height)
    #	* u (longitudinal velocity)
    #	* v (latitudinal velocity)
    # 	* quiver (velocity quiver-plot)  # not implemented
    #	* vorticity (relative vorticity)  # not implemented
    #	* mesh  # not implemented

    # Choose projection for height plot:
    #	* cyl (cylindrical) # not implemented
    #	* ortho (orthogonal)
    projection = 'ortho'

    # Choose if you want to save the movie and if yes which format do you prefer:
    #	* mp4 # not implemented
    #	* mpg # not implemented
    # and the frames per seconds
    # save_movie = True
    # movie_format = 'gif'
    fps = 3


    # --- LOAD DATA --- #

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
    
    t = t[:,0]

    # --- PLOT MODEL VARIABLES --- #
    
    plot_variables = {
        'h':h, 
        'u':u, 
        'v':v, 
    }
    plot_titles = {
        'h': 'Fluid height [m]',
        'u': 'Zonal velocity [m/s]',
        'v': 'Meridional velocity [m/s]',
    }
    
    var = plot_variables[what_to_plot]
    title = plot_titles[what_to_plot]
    
    fig1 = plt.figure(figsize=[8,8])

    if (projection == 'cyl'):
        proj=ccrs.PlateCarree(central_longitude=0.0, globe=None)

    elif (projection == 'ortho'):
        proj=ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0, globe=None)

    ax = plt.subplot(1,1,1, projection=proj)

    color_scale = np.linspace(var.min(), var.max(), 30, endpoint = True)
    x1, y1 = phi*180.0/math.pi, theta*180.0/math.pi

    def update(frame):
        # for each frame, update the data stored on each artist.
        ax.contourf(x1[1:-1,:], y1[1:-1,:], var[:,:,frame], color_scale, transform=ccrs.PlateCarree())
        ax.set_title(f'{title}: time = {t[frame] / 3600.0:5.2f} hours\n')
        return ax

    if make_gif:
        ani = animation.FuncAnimation(fig=fig1, func=update, frames=Nt, interval=1)
        ani.save(filename=baseName+"_"+what_to_plot+".gif", writer="pillow", fps=fps)
    
    if make_pngs:
        update(0)
        plt.tight_layout()
        plt.savefig(baseName+"_"+what_to_plot+'_IC.png')
        
        update(-1)
        plt.tight_layout()
        plt.savefig(baseName+"_"+what_to_plot+'_T.png')
            
if __name__ == '__main__':
    
    GRIDTOOLS_ROOT = '.' #os.environ.get('GRIDTOOLS_ROOT')
    #baseName = GRIDTOOLS_ROOT + '/data/swes-numpy-0-M180-N90-T5-1-'
    # baseName = GRIDTOOLS_ROOT + '/data/swes-gt4py-0-M180-N90-T4-0-'
    
    baseName = './data/IC0_T4_noDiff_gt4py/swes-gt4py-0-M180-N90-T4-0-'
    # baseName = './data/IC1_T4_noDiff_gt4py/swes-gt4py-1-M180-N90-T4-0-'
    print('[animation.py] Will be saving to '+baseName+'... .')
    
    make_gif = False
    
    print('[animation.py] Plotting h...')
    make_animation(baseName, what_to_plot='h',make_gif=make_gif)
    # print('[animation.py] Plotting u...')
    # make_animation(baseName, what_to_plot='u',make_gif=make_gif)
    # print('[animation.py] Plotting v...')
    # make_animation(baseName, what_to_plot='v',make_gif=make_gif)
    print('[animation.py] Done.')
    
    # --- TO DO: add settings as keywords with defaults and/or line arguments --- #


