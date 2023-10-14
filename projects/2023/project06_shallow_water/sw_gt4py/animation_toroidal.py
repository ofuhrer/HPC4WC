import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter 
import os
import pickle

def normalize(h):
    # normalize h from 0 to 1
    h_mean = np.mean(h, axis=0)
    h_deviation = h - h_mean
    return (h_deviation - np.min(h_deviation)) / (
        np.max(h_deviation) - np.min(h_deviation)
    )

def make_animation(
    baseName,
    what_to_plot='h',
    make_gif=False, # not implemented
    make_pngs=True,
):
    
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
    
    phi = phi[1:-1,:]
    theta = theta[1:-1,:]
    
    u_mag = np.sqrt(u ** 2 + v ** 2)
    
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

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.set_zlim(-1, 1)
    ax.view_init(40, 10)
    ax.set_axis_off()
    # ax.set_aspect("equal") # not implemented
    ax.dist = 7
    
    aspect_ratio = 0.4
    c, a = 1, aspect_ratio
    x = (c + a * np.cos(theta)) * np.cos(phi)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)
    
    # --- colorbar
    my_col = cm.plasma(normalize(var[:,:,0]))
    s = ax.plot_surface(
        x, y, z, rstride=1, cstride=1, facecolors=my_col, linewidth=0, antialiased=False
    )
    s.set_clim(vmin=0, vmax=1)
    s.set_cmap(cm.plasma)
    plt.colorbar(s)
    
    # --- gif
    def update(frame):
        print(f'Plotting frame {frame+1}/{Nt}')
        my_col = cm.plasma(normalize(var[:,:,frame]))
        s = ax.plot_surface(
            x, y, z, rstride=1, cstride=1, facecolors=my_col, linewidth=0, antialiased=False
        )
        
        ax.set_title(f'{title}: time = {t[frame] / 3600.0:5.2f} hours\n')

        return ax
    
    if make_gif:
        ani = animation.FuncAnimation(fig=fig, func=update, frames=Nt, interval=1)
        ani.save(filename=baseName+"_"+what_to_plot+".gif", writer="pillow",fps=fps)
    
    if make_pngs:
        
        # --- initial condition
        update(0)
        plt.tight_layout()
        plt.savefig(baseName+"_"+what_to_plot+'_IC.png', bbox_inches="tight")

        # --- final  time
        update(-1)
        plt.tight_layout()
        plt.savefig(baseName+"_"+what_to_plot+'_T.png', bbox_inches="tight")
    
if __name__ == '__main__':
    
    baseName = './data/swes-torus-numpy--1-M180-N90-T0-1-'

    print('[animation.py] Will be saving to '+baseName+'... .')
    
    print('[animation.py] Plotting h...')
    make_animation(baseName, what_to_plot='h')
    print('[animation.py] Plotting u...')
    make_animation(baseName, what_to_plot='u')
    print('[animation.py] Plotting v...')
    make_animation(baseName, what_to_plot='v')
    
    print('[animation.py] Done.')
    
    # --- TO DO: add settings as keywords with defaults and/or line arguments --- #


