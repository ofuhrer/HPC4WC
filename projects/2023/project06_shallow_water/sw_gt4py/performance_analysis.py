#!/usr/bin/env python3
"""
Performance analysis for the NumPy and gt4py implementations
of a finite difference solver for Shallow Water Equations (SWES)
on a sphere and torus.
"""

import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from driver import driver

# --- GRID SIZES --- #
# keeping the 1:2 aspect ratio

# nxs = [ 90,180,360, 720]
# nys = [180,360,720,1480]
nxs = [ 180,360,360,720,720]
nys = [ 90 ,90 ,180,180,360]
ns  = [
    nx*ny for nx,ny in zip(nxs,nys)
]

# --- TEST CASES --- #

ICs = [1] # [0,1]

# --- CODE VERSIONS --- #

versions = ['gt4py', 'numpy']
backends = ['numpy','cuda', 'gt:cpu_ifirst', 'gt:gpu'] # 'gt:cpu_ifirst' # "gt:cpu_kfirst" # "gt:gpu" # 'cuda', 
version_backends = ['numpy']+['gt4py-'+b for b in backends] 
print(version_backends)

# --- GEOMETRIES --- #

geometries = ['torus'] # 'sphere' or 'torus'

# --- OTHER --- #

save    = 1e17 # save only at the end
verbose = 500
folder = 'data/performance/'

def plot_n_time(wall_times_by_vb, name='', title=''):
    
    plt.figure()
    
    for vb in list(wall_times_by_vb.keys()):
        plt.plot(ns, wall_times_by_vb[vb],'-x',label=vb)
    
    plt.legend()
    plt.title(title)
    plt.xlabel('n = nx*ny')
    plt.ylabel('wall clock time [s]')
    plt.tight_layout()
    plt.savefig(f'{folder}performance_{name}.png')

# --- WARM UP CACHES --- #
### TEST !!! ###
'''print('\nWarming up caches...\n')
_ = driver(
    TEST=True,
    PLOT=False,
    version = 'numpy',
    backend = '',
    geometry = 'sphere',
    IC = ICs[0],
    M = nxs[-1],
    N = nys[-1],
    verbose = 1e10,
    save = 1e10,
)
_ = driver(
    TEST=True,
    PLOT=False,
    version = 'gt4py',
    backend = 'cuda',
    geometry = 'sphere',
    IC = ICs[0],
    M = nxs[-1],
    N = nys[-1],
    verbose = 1e10,
    save = 1e10,
)
print('\nWarming up caches... Done.\n')
'''
# --- LOOP: SEPARATE TESTS --- #

for geometry in geometries:
    for IC in ICs:
        name = f'{geometry}_IC{IC}'
        
        # --- INNER LOOP: PLOTTED JOINTLY --- #
        
        wall_times = {
            vb:[] for vb in version_backends
        }
        for vb in version_backends:

            version = vb[:5]
            if version=='numpy':
                backend = 'numpy'
            else:
                backend = vb[6:]

            for nx,ny in zip(nxs,nys):

                wall_time = driver(
                    TEST=False,
                    PLOT=False,
                    version = version,
                    backend = backend,
                    geometry = geometry,
                    IC = IC,
                    T = 1,
                    M = nx,
                    N = ny,
                    verbose = verbose,
                    save = save,
                    folder=folder
                )

                wall_times[vb].append(wall_time)
        
        # --- SAVE just in case --- #
        ### TEST !!! ###
        with open('performance_'+name+'.pckl', 'wb') as f:
            pickle.dump(wall_times, f)
        
        # --- PLOTTING --- #
        ### add: plot from pickle without running ###
        plot_n_time(wall_times,name=name,title=name)

        
