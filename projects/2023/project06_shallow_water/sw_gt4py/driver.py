#!/usr/bin/env python3
"""
Driver for running NumPy implementation of a finite difference
solver for Shallow Water Equations on a Sphere (SWES).
"""

import os
import pickle
import math
import numpy as np

from animation import make_animation

def driver(
    TEST=False,
    PLOT=True,
    make_gif=False,
    version = 'gt4py', # 'numpy' # 
    backend = 'conda', # 'numpy' # 'gt:cpu_ifirst' # "gt:cpu_kfirst" # "gt:gpu" #
    geometry = 'sphere', # 'torus'
    IC = 0, # 1
    T = 4,
    M = 181, # nx
    N = 90, # ny
    diffusion = True, # nu = 5e5 (earth)
    verbose=500,
    save=500,
    folder = "/data/", # GRIDTOOLS_ROOT = os.environ.get('GRIDTOOLS_ROOT')
):
    
    if version == 'numpy':
        backend == ''
        
    print(f'\n[driver.py] Starting SWES simulation')
    print(f'in {version} version')
    if version=='gt4py':
        print(f'with {backend} backend')
    print(f'on the {geometry}.\n')
          
    # --- SETTINGS --- #

    # Solver version:
    #	* numpy (NumPy version)
    #   * gt4py (DSL version)
    
    # Import
    if (geometry == 'sphere'):
        if (version == 'numpy'):
            import swes_numpy as SWES
        elif (version == 'gt4py'):
            import swes_gt4py as SWES
    elif (geometry == 'torus'):
        if (version == 'numpy'):
            print('Not implemented.')
            import swes_torus_numpy as SWES
        elif (version == 'gt4py'):
            print('Not implemented.')
            import swes_torus_gt4py as SWES


    # Initial condition:
    #	* 0: sixth test case of Williamson's suite
    #	* 1: second test case of Williamson's suite

    # Simulation length (in days); better to use integer values.
    # Suggested simulation length for Williamson's test cases:
    T = 0.25 if TEST else T

    # CFL number
    CFL = 0.5

    # Various solver settings:
    #	* diffusion: take diffusion into account

    # Output settings:
    #	* verbose: 	specify number of iterations between two consecutive output
    #	* save:		specify number of iterations between two consecutive stored timesteps
    verbose = 50 if TEST else verbose
    save = 50 if TEST else save

    # --- RUN THE SOLVER --- #
    if version == 'gt4py':
        pb = SWES.Solver(T, M, N, IC, CFL, diffusion, backend=backend)
    else:
        pb = SWES.Solver(T, M, N, IC, CFL, diffusion)
    if (save > 0):
        wall_time, t, phi, theta, h, u, v = pb.solve(verbose, save)
    else:
        wall_time, h, u, v = pb.solve(verbose, save)

    # --- STORE THE SOLUTION --- #

    if (save > 0):
        baseName = folder + 'swes-%s-%s-%s-%s-M%i-N%i-T%i-%i-' % (geometry, version, backend, str(IC), M, N, T, diffusion)

        # Save h
        with open(baseName + 'h', 'wb') as f:
            pickle.dump([M, N, t, phi, theta, h], f, protocol = 2)

        # Save u
        with open(baseName + 'u', 'wb') as f:
            pickle.dump([M, N, t, phi, theta, u], f, protocol = 2)

        # Save v
        with open(baseName + 'v', 'wb') as f:
            pickle.dump([M, N, t, phi, theta, v], f, protocol = 2)

        print('\nDone computing.\n')

        # --- PLOT THE SOLUTION --- #

        if PLOT:
            print('[animation.py] Plotting h...')
            make_animation(baseName, what_to_plot='h',make_gif=make_gif)
            print('[animation.py] Plotting u...')
            make_animation(baseName, what_to_plot='u',make_gif=make_gif)
            print('[animation.py] Plotting v...')
            make_animation(baseName, what_to_plot='v',make_gif=make_gif)
            print('[animation.py] Done.')
        
    print(f'\n[driver.py] Starting SWES simulation')
    print(f'in {version} version')
    if version=='gt4py':
        print(f'with {backend} backend')
    print(f'in {wall_time/60:8.2f} min.\n')
    
    return wall_time

if __name__ == '__main__':
    
    # --- some default run --- #
    
    driver(
        TEST=True,
        PLOT=False,
        make_gif = False,
        version = 'gt4py', # 'numpy' # 
        backend = 'cuda', # 'numpy', # 'gt:cpu_ifirst' # "gt:cpu_kfirst" # "gt:gpu" #
        geometry = 'sphere', # 'torus'
        IC = 0, # 1
        T = 4,
        M = 180, # nx
        N = 90, # ny
        diffusion = True, # nu = 5e5 (earth)
        verbose=500,
        save=500,
        folder = "/data/",
    )