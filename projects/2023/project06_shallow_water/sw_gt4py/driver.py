#!/usr/bin/env python3
"""
Driver for running NumPy implementation of a finite difference
solver for Shallow Water Equations on a Sphere (SWES).
"""

import os
import pickle
import math
from animation_v2 import make_animation
import numpy as np

# --- SETTINGS --- #

TEST = True
PLOT = True

# Solver version:
#	* numpy (NumPy version)
#   * gt4py (DSL version)
version = 'gt4py'

# Import
if (version == 'numpy'):
    import swes_numpy as SWES
elif (version == 'gt4py'):
    import swes_gt4py as SWES
    
# Initial condition:
#	* 0: sixth test case of Williamson's suite
#	* 1: second test case of Williamson's suite
IC = 0

# Simulation length (in days); better to use integer values.
# Suggested simulation length for Williamson's test cases:
T = 0.5 if TEST else 4

# Grid dimensions
M = 180
N = 90

# CFL number
CFL = 0.5

# Various solver settings:
#	* diffusion: take diffusion into account
diffusion = False

# Output settings:
#	* verbose: 	specify number of iterations between two consecutive output
#	* save:		specify number of iterations between two consecutive stored timesteps
verbose = 500
save = 500

# --- RUN THE SOLVER --- #

pb = SWES.Solver(T, M, N, IC, CFL, diffusion)
if (save > 0):
    t, phi, theta, h, u, v = pb.solve(verbose, save)
else:
    h, u, v = pb.solve(verbose, save)

# --- STORE THE SOLUTION --- #

if (save > 0):
    GRIDTOOLS_ROOT = "." #os.environ.get('GRIDTOOLS_ROOT')
    baseName = GRIDTOOLS_ROOT + '/data/swes-%s-%s-M%i-N%i-T%i-%i-' % (version, str(IC), M, N, T, diffusion)

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
        make_animation(baseName, what_to_plot='h')
        print('[animation.py] Plotting u...')
        make_animation(baseName, what_to_plot='u')
        print('[animation.py] Plotting v...')
        make_animation(baseName, what_to_plot='v')
        print('[animation.py] Done.')