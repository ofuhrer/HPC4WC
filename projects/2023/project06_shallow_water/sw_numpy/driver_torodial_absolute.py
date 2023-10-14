#!/usr/bin/env python3
"""
Driver for running NumPy implementation of a finite difference
solver for Shallow Water Equations on a Sphere (SWES).
"""

import os
import pickle
import math

# --- SETTINGS --- #

# Solver version:
#	* numpy (NumPy version)
#   * gt4py (DSL version)
version = 'numpy'

# Import
if (version == 'numpy'):
    import swes_numpy_torodial_absolute as SWES

# Initial condition:
#	* 0: sixth test case of Williamson's suite
#	* 1: second test case of Williamson's suite
IC = 1

# Simulation length (in days); better to use integer values.
# Suggested simulation length for Williamson's test cases:
T = 50

# Grid dimensions
M = 180
N = 90

# CFL number
CFL = 0.5

# Various solver settings:
#	* diffusion: take diffusion into account
diffusion = True

# Output settings:
#	* verbose: 	specify number of iterations between two consecutive output
#	* save:		specify number of iterations between two consecutive stored timesteps
verbose = 100#500
save = 100#500

# --- RUN THE SOLVER --- #

pb = SWES.Solver(T, M, N, IC, CFL, diffusion)
if (save > 0):
    t, phi, theta, h, u, v = pb.solve(verbose, save)
else:
    h, u, v = pb.solve(verbose, save)

# --- STORE THE SOLUTION --- #

if (save > 0):
    # try to get gridtools_root (does not work on all systems)
    root = os.environ.get('GRIDTOOLS_ROOT')
    
    # if it does not work, set root to parent directory of script location
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        print('Warning: GRIDTOOLS_ROOT not set, using %s as root directory for data output' % root)
        # make sure the data directory exists
        os.makedirs(root + '/data',exist_ok=True)

    baseName = root + '/data/swes-%s-%s-M%i-N%i-T%i-%i-' % (version, str(IC), M, N, T, diffusion)

    # Save h
    with open(baseName + 'h', 'wb') as f:
        pickle.dump([M, N, t, phi, theta, h], f, protocol = 2)

    # Save u
    with open(baseName + 'u', 'wb') as f:
        pickle.dump([M, N, t, phi, theta, u], f, protocol = 2)

    # Save v
    with open(baseName + 'v', 'wb') as f:
        pickle.dump([M, N, t, phi, theta, v], f, protocol = 2)
