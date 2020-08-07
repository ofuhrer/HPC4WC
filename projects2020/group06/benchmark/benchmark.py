import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt

from heat3d_np import heat3d_np
from heat3d_cp import heat3d_cp

from heat3d_devito import heat3d_devito
from heat3d_gt4py import heat3d_gt4py

def benchmark(nx, gpu=False):
    # const
    alpha = 19.
    Tcool, Thot = 300., 400.
    x = y = z = 2.

    # var
    ny = nx
    nz = nx
    nt = 300

    dx = dy = dz = x / (nx - 1)
    dt = alpha/10000 * (1/dx**2 + 1/dy**2 + 1/dz**2)**(-1)

    in_field = Tcool*np.ones((nx,ny,nz))
    in_field[nx//4:3*nx//4, ny//4:3*ny//4, nz//4:3*nz//4] = Thot

    if gpu == True:
        time_cp = heat3d_cp(alpha, Tcool, Thot, dx, dy, dz, dt, nx, ny, nz, nt, result='time')
        time_devito_gpu = heat3d_devito(in_field, alpha, Tcool, dt, nt, platform='nvidiaX', result='time')
        time_gt4py_gpu = heat3d_gt4py(in_field, alpha, Tcool, dx, dy, dz, dt, nt, backend='gtcuda', result='time')
        time_gt4py_cpu = heat3d_gt4py(in_field, alpha, Tcool, dx, dy, dz, dt, nt, backend='gtx86', result='time')

        return time_cp, time_devito_gpu, time_gt4py_gpu, time_gt4py_cpu

    else:
        time_np = heat3d_np(in_field, alpha, Tcool, dx, dy, dz, dt, nt, result='time')    
        time_devito = heat3d_devito(in_field, alpha, Tcool, dt, nt, result='time')
        time_gt4py = heat3d_gt4py(in_field, alpha, Tcool, dx, dy, dz, dt, nt, backend='gtx86', result='time')

        return time_np, time_devito, time_gt4py
