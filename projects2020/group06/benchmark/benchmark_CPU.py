import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt

from heat3d_np import heat3d_np
from heat3d_cp import heat3d_cp

from heat3d_devito import heat3d_devito
from heat3d_gt4py import heat3d_gt4py

from benchmark import benchmark

def main():
    
    x = []
    for n in range(3, 9+1):
        x.append(2 ** n)

    time_np = []
    time_devito = []
    time_gt4py = []
    
    print('time CPU')
    for nx in x:
        print(f'nx = {nx}')    
        timing = benchmark(nx)
        time_np.append(timing[0])
        time_devito.append(timing[1])
        print(timing[1])
        time_gt4py.append(timing[2])  
 
    fig, ax = plt.subplots(figsize=(10,6))
    ax.tick_params(direction='in')
    ax.set(xlabel='grid size', ylabel='time [s]',
           title='timing 3d heat equation on CPU')
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log')
    plt.plot(x, time_np, label='numpy')
    plt.plot(x, time_devito, label='devito')
    plt.plot(x, time_gt4py, label='gt4py')
    
    ax.legend()
    fig.savefig('cpu.png')
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.tick_params(direction='in')
    ax.set(xlabel='grid size', ylabel='time [s]',
           title='timing 3d heat equation on CPU')
    ax.set_xscale('log', basex=2)
    plt.plot(x, time_np, label='numpy')
    plt.plot(x, time_devito, label='devito')
    plt.plot(x, time_gt4py, label='gt4py')
    
    ax.legend()
    fig.savefig('cpu_comp.png')
    
if __name__ == '__main__':
    main()