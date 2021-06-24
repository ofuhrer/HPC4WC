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

    time_cp = []
    time_devito_gpu = []
    time_gt4py_gpu = []
    time_gt4py_cpu = []

    print('time GPU')
    for nx in x:
        print(f'nx = {nx}')    
        timing_gpu = benchmark(nx, gpu=True)
        time_cp.append(timing_gpu[0])
        time_devito_gpu.append(timing_gpu[1])
        time_gt4py_gpu.append(timing_gpu[2]) 
        time_gt4py_cpu.append(timing_gpu[3]) 
            
    fig,ax = plt.subplots(figsize=(10,6))
    ax.tick_params(direction='in')
    ax.set(xlabel='grid size', ylabel='time [s]',
           title='timing 3d heat equation on GPU')
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log')
    plt.plot(x, time_cp, label='cupy')
    plt.plot(x, time_devito_gpu, label='devito_gpu')
    plt.plot(x, time_gt4py_gpu, label='gt4py_gpu')
    #plt.plot(x, time_gt4py_cpu, label='gt4py_cpu')
    ax.legend()
    fig.savefig('gpu.png')
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.tick_params(direction='in')
    ax.set(xlabel='grid size', ylabel='time [s]',
           title='timing 3d heat equation on GPU')
    ax.set_xscale('log', basex=2)
    plt.plot(x, time_cp, label='cupy')
    plt.plot(x, time_devito_gpu, label='devito_gpu')
    plt.plot(x, time_gt4py_gpu, label='gt4py_gpu')
    #plt.plot(x, time_gt4py_cpu, label='gt4py_cpu')
    
    ax.legend()
    fig.savefig('gpu_comp.png')

    
if __name__ == '__main__':
    main()