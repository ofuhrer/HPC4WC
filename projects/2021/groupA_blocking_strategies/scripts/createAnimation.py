import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Default k value
k = 64

fig = plt.figure()
im = None
data = None
nx = -1

def update(curr_k):
    plt.title(f"K={curr_k:02}")
    im.set_data(data[curr_k*nx : nx*curr_k + nx,:])
    return im,

if len(sys.argv) < 3:
    print("createAnimation.py <[in]mat-file> <[out]mp4-file> [k=64]")
    exit(1)

data = np.loadtxt(sys.argv[1])

if len(sys.argv) == 4:  
    k = int(sys.argv[3])

nx = data.shape[0] // k

im = plt.imshow(data[32*nx: 33*nx, :], animated=True)

ani = animation.FuncAnimation(fig, update, range(k))
FFwriter = animation.FFMpegWriter(fps=5)
ani.save(sys.argv[2], writer=FFwriter)
