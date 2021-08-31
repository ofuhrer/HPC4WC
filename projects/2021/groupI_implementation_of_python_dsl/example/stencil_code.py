import numpy as np
import time

from toydsl.driver.driver import computation
from toydsl.frontend.language import Horizontal, Vertical, end, start
import matplotlib.pyplot as plt


@computation
def copy_stencil(out_field, in_field):
    with Vertical[start:end]:
        with Horizontal[start : end, start: end]:
            out_field[0, 0, 0] = in_field[0, 0, 0]


@computation
def vertical_blur(out_field, in_field):
    with Vertical[start+1:end-1]:
        with Horizontal[start : end, start: end]:
            out_field[0, 0, 0] = (in_field[0, 0, 1] + in_field[0, 0, 0] + in_field[0, 0, -1]) / 3


@computation
def lapoflap(out_field, in_field, tmp1_field):
    """
    out = in - 0.03 * laplace of laplace
    """
    with Vertical[start:end]:
        with Horizontal[start+1 : end-1, start+1: end-1]:
            tmp1_field[0, 0, 0] = -4.0 * in_field[0,0,0] + in_field[-1,0,0] + in_field[1,0,0] + in_field[0,-1,0] + in_field[0,1,0]
        with Horizontal[start+1 : end-1, start+1: end-1]:
            out_field[0, 0, 0] = in_field[0, 0, 0] - 0.03 * (-4.0 * tmp1_field[0,0,0] + tmp1_field[-1,0,0] + tmp1_field[1,0,0] + tmp1_field[0,-1,0] + tmp1_field[0,1,0])

def set_up_data():
    """
    Set up the input for the test example
    """
    i = [0, 64]
    j = [0, 128]
    k = [0, 128]
    shape = (i[-1], j[-1], k[-1])
    a = np.zeros(shape)
    a[:,j[-1]//5:4*(j[-1]//5),k[-1]//5:4*(k[-1]//5)]=1
    b = np.zeros(shape)
    c = np.zeros(shape)
    d = np.zeros(shape)
    return a, b, c, d, i, j, k


if __name__ == "__main__":
    input, output, tmp1, tmp2, i, j, k = set_up_data()

    num_runs = 1024

    plt.ioff()
    plt.imshow(input[input.shape[0] // 2, :, :], origin="lower")
    plt.colorbar()
    plt.savefig("in_field.png")
    plt.close()
    
    start = time.time_ns()
    for _ in range(num_runs):
        lapoflap(output, input,tmp1, i, j, k)
        input = output
    end = time.time_ns()
    print(output[:, :, 0].T)

    plt.imshow(output[output.shape[0] // 2, :, :], origin="lower")
    plt.colorbar()
    plt.savefig("out_field.png")
    plt.close()


    print("Called DSL function {} times in {} seconds".format(num_runs, (end-start)/(10**9)))
