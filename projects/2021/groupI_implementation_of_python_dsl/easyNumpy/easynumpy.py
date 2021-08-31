import numpy as np
import time

import matplotlib.pyplot as plt

def copy_stencil(out_field, in_field, i, j, k):
    out_field[:, :, :] = in_field[:, :, :]

def vertical_blur(out_field, in_field, i, j, k):
    for z in range(i[0]+1,i[-1]-1):
        out_field[z, :, :] = (in_field[z+1, :, :] + in_field[z, :, :] + in_field[z-1, :, :]) / 3

def lapoflap(out_field, in_field, tmp1_field, tmp2_field, i, j, k):
    """
    out = in - 0.03 * laplace of laplace
    """
    ib = k[0]+1
    ie = k[-1]-1
    jb = j[0]+1
    je = j[-1]-1

    tmp1_field[:, jb:je, ib:ie] = - 4. * in_field[:, jb:je, ib:ie]  \
        + in_field[:, jb:je, ib - 1:ie - 1] + in_field[:, jb:je, ib + 1:ie + 1]  \
        + in_field[:, jb - 1:je - 1, ib:ie] + in_field[:, jb + 1:je + 1, ib:ie]

    tmp2_field[:, jb:je, ib:ie] = - 4. * tmp1_field[:, jb:je, ib:ie]  \
        + tmp1_field[:, jb:je, ib - 1:ie - 1] + tmp1_field[:, jb:je, ib + 1:ie + 1]  \
        + tmp1_field[:, jb - 1:je - 1, ib:ie] + tmp1_field[:, jb + 1:je + 1, ib:ie]

    out_field[:, :, :] = in_field[:, :, :] - 0.03 * tmp2_field[:, :, :]
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
    # plt.imshow(input[:, :, input.shape[0] // 2], origin="lower")
    plt.colorbar()
    plt.savefig("in_field.png")
    plt.close()

    cpp_times = []
    start = time.time_ns()
    for _ in range(num_runs):
        #cpp_time = copy_stencil(output, input, i, j, k)
        # vertical_blur(output, input, i, j, k)
        lapoflap(output, input,tmp1,tmp2, i, j, k)
        input = output
    end = time.time_ns()
    # Using this inupt, we expect the output of b to be
    # [[2. 2. 2. 2. 0.]
    #  [2. 2. 2. 2. 0.]
    #  [2. 2. 2. 2. 0.]
    #  [2. 2. 2. 2. 0.]
    #  [1. 1. 1. 1. 0.]]
    print(output[:, :, 0].T)

    plt.imshow(output[output.shape[0] // 2, :, :], origin="lower")
    # plt.imshow(output[:, :, output.shape[0] // 2], origin="lower")
    plt.colorbar()
    plt.savefig("out_field.png")
    plt.close()


    print("Called DSL function {} times in {} seconds".format(num_runs, (end-start)/(10**9)))
    #print("Measured times inside DSL function {} cycles".format(np.mean(cpp_times)))
