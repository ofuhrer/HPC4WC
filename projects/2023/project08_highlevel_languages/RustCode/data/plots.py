import numpy as np
import matplotlib.pyplot as plt
import sys

names = []
arrays = []
for i in range(1,len(sys.argv)):
    name = str(sys.argv[i])
    names.append(name)
    array = np.genfromtxt('./data/' + name + '.csv', delimiter=',')
    arrays.append(array)

print(names)

def fields2d(arrays: list, names: list):
    """
    Plots csv files using imshow. Indended usage is calling this file from cmd line with
    the names of the csv files (without .csv extension) as argument. E.g.
    py3 data/plots.py in_field out_field
    """
    for (array, name) in zip(arrays, names):
        plt.imshow(array[:, :], origin="lower")
        plt.colorbar()
        plt.savefig("./data/" + name + "_l.png")
        plt.cla(); plt.clf()

        # plt.imshow(array[:, :], origin="upper")
        # plt.colorbar()
        # plt.savefig("./data/" + name + "_u.png")
        # plt.cla(); plt.clf()

fields2d(arrays, names)


# =============================================================================
# old stuff, should work but you can only always plot exactly two csv files
# first_field_str = str(sys.argv[1])
# second_field_str = str(sys.argv[2])

# # relative paths are set up to be run from RustCode/
# # i.e. run like python3 ./data/plots.py in_field zero_field if you want to plot in_field.csv and zero_field.csv
# in_field = np.genfromtxt('./data/' + first_field_str + '.csv', delimiter=',')
# out_field = np.genfromtxt('./data/' + second_field_str + '.csv', delimiter=',')

def save_as(x, name:str ):
    with open("./data/" + name + ".csv", "wb") as f:
        np.savetxt(f, x, fmt='%0.2f', delimiter=",")

def normalize_array(x):
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm

# def fields3d(in_field, out_field):
#     # plt.imshow(in_field[in_field.shape[0] // 2, :, :], origin="lower")
#     # plt.colorbar()
#     # plt.savefig("first_arg.png")

#     # plt.imshow(out_field[out_field.shape[0] // 2, :, :], origin="lower")
#     # plt.colorbar()
#     # plt.savefig("second_arg.png")
#     pass

# def fields2d(in_field, out_field):
#     # from skimage import io
#     # save_as(in_field, "np_in_field")
#     # in_field = normalize_array(in_field)
#     # save_as(in_field, "np_in_fieldnorm")
#     plt.imshow(in_field[:, :], origin="upper")
#     plt.colorbar()
#     plt.savefig("./data/" + first_field_str + ".png")

#     plt.cla(); plt.clf()

#     plt.imshow(out_field[:, :], origin="upper")
#     plt.colorbar()
#     plt.savefig("./data/" + second_field_str + ".png")

# fields2d(in_field, out_field)

