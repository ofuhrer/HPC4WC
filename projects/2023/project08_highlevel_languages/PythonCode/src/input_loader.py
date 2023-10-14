import numpy as np
from pandas import read_csv

def load_input():
    # this turns alpha into int(alpha) which is 0
    # return np.genfromtxt('../../universal_input/input_dimensions.csv', delimiter=',', skip_header=1, dtype=np.int32)
    data = read_csv("../../universal_input/input_dimensions.csv")

    # Extract and store rows as tuples
    rows_as_tuples = []
    for index, row in data.iterrows():
        row_tuple = (int(row['x_dim']), int(row['y_dim']), int(row['z_dim']), float(row['alpha']), int(row['num_iter']))
        rows_as_tuples.append(row_tuple)

    return rows_as_tuples

def generate_initial_array(dim_df):

    num_halo = 2
    nx = dim_df[0]
    ny = dim_df[1]
    nz = dim_df[2]

    arr = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
    arr[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0

    return arr