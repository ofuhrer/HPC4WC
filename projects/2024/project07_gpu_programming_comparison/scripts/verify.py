import sys
from os.path import join
from pathlib import Path
from prettytable import PrettyTable
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pandas as pd


def read_field_from_csv(filename):
    nx, ny, nz = pd.read_csv(filename, nrows=1, dtype=int, header=None).to_numpy().reshape(-1)
    data = pd.read_csv(filename, skiprows=1, dtype=np.float64, header=None).to_numpy().reshape(-1)
    return np.reshape(data, (nz, ny, nx))


def read_field_from_npy(filename):
    return np.load(filename)


def read_file(prefix):

    in_field_npy = join(prefix, 'in_field.npy')
    in_field_csv = join(prefix, 'in_field.csv')
    out_field_npy = join(prefix, 'out_field.npy')
    out_field_csv = join(prefix, 'out_field.csv')

    if Path(in_field_npy).is_file():
        in_field = read_field_from_npy(in_field_npy)
    elif Path(in_field_csv).is_file():
        in_field = read_field_from_csv(in_field_csv)
    else:
        raise Exception(f'Missing input file in \"{prefix}\".')

    if Path(out_field_npy).is_file():
        out_field = read_field_from_npy(out_field_npy)
    elif Path(out_field_csv).is_file():
        out_field = read_field_from_csv(out_field_csv)
    else:
        raise Exception(f'Missing output file in \"{prefix}\".')

    return in_field, out_field


def verify(prefix1, prefix2):

    in_field1, out_field1 = read_file(prefix1)
    in_field2, out_field2 = read_file(prefix2)

    in_res = in_field2 - in_field1
    out_res = out_field2 - out_field1

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    im1 = axs[0].imshow(in_res[in_res.shape[0] // 2, :, :], origin='lower')
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title('Error (Initial condition)')

    im2 = axs[1].imshow(out_res[out_res.shape[0] // 2, :, :], origin='lower')
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title('Error (Final result)')

    plt.savefig(f'error_{prefix1}_{prefix2}.png', bbox_inches='tight')

    in_res = in_res.reshape(-1)
    out_res = out_res.reshape(-1)

    in_l1 = linalg.norm(in_res, 1)
    in_l2 = linalg.norm(in_res, 2)
    in_li = linalg.norm(in_res, np.inf)

    out_l1 = linalg.norm(out_res, 1)
    out_l2 = linalg.norm(out_res, 2)
    out_li = linalg.norm(out_res, np.inf)

    table = PrettyTable()
    table.field_names = ['', 'Input', 'Output']
    table.add_row(['L1', in_l1, out_l1])
    table.add_row(['L2', in_l2, out_l2])
    table.add_row(['L∞', in_li, out_li])

    print(table)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 3:
        verify(sys.argv[1], sys.argv[2])
    else:
        print('Input: python verify.py <prefix1> <prefix2>')
