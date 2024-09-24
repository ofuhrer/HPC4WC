from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_field_from_csv(filename):
    nx, ny, nz = pd.read_csv(filename, nrows=1, dtype=int, header=None).to_numpy().reshape(-1)
    data = pd.read_csv(filename, skiprows=1, dtype=np.float64, header=None).to_numpy().reshape(-1)
    return np.reshape(data, (nz, ny, nx))


def read_field_from_npy(filename):
    return np.load(filename)


def plot_results():

    if Path('in_field.npy').is_file():
        in_field = read_field_from_npy('in_field.npy')
    elif Path('in_field.csv').is_file():
        in_field = read_field_from_csv('in_field.csv')
    else:
        raise Exception('Missing in_field.npy/in_field.csv.')

    if Path('out_field.npy').is_file():
        out_field = read_field_from_npy('out_field.npy')
    elif Path('out_field.csv').is_file():
        out_field = read_field_from_csv('out_field.csv')
    else:
        raise Exception('Missing out_field.npy/out_field.csv.')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    im1 = axs[0].imshow(in_field[in_field.shape[0] // 2, :, :], origin='lower', vmin=-0.1, vmax=1.1)
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title('Initial condition')

    im2 = axs[1].imshow(out_field[out_field.shape[0] // 2, :, :], origin='lower', vmin=-0.1, vmax=1.1)
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title('Final result')

    plt.savefig('plot.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print('NOTE: Script should be called from a working directory (e.g., \'cuda/\').')
    plot_results()
