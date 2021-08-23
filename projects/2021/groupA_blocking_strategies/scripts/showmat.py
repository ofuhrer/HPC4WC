import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


if len(sys.argv) == 3:

    data = np.loadtxt(sys.argv[1])
    fig,ax = plt.subplots()
    ax.matshow(data)
    ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig("./plots/"+sys.argv[2])
    # plt.show()

else:
    print("showmat.py <[in]mat-file> <[out]png-file>")