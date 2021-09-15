import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Plot performance from log using with different settings
"""

df = pd.read_csv('../output/log', delimiter=r"\s+")

plt.figure()

for gt4py in [False, True]:
    # NumPy
    if not gt4py:
        t = []    # elapsed time
        err = []  # standard deviation
        g = []    # number of grid points

        for nx in [30, 50, 100, 200, 500, 1000, 1500]:
            d = df.loc[(df['nx'] == nx) & (df['Backend'] ==
                                           'numpy') & (df['GT4Py'] == False)]

            t.append(d['Elapsed'].mean())
            err.append(d['Elapsed'].std())
            g.append((d['nx'] * d['ny'] * d['nz']).mean())

        plt.errorbar(g, t, yerr=err, label='NumPy')

    # GT4Py
    else:
        for tmp in [True, False]:
            # Use temporary variables
            if tmp:
                for backend in ['numpy', 'gtmc', 'gtx86', 'gtc:dace', 'gtcuda']:
                    t = []    # elapsed time
                    err = []  # standard deviation
                    g = []    # number of grid points

                    for nx in [30, 50, 100, 200, 500, 1000, 1500]:
                        d = df.loc[(df['nx'] == nx) & (
                            df['Backend'] == backend) & (df['GT4Py']) & (df['tmp'])]

                        t.append(d['Elapsed'].mean())
                        err.append(d['Elapsed'].std())
                        g.append((d['nx'] * d['ny'] * d['nz']).mean())

                    plt.errorbar(g, t, yerr=err, label=backend + ' backend')

            # Without temporary variables
            else:
                t = []    # elapsed time
                err = []  # standard deviation
                g = []    # number of grid points

                for nx in [30, 50, 100, 200, 500, 1000, 1500]:
                    d = df.loc[(df['nx'] == nx) & (
                        df['Backend'] == 'numpy') & (df['tmp'] == False)]

                    t.append(d['Elapsed'].mean())
                    err.append(d['Elapsed'].std())
                    g.append((d['nx'] * d['ny'] * d['nz']).mean())

                plt.errorbar(g, t, yerr=err, label='numpy' + ' backend (4D)')

plt.xscale('log', base=10, nonpositive='clip')
plt.yscale('log', base=10, nonpositive='clip')

plt.xlabel('# grid points')
plt.ylabel('Elapsed time (s)')
plt.legend(frameon=False)
plt.savefig("../output/WENO2D_performance", dpi=500)
