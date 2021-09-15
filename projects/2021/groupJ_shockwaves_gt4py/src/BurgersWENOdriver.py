import numpy as np
from BurgersWENOcalc import *
from BurgersWENO import *
import matplotlib.pyplot as plt
from matplotlib import cm

import os
import click
import timeit


@click.command()
@click.option("--m", type=int, default=3, required=False,
              help="m(2m-1)th order of ENO(WENO) approximation")
@click.option(
    "--nx",
    type=int,
    required=True,
    help="Number of gridpoints in x-direction")
@click.option(
    "--ny",
    type=int,
    required=True,
    help="Number of gridpoints in y-direction")
@click.option(
    "--nz",
    type=int,
    required=True,
    help="Number of gridpoints in z-direction")
@click.option(
    "--gt4py",
    type=bool,
    required=False,
    default=True,
    help="Use of GT4Py?")
@click.option(
    "--tmp",
    type=bool,
    required=False,
    default=True,
    help="Use of temporary variables?")
@click.option(
    "--backend",
    type=str,
    required=False,
    default="numpy",
    help="NumPy or GT4Py backends?")
@click.option(
    "--plot",
    type=bool,
    required=False,
    default=False,
    help="Make a plot of the result?")
@click.option(
    "--time",
    type=bool,
    required=False,
    default=False,
    help="Write the elapsed time?")
    
    
class BurgersWENO5():
    """
    The 5th-order weighted essentially non-oscillatory (WENO) scheme (WENO5) for spatial distribution
    and the 3rd-order 3-stage Strong Stability Preserving Runge Kutta (SSP-RK(3,3)) as time integration
    to solve Bureger's equation.
    Implementations according to Hesthaven (2018)
    """

    def initial_condition(self, x):
        """Compute initial condition"""
        return np.sin(2 * np.pi * x)


    def LegendreGQ(self, m):
        """Compute the m'th order Legendre Gauss quadratur e point s, x, and weight s, w"""

        if m == 0:
            x[0] = 0
            w[0] = 2

        # Form symmetric matrix from recurrence
        J = np.zeros(m + 1)
        h1 = 2 * np.linspace(0, m, m + 1)
        J = np.diag(2.0 / (h1[0:m] + 2) * np.sqrt(np.arange(1, m + 1) * (np.arange(1, m + 1)) * \
                            (np.arange(1, m + 1)) * (np.arange(1, m + 1)) / (h1[0:m] + 1) / (h1[0:m] + 3)), 1)
        J[0, 0] = 0
        J = J + np.transpose(J)

        # Compute quadrature by eigenvalue solve
        D, V = np.linalg.eig(J)
        x = D
        w = 2 * V[0, :] ** 2

        return x, w


    def harmonic(self, x):
        # special.psi(x + 1) + np.euler_gamma
        if x == 0:
            return 0
        elif x < 2:
            return 1
        else:
            return 1 / x + (self.harmonic(x - 1))


    def ReconstructWeights(self, m, r):
        """
        Compute weights c_ir for reconstruction v{j+1/2}=\sum{j=0}ˆ{m−1} c{i_r} v_{i−r+j}
        with m=order  and r=shift (−1<=r<=m−1).
        """

        def fh(s, m):
            return (-1) ** (s + m) * prod(s) * prod(m - s)

        c = np.zeros(m)

        for i in range(0, m):
            q = np.linspace(i + 1, m, m - i)
            for q in range(i + 1, m + 1):
                if q != r + 1:
                    c[i] += fh(r + 1, m) / fh(q, m) / (r + 1 - q)
                else:
                    c[i] -= self.harmonic(m - r - 1) - self.harmonic(r + 1)

        return c


    def LinearWeights(self, m, r0):
        """Compute linear weights for maximum accuracy 2m-1, using stencil shifted $r0=-1,0$ points upwind"""

        A = np.zeros((m, m))
        b = np.zeros(m)

        # Setup linear system for coefficients
        for i in range(1, m + 1):
            col = self.ReconstructWeights(m, i - 1 + r0)
            A[0: (m + 1 - i), i - 1] = col[i - 1: m]

        # Setup righthand side for maximum accuracy and solve
        crhs = self.ReconstructWeights(2 * m - 1, m - 1 + r0)
        b = crhs[m - 1: 2 * m - 1]
        d = np.linalg.solve(A, b)
        return d


    def betarcalc(self, x, m, mxGQ, mwGQ):
        """
        Compute matrix to allow evaluation of smoothness indicator in
        WENO based on stencil [ x ] of length m+1.
        Returns sum of operators for l =1.m-1
        """

        # Evaluate Lagrange polynomials
        cw = self.lagrangeweights(x)

        # Compute error matrix for l =1..m-1
        errmat = np.zeros((m, m))
        for l in range(2, m + 1):
            # Evaluate coefficients for derivative of Lagrange polynomial
            dw = np.zeros((m, m - l + 1))
            for k in range(0, m - l + 1):
                for q in range(0, m):
                    dw[q, k] = sum(cw[(q + 1): m + 1, k + l])

            # Evaluate entries in matrix for order l
            Qmat = np.zeros((m, m))
            for p in range(0, m):
                for q in range(0, m):
                    D = matmul(dw[q, :], dw[p, :])
                    Qmat[p, q] = Qcalc(D, m, l, mxGQ, mwGQ)

            errmat = errmat + Qmat

        return errmat


    def lagrangeweights(self, x):
        """
        Compute weights for Taylor expansion of Lagrange polynomial based on x and evaluated at 0.
        Method due to Fornberg (SIAM Review, 1998, 685−691)
        """

        npp = x.shape[0]
        cw = np.zeros((npp, npp))
        cw[0, 0] = 1.0
        c1 = 1.0
        c4 = x[0]

        for i in range(2, npp + 1):
            mn = min(i, npp - 1) + 1
            c2 = 1.0
            c5 = c4
            c4 = x[i - 1]
            for j in range(1, i):
                c3 = x[i - 1] - x[j - 1]
                c2 = c2 * c3
                if j == i - 1:
                    for k in range(mn, 1, -1):
                        cw[i - 1, k - 1] = (
                            c1 * ((k - 1) * cw[i - 2, k - 2] -
                                  c5 * cw[i - 2, k - 1]) / c2
                        )
                    cw[i - 1, 0] = -c1 * c5 * cw[i - 2, 0] / c2
                for k in range(mn, 1, -1):
                    cw[j - 1, k - 1] = (
                        c4 * cw[j - 1, k - 1] - (k - 1) * cw[j - 1, k - 2]
                    ) / c3
                cw[j - 1, 0] = c4 * cw[j - 1, 0] / c3
            c1 = c2
        return cw


    def __init__(self, m, nx, ny, nz, gt4py, tmp, backend, plot, time):
        """
        The 5th-order weighted essentially non-oscillatory (WENO) scheme (WENO5) for spatial distribution
        and the 3rd-order 3-stage Strong Stability Preserving Runge Kutta (SSP-RK(3,3)) as time integration
        to solve Bureger's equation.
        Implementations according to Hesthaven (2018)
        """

        # Set WENO parameters
        p = 1
        eps = 1e-6

        # Set problem parameters
        L = 1
        FinalTime = 0.15
        CFL = 0.9
        BCl = "P"
        BCr = "P"
        dx = L / nx

        # Define domain and initial conditions
        x = np.linspace(0, L, nx + 1)
        y = np.linspace(0, (ny + 1) * dx, ny + 1)

        # Compute cell averages using Legendre Gauss rule of order NGQ
        u = np.zeros((nx + 1, ny + 1))

        # Use sin initial condition
        NGQ = 10
        xGQ, wGQ = self.LegendreGQ(NGQ)
        for j in range(0, u.shape[1]):
            for i in range(1, NGQ + 2):
                u[:, j] = u[:, j] + wGQ[i - 1] * \
                    self.initial_condition(x + xGQ[i - 1] * dx / 2)

        u = u / 2

        N = u.shape[0]

        # Solve Problem
        dw = self.LinearWeights(m, 0)  # [0.3, 0.6, 0.1]
        dwf = np.flipud(dw)  # [0.1, 0.6, 0.3]
        mxGQ, mwGQ = self.LegendreGQ(m)

        # Initialize reconstruction weights
        Crec = np.zeros((m + 1, m))

        for r in range(-1, m):
            Crec[r + 1, :] = self.ReconstructWeights(m, r)

        # Compute smoothness indicator matrices
        beta = np.zeros((m, m, m))
        for r in range(0, m):
            xl = -1 / 2 + np.linspace(-r, m - r, m + 1)
            beta[:, :, r] = self.betarcalc(xl, m, mxGQ, mwGQ)

        tic = timeit.default_timer()
        for j in range(0, u.shape[1]):
            u[:, j] = BurgersWENO(x, u[:, j], dx, m, CFL, FinalTime, dw, dwf,
                        Crec, beta, mxGQ, mwGQ, N, p, eps, gt4py, backend, tmp)
        toc = timeit.default_timer()

        if time:
            f = open('../output/log', 'a')
            # Write header
            if os.stat('../output/log').st_size == 0:
                f.write(
                    "GT4Py   tmp   Backend         nx      ny   nz   Elapsed" +
                    '\n')
            f.write("{:>5}".format(str(gt4py)) + "  ")
            f.write("{:>5}".format(str(tmp)) + "  ")
            f.write("{:10s}".format(backend) + "  ")
            f.write("{:>6}  {:>6}  {:>3}".format(nx, ny, nz) + "  ")
            f.write("{:.6f}".format(toc - tic) + '\n')
            f.close()

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            x, y = np.meshgrid(x, y, indexing="ij")
            v = ax.plot_surface(x, y, u, rstride=1, cstride=1,
                                cmap=cm.coolwarm, antialiased=False)
            cb = fig.colorbar(v, shrink=0.5, pad=0.075)
            cb.set_label("u")
            plt.savefig("../output/WENO2D_" + backend + '.png')


if __name__ == "__main__":
    WENO5 = BurgersWENO5()
