import numpy as np

import gt4py as gt
import cupy as cp

from BurgersWENOcalc import *
from gt4py import gtscript
from gt4py.gtscript import FORWARD, PARALLEL, Field, computation, interval


def extend(x, u, dx, m, BCl, ul, BCr, ur):
    """
    Extend dependent and independent vectors (x,u), by m cells subject to
    approproate boundary conditions.
    BC = "D" - Dirichlet; BC = "N" - Neumann; BC = "P" - periodic
    ul/ur  BC value only active for Dirichlet BC
    """

    xl = min(x)
    xr = max(x)
    N = u.shape[0]
    xe = np.zeros((N + 2 * m))
    ue = np.zeros((N + 2 * m))
    q = np.arange(1, m + 1)

    # Extend x
    xe[m - q] = xl - q * dx
    xe[N + m + q - 1] = xr + q * dx
    xe[m: (N + m)] = x[0:N]

    # Periodic extension of u
    if (BCl == "P") | (BCr == "P"):
        ue[m - q] = u[N - q - 1]
        ue[N + m + q - 1] = u[q]
        ue[m: (N + m)] = u[0:N]

        return xe, ue

    # Left extension
    if BCl == "D":
        ue[m - q + 1] = u[q + 1] + 2 * ul
    else:
        ue[m - q + 1] = u[q + 1]

    # Right extension
    if BCr == "D":
        ue[N + m + q - 1] = u[N - q - 1] + 2 * ur
    else:
        ue[N + m + q - 1] = u[N - q - 1]

    ue[m: (N + m)] = u[0:N]

    return xe, ue


def BurgersWENO(x, u, dx, m, CFL, FinalTime, dw, dwf, Crec, beta, mxGQ, mwGQ, N, p, eps, gt4py, backend, tmp):
    """Integrate 1D Burgers equation until FinalTime using an WENO scheme and 3rd order SSP−RK method"""

    time = 0
    tstep = 0
    
    # Compile stencil
    if gt4py:
        if tmp:
            BurgersWENO5_2D = gtscript.stencil(backend=backend, definition=BurgersWENO2D)
            
            Crec_   = gt.storage.from_array(Crec[cp.newaxis, ...], backend, (0, 0, 0),  (N + 2, Crec.shape[0], Crec.shape[1]))
            beta0   = gt.storage.from_array(beta[cp.newaxis, ..., 0], backend, (0, 0, 0), (N + 2, m, m))
            beta1   = gt.storage.from_array(beta[cp.newaxis, ..., 1], backend, (0, 0, 0), (N + 2, m, m))
            beta2   = gt.storage.from_array(beta[cp.newaxis, ..., 2], backend, (0, 0, 0), (N + 2, m, m))

            dw_  = gt.storage.from_array(dw,  backend, (0,), (m,), mask=(False, False, True))   #1 
                    
        else:
            BurgersWENO5_2D = gtscript.stencil(backend=backend, definition=BurgersWENO2D_)

    # Integrate scheme
    while time < FinalTime:
        maxvel = np.amax(2 * abs(u))
        k = CFL * dx / maxvel
        if time + k > FinalTime:
            k = FinalTime - time

        # Extend data and assign boundary conditions
        xe, ue = extend(x, u, dx, m, "P", 2, "P", 0)

        if gt4py:
    
            xe = gt.storage.from_array(xe[..., np.newaxis, np.newaxis], backend, (0, 0, 0), (xe.shape[0], 1, 1))
            ue = gt.storage.from_array(ue[..., np.newaxis, np.newaxis], backend, (m - 1, 0, 0), (ue.shape[0], 1, 1))

            du1 = gt.storage.zeros(backend, (m, 0, 0), (N + 2 * m, 1, 1), float)
            du2 = gt.storage.zeros(backend, (m, 0, 0), (N + 2 * m, 1, 1), float)
            du3 = gt.storage.zeros(backend, (m, 0, 0), (N + 2 * m, 1, 1), float)

            u1 = gt.storage.zeros(backend, (m, 0, 0), (N + 2 * m, 1, 1), float)
            u2 = gt.storage.zeros(backend, (m, 0, 0), (N + 2 * m, 1, 1), float)
     
            u1_ = gt.storage.zeros(backend, (m, 0, 0), (N + 2 * m, 1, 1), float)
            u2_ = gt.storage.zeros(backend, (m, 0, 0), (N + 2 * m, 1, 1), float)

            um = gt.storage.zeros(backend, (0, 0, 0), (N + 2, 1, 1), float)
            up = gt.storage.zeros(backend, (0, 0, 0), (N + 2, 1, 1), float)
            
            k_  = gt.storage.from_array(k,  backend, (0,), (m,))   #1    

            if tmp:           
                
                BurgersWENO5_2D(du1, xe, ue, um, up, dx, m, Crec_, dw_, beta0, beta1, beta2, p, eps, maxvel)  
                if backend == 'gtcuda':
                    du1.synchronize()
                    du1 = du1.view(np.ndarray)
                    ue = ue.view(np.ndarray)
                u1 = ue + k * du1
                u1_ = gt.storage.from_array(u1[...], backend, (m-1, 0, 0), (u1.shape[0], 1, 1))
              
                BurgersWENO5_2D(du2, xe, u1_, um, up, dx, m, Crec_, dw_, beta0, beta1, beta2, p, eps, maxvel)
                if backend == 'gtcuda':
                    du2.synchronize()
                    du2 = du2.view(np.ndarray)
                u2 = (3 * ue + u1 + k * du2) / 4
                u2_ = gt.storage.from_array(u2[...], backend, (m-1, 0, 0), (u2.shape[0], 1, 1))

                BurgersWENO5_2D(du3, xe, u2_, um, up, dx, m, Crec_, dw_, beta0, beta1, beta2, p, eps, maxvel)
                if backend == 'gtcuda':
                    du3.synchronize()
                ue = (ue + 2 * u2_ + 2 * k * du3) / 3 
                ue.synchronize()      
                
            else:
                # Do not use temporary variables
                upl = gt.storage.zeros(
                    backend, (0, 0, 0, 0), (N + 2, 1, 1, m), float)
                uml = gt.storage.zeros(
                    backend, (0, 0, 0, 0), (N + 2, 1, 1, m), float)
                betar = gt.storage.zeros(
                    backend, (0, 0, 0, 0), (N + 2, 1, 1, m), float)
                alpham = gt.storage.zeros(
                    backend, (0, 0, 0, 0), (N + 2, 1, 1, m), float)
                alphap = gt.storage.zeros(
                    backend, (0, 0, 0, 0), (N + 2, 1, 1, m), float)

                # 3rd order SSP−RK method
                BurgersWENO5_2D(du1, xe, ue, um, up, upl, uml,
                               betar, alpham, alphap, dx, m, p, eps, maxvel)
                u1 = ue + k * du1

                BurgersWENO5_2D(du2, xe, u1, um, up, upl, uml,
                               betar, alpham, alphap, dx, m, p, eps, maxvel)
                u2 = (3 * ue + u1 + k * du2) / 4

                BurgersWENO5_2D(du3, xe, u2, um, up, upl, uml,
                               betar, alpham, alphap, dx, m, p, eps, maxvel)
                ue = (ue + 2 * u2 + 2 * k * du3) / 3
            
                
            u[...] = ue[m:-m, 0, 0]


        else:
            # 3rd order SSP−RK method
            rhsu = BurgersWENOrhs(xe, ue, dx, k, m, Crec,
                                  maxvel, dw, dwf, beta, N, p, eps)  # du1
            u1 = ue + k * rhsu

            rhsu = BurgersWENOrhs(xe, u1, dx, k, m, Crec,
                                  maxvel, dw, dwf, beta, N, p, eps)  # du2
            u2 = (3 * ue + u1 + k * rhsu) / 4

            rhsu = BurgersWENOrhs(xe, u2, dx, k, m, Crec,
                                  maxvel, dw, dwf, beta, N, p, eps)  # du3
            ue = (ue + 2 * u2 + 2 * k * rhsu) / 3

            u = ue[m:-m]

        time = time + k
        tstep = tstep + 1

    return u


@gtscript.function
def BurgersLF(u, v, lam, maxvel):
    """Evaluate global Lax Friedrich numerical flux for Burgers equation"""

    return (u ** 2 + v ** 2) / 2 - maxvel / 2 * (v - u)


def BurgersWENO2D(du:Field[float], xe:Field[float], ue: Field[float], um:Field[float], up:Field[float], dx: float, m:int, 
                  Crec:Field[float], dw:Field[gtscript.K, float],  
                  beta0:Field[float], beta1:Field[float],beta2:Field[float], 
                  p:int, eps:float, maxvel:float):
    with computation(PARALLEL), interval(0,1):
        if m == 1:
            um = ue
            up = ue
        else:
            # Compute um and up based on different stencils and smoothness indicators and alpha
            upl0 = Crec[0, 1, 0] * ue[0, 0, 0]  + Crec[0, 1, 1] * ue[1, 0, 0]  + Crec[0, 1, 2] * ue[2, 0, 0]
            upl1 = Crec[0, 2, 0] * ue[-1, 0, 0] + Crec[0, 2, 1] * ue[0, 0, 0]  + Crec[0, 2, 2] * ue[1, 0, 0]
            upl2 = Crec[0, 3, 0] * ue[-2, 0, 0] + Crec[0, 3, 1] * ue[-1, 0, 0] + Crec[0, 3, 2] * ue[0, 0, 0]

            uml0 = Crec[0, 0, 0] * ue[0, 0, 0]  + Crec[0, 0, 1] * ue[1, 0, 0]  + Crec[0, 0, 2] * ue[2, 0, 0]
            uml1 = Crec[0, 1, 0] * ue[-1, 0, 0] + Crec[0, 1, 1] * ue[0, 0, 0]  + Crec[0, 1, 2] * ue[1, 0, 0]
            uml2 = Crec[0, 2, 0] * ue[-2, 0, 0] + Crec[0, 2, 1] * ue[-1, 0, 0] + Crec[0, 2, 2] * ue[0, 0, 0]

            betar0 = (beta0[0, 0, 0] * ue[0, 0, 0] + beta0[0, 0, 1] * ue[1, 0, 0] + beta0[0, 0, 2] * ue[2, 0, 0]) * ue[0, 0, 0] + \
                     (beta0[0, 1, 0] * ue[0, 0, 0] + beta0[0, 1, 1] * ue[1, 0, 0] + beta0[0, 1, 2] * ue[2, 0, 0]) * ue[1, 0, 0] + \
                     (beta0[0, 2, 0] * ue[0, 0, 0] + beta0[0, 2, 1] * ue[1, 0, 0] + beta0[0, 2, 2] * ue[2, 0, 0]) * ue[2, 0, 0]

            betar1 = (beta1[0, 0, 0] * ue[-1, 0, 0] + beta1[0, 0, 1] * ue[0, 0, 0] + beta1[0, 0, 2] * ue[1, 0, 0]) * ue[-1, 0, 0] + \
                     (beta1[0, 1, 0] * ue[-1, 0, 0] + beta1[0, 1, 1] * ue[0, 0, 0] + beta1[0, 1, 2] * ue[1, 0, 0]) * ue[0, 0, 0] + \
                     (beta1[0, 2, 0] * ue[-1, 0, 0] + beta1[0, 2, 1] * ue[0, 0, 0] + beta1[0, 2, 2] * ue[1, 0, 0]) * ue[1, 0, 0]

            betar2 = (beta2[0, 0, 0] * ue[-2, 0, 0] + beta2[0, 0, 1] * ue[-1, 0, 0] + beta2[0, 0, 2] * ue[0, 0, 0]) * ue[-2, 0, 0] + \
                     (beta2[0, 1, 0] * ue[-2, 0, 0] + beta2[0, 1, 1] * ue[-1, 0, 0] + beta2[0, 1, 2] * ue[0, 0, 0]) * ue[-1, 0, 0] + \
                     (beta2[0, 2, 0] * ue[-2, 0, 0] + beta2[0, 2, 1] * ue[-1, 0, 0] + beta2[0, 2, 2] * ue[0, 0, 0]) * ue[0, 0, 0]

            # Compute alpha weights - classic WENO
            alpham0 = dw[2] / (betar0 + eps) ** (2 * p)    
            alpham1 = dw[1] / (betar1 + eps) ** (2 * p) 
            alpham2 = dw[0] / (betar2 + eps) ** (2 * p) 
             
            alphap0 = dw[0] / (betar0 + eps) ** (2 * p)   
            alphap1 = dw[1] / (betar1 + eps) ** (2 * p)
            alphap2 = dw[2] / (betar2 + eps) ** (2 * p)  
        
            # Compute nonlinear weights and cell interface values
            um = (alpham0 * uml0 + alpham1 * uml1 +  alpham2 * uml2) / \
                 (alpham0 + alpham1 + alpham2)

            up = (alphap0 * upl0 +  alphap1 * upl1 +  alphap2 * upl2) / \
                 (alphap0 + alphap1 + alphap2)

    with computation(FORWARD), interval(...):
        du =  - (BurgersLF(up[1, 0, 0], um[2, 0, 0], 0, maxvel)-BurgersLF(up[0, 0, 0], um[1, 0, 0], 0, maxvel)) / dx     



def BurgersWENO2D_(du: Field[float], xe: Field[float], ue: Field[float], um: Field[float], up: Field[float],
                   upl: Field[gtscript.IJK, (float, (3,))], uml: Field[gtscript.IJK, (float, (3,))],
                   betar: Field[gtscript.IJK, (float, (3,))],
                   alpham: Field[gtscript.IJK, (float, (3,))], alphap: Field[gtscript.IJK, (float, (3,))],
                   dx: float, m: int, p: int, eps: float, maxvel: float):

    with computation(PARALLEL), interval(0, 1):
        upl[0, 0, 0][0] =  1 / 3 * ue[ 0, 0, 0] + 5 / 6 * ue[ 1, 0, 0] -  1 / 6 * ue[2, 0, 0]
        upl[0, 0, 0][1] = -1 / 6 * ue[-1, 0, 0] + 5 / 6 * ue[ 0, 0, 0] +  1 / 3 * ue[1, 0, 0]
        upl[0, 0, 0][2] =  1 / 3 * ue[-2, 0, 0] - 7 / 6 * ue[-1, 0, 0] + 11 / 6 * ue[0, 0, 0]

        uml[0, 0, 0][0] = 11 / 6 * ue[ 0, 0, 0] - 7 / 6 * ue[ 1, 0, 0] + 1 / 3 * ue[2, 0, 0]
        uml[0, 0, 0][1] =  1 / 3 * ue[-1, 0, 0] + 5 / 6 * ue[ 0, 0, 0] - 1 / 6 * ue[1, 0, 0]
        uml[0, 0, 0][2] = -1 / 6 * ue[-2, 0, 0] + 5 / 6 * ue[-1, 0, 0] + 1 / 3 * ue[0, 0, 0]

        betar[0, 0, 0][0] = 10 / 3 * ue[0, 0, 0] * ue[0, 0, 0] - 31 / 3 * ue[0, 0, 0] * ue[1, 0, 0] + 25 / 3 * ue[1, 0, 0] * ue[1, 0, 0] + \
                            11 / 3 * ue[0, 0, 0] * ue[2, 0, 0] - 19 / 3 * ue[1, 0, 0] * ue[2, 0, 0] +  4 / 3 * ue[2, 0, 0] * ue[2, 0, 0]

        betar[0, 0, 0][1] = 4 / 3 * ue[-1, 0, 0] * ue[-1, 0, 0] - 13 / 3 * ue[-1, 0, 0] * ue[0, 0, 0] + 13 / 3 * ue[0, 0, 0] * ue[0, 0, 0] + \
                            5 / 3 * ue[-1, 0, 0] * ue[ 1, 0, 0] - 13 / 3 * ue[ 0, 0, 0] * ue[1, 0, 0] +  4 / 3 * ue[1, 0, 0] * ue[1, 0, 0]

        betar[0, 0, 0][2] = 4 / 3 * ue[-2, 0, 0] * ue[-2, 0, 0] - 19 / 3 * ue[-2, 0, 0] * ue[-1, 0, 0] + 25 / 3 * ue[-1, 0, 0] * ue[-1, 0, 0] + \
                           11 / 3 * ue[-2, 0, 0] * ue[ 0, 0, 0] - 31 / 3 * ue[-1, 0, 0] * ue[ 0, 0, 0] + 10 / 3 * ue[ 0, 0, 0] * ue[ 0, 0, 0]

        alpham[0, 0, 0][0] = (1 / 10) / (betar[0, 0, 0][0] + eps) ** (2 * p)
        alpham[0, 0, 0][1] = (3 /  5) / (betar[0, 0, 0][1] + eps) ** (2 * p)
        alpham[0, 0, 0][2] = (3 / 10) / (betar[0, 0, 0][2] + eps) ** (2 * p)

        alphap[0, 0, 0][0] = (3 / 10) / (betar[0, 0, 0][0] + eps) ** (2 * p)
        alphap[0, 0, 0][1] = (3 /  5) / (betar[0, 0, 0][1] + eps) ** (2 * p)
        alphap[0, 0, 0][2] = (1 / 10) / (betar[0, 0, 0][2] + eps) ** (2 * p)

        um = (alpham[0, 0, 0][0] * uml[0, 0, 0][0] + alpham[0, 0, 0][1] * uml[0, 0, 0][1] + alpham[0, 0, 0][2] * uml[0, 0, 0][2]) / \
             (alpham[0, 0, 0][0] + alpham[0, 0, 0][1] + alpham[0, 0, 0][2])
        up = (alphap[0, 0, 0][0] * upl[0, 0, 0][0] + alphap[0, 0, 0][1] * upl[0, 0, 0][1] + alphap[0, 0, 0][2] * upl[0, 0, 0][2]) / \
             (alphap[0, 0, 0][0] + alphap[0, 0, 0][1] + alphap[0, 0, 0][2])

    with computation(FORWARD), interval(...):
        du = -(BurgersLF(up[1, 0, 0], um[2, 0, 0], 0, maxvel) -
               BurgersLF(up[0, 0, 0], um[1, 0, 0], 0, maxvel)) / dx


def BurgersWENOrhs(xe, ue, dx, k, m, Crec, maxvel, dw, dwf, beta, N, p, eps):
    """Evaluate the RHS of Burgers equations using a WENO reconstruction"""

    du = np.zeros((N + 2 * m))

    # Define cell left and right interface values
    um = np.zeros((N + 2))
    up = np.zeros((N + 2))

    for i in range(1, N + 3):  # i=1 [0...5] i=2 [1...6] ...
        um[i - 1], up[i - 1] = WENO(xe[i - 1: i + 2 * (m - 1)],
                                    ue[i - 1: i + 2 * (m - 1)], m, Crec, dw, dwf, beta, p, eps,)

    du[m:-m] = (-(BurgersLF(up[1: N + 1], um[2: N + 2], 0, maxvel) -
                  BurgersLF(up[0:N], um[1: N + 1], 0, maxvel)) / dx)

    return du
    
    
def WENO(xloc, uloc, m, Crec, dw, dwf, beta, p, eps):
    """
    Reconstruct the left (mu) and right (up) cell interface values using an WENO
    approach based on 2m-1 long vectors uloc with cell
    """

    if m == 1:
        um = uloc[0]
        up = uloc[0]

    else:
        alpham = np.zeros(m)  # 3
        alphap = np.zeros(m)  # 3
        upl = np.zeros(m)  # 3
        uml = np.zeros(m)  # 3
        betar = np.zeros(m)  # 3  Crec (4,3)  beta(3, 3, 3)

        # Compute um and up based on different stencils and smoothness indicators and alpha
        for r in range(0, m):
            umh = uloc[
                m - r + np.arange(0, m) - 1
            ]  # 3  r=0 [2 3 4] r=1 [1 2 3] r=2 [0 1 2]

            upl[r] = matmul(Crec[r + 1, :], umh)
            uml[r] = matmul(Crec[r, :], umh)
            betar[r] = matmul(matmul(umh, beta[:, :, r]), umh)  # 3 (3,3) 3

        # Compute alpha weights - classic WENO
        alphap = dw / (betar + eps) ** (2 * p)  # 3
        alpham = dwf / (betar + eps) ** (2 * p) # 3

        # Compute nonlinear weights and cell interface values
        um = matmul(alpham, uml) / sum(alpham)  # 1
        up = matmul(alphap, upl) / sum(alphap)  # 1

    return um, up
