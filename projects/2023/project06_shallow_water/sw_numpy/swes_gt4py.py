#!/usr/bin/env python3

import sys
import math
import numpy as np
from scipy.stats import norm


class Solver:
    """
    NumPy implementation of a solver class for
    Shallow Water Equations over the surface of a sphere (SWES).

    Notation employed:
    * h			fluid height
    * hs		terrain height (topography)
    * ht		total height (ht = hs + h)
    * phi		longitude
    * R			planet radius
    * theta		latitude
    * u			longitudinal fluid velocity
    * v			latitudinal fluid velocity
    """


    def __init__(self, T, M, N, IC, CFL, diffusion):
        """
        Constructor.

        :param	T:	simulation length [days]
        :param	M:	number of grid points along longitude
        :param	N:	number of grid points along latitude
        :param	IC:	tuple storing in first position the ID of the initial condition,
                    then possibly needed parameters. Options for ID:
                    * 0: test case 6 by Williamson
                    * 1: test case 2 by Williamson
        :param	CFL:	CFL number
        :param	diffusion:
                    * TRUE to take viscous diffusion into account,
                    * FALSE otherwise
        """

        # --- TO DO --- #
        # move all needed fields (incl dx) to gt4py
        # in_field = gt.storage.from_array(in_field_np, backend, default_origin)
        # --- TO DO --- #
        
        # --- Build grid --- #

        assert ((M > 1) and (N > 1)), "Number of grid points along each direction must be greater than one."

        # Discretize longitude
        self.M = M
        self.dphi = 2.0 * math.pi / self.M
        self.phi1D = np.linspace(-self.dphi, 2.0*math.pi + self.dphi, self.M+3)

        # Discretize latitude
        # Note: we exclude the poles and only consider a channel from -85 S to 85 N to avoid pole problem
        # Note: the number of grid points must be even to prevent f to vanish
        #       (important for computing initial height and velocity in geostrophic balance)
        if (N % 2 == 0):
            self.N = N
        else:
            self.N = N + 1
            print('Warning: Number of grid points along latitude has been increased by one unit so to make it even.')
        self.theta_range = 85.0
        self.dtheta = (2*self.theta_range/180.0) * math.pi / (self.N - 1)
        self.theta1D = np.linspace(-self.theta_range/180.0*math.pi, self.theta_range/180.0*math.pi, self.N)

        # Build grid
        self.phi, self.theta = np.meshgrid(self.phi1D, self.theta1D, indexing = 'ij')

        # Cosine of mid-point values for theta along y
        self.c = np.cos(self.theta)
        self.cMidy = np.cos(0.5 * (self.theta[1:-1,1:] + self.theta[1:-1,:-1]))

        # Compute $\tan(\theta)$
        self.tg = np.tan(self.theta[1:-1,1:-1])
        self.tgMidx = np.tan(0.5 * (self.theta[:-1,1:-1] + self.theta[1:,1:-1]))
        self.tgMidy = np.tan(0.5 * (self.theta[1:-1,:-1] + self.theta[1:-1,1:]))

        # --- Set planet's constants --- #

        self.setPlanetConstants()

        # --- Cartesian coordinates and increments --- #

        # Coordinates
        self.x	= self.a * np.cos(self.theta) * self.phi
        self.y	= self.a * self.theta
        self.y1 = self.a * np.sin(self.theta)

        # Increments
        self.dx  = self.x[1:,:] - self.x[:-1,:]
        self.dy  = self.y[:,1:] - self.y[:,:-1]
        self.dy1 = self.y1[:,1:] - self.y1[:,:-1]

        # Compute mimimum distance between grid points on the sphere.
        # This will be useful for CFL condition
        self.dxmin = self.dx.min()
        self.dymin = self.dy.min()

        # "Centred" increments. Useful for updating solution
        # with Lax-Wendroff scheme
        self.dxc  = 0.5 * (self.dx[:-1,1:-1] + self.dx[1:,1:-1])
        self.dyc  = 0.5 * (self.dy[1:-1,:-1] + self.dy[1:-1,1:])
        self.dy1c = 0.5 * (self.dy1[1:-1,:-1] + self.dy1[1:-1,1:])

        # --- Time discretization --- #

        assert(T >= 0), "Final time must be non-negative."

        # Convert simulation length from days to seconds
        self.T = 24.0 * 3600.0 * T

        # CFL number; this will be used to determine the timestep
        # at each iteration
        self.CFL = CFL

        # --- Terrain height --- #

        # Note: currently just a flat surface
        self.hs = np.zeros((self.M+3, self.N), float)

        # --- Set initial conditions --- #

        assert(IC in range(2)), "Invalid problem ID. See code documentation for implemented initial conditions."
        self.IC = IC
        self.setInitialConditions()

        # --- Setup diffusion --- #
        self.diffusion = diffusion

        # Pre-compute coefficients of second-order approximations of first-order derivative
        if (self.diffusion):
            # Centred finite difference along longitude
            # Ax, Bx and Cx denote the coefficients associated
            # to the centred, upwind and downwind point, respectively
            self.Ax = (self.dx[1:,1:-1] - self.dx[:-1,1:-1]) / (self.dx[1:,1:-1] * self.dx[:-1,1:-1])
            self.Ax = np.concatenate((self.Ax[-2:-1,:], self.Ax, self.Ax[1:2,:]), axis = 0)

            self.Bx = self.dx[:-1,1:-1] / (self.dx[1:,1:-1] * (self.dx[1:,1:-1] + self.dx[:-1,1:-1]))
            self.Bx = np.concatenate((self.Bx[-2:-1,:], self.Bx, self.Bx[1:2,:]), axis = 0)

            self.Cx = - self.dx[1:,1:-1] / (self.dx[:-1,1:-1] * (self.dx[1:,1:-1] + self.dx[:-1,1:-1]))
            self.Cx = np.concatenate((self.Cx[-2:-1,:], self.Cx, self.Cx[1:2,:]), axis = 0)

            # Centred finite difference along latitude
            # Ay, By and Cy denote the coefficients associated
            # to the centred, upwind and downwind point, respectively
            self.Ay = (self.dy[1:-1,1:] - self.dy[1:-1,:-1]) / (self.dy[1:-1,1:] * self.dy[1:-1,:-1])
            self.Ay = np.concatenate((self.Ay[:,0:1], self.Ay, self.Ay[:,-1:]), axis = 1)

            self.By = self.dy[1:-1,:-1] / (self.dy[1:-1,1:] * (self.dy[1:-1,1:] + self.dy[1:-1,:-1]))
            self.By = np.concatenate((self.By[:,0:1], self.By, self.By[:,-1:]), axis = 1)

            self.Cy = - self.dy[1:-1,1:] / (self.dy[1:-1,:-1] * (self.dy[1:-1,1:] + self.dy[1:-1,:-1]))
            self.Cy = np.concatenate((self.Cy[:,0:1], self.Cy, self.Cy[:,-1:]), axis = 1)

        # --- TO DO --- #
        # # compile diffusion stencil
        # kwargs = {"verbose": True} if backend in ("gtx86", "gtmc", "gtcuda") else {}
        # diffusion_stencil = gtscript.stencil(
        #     definition=diffusion_defs,
        #     backend=backend,
        #     externals={"laplacian": laplacian},
        #     rebuild=False,
        #     **kwargs,
        # )
        # --- TO DO --- #

    def setPlanetConstants(self):
        """
        Set Earth's constants.

        :attribute	g:				gravity	[m/s2]
        :attribute	rho:			average atmosphere density	[kg/m3]
        :attribute	a:				average radius	[m]
        :attribute	omega:			rotation rate	[Hz]
        :attribute	scaleHeight:	atmosphere scale height	[m]
        :attribute	nu:				viscosity	[m2/s]
        :attribute	f:				Coriolis parameter	[Hz]

        :param:

        :return:
        """

        # Earth
        self.g				= 9.80616
        self.rho			= 1.2
        self.a				= 6.37122e6
        self.omega			= 7.292e-5
        self.scaleHeight	= 8.0e3
        self.nu				= 5.0e5

        # Coriolis parameter
        self.f = 2.0 * self.omega * np.sin(self.theta)


    def setInitialConditions(self):
        """
        Set initial conditions.

        :attribute	h:	initial fluid height
        :attribute	u:	initial longitudinal velocity
        :attribute	v:	initial latitudinal velocity

        :param:

        :return:
        """

        # --- IC 0: sixth test case taken from Williamson's suite --- #
        # ---       Rossby-Haurwitz Wave                          --- #

        if (self.IC == 0):
            # Set constants
            w  = 7.848e-6
            K  = 7.848e-6
            h0 = 8e3
            R  = 4.0

            # Compute initial fluid height
            A = 0.5 * w * (2.0 * self.omega + w) * (np.cos(self.theta) ** 2.0) + \
                0.25 * (K ** 2.0) * (np.cos(self.theta) ** (2.0 * R)) * \
                ((R + 1.0) * (np.cos(self.theta) ** 2.0) + \
                 (2.0 * (R ** 2.0) - R - 2.0) - \
                 2.0 * (R ** 2.0) * (np.cos(self.theta) ** (-2.0)))
            B = (2.0 * (self.omega + w) * K) / ((R + 1.0) * (R + 2.0)) * \
                (np.cos(self.theta) ** R) * \
                (((R ** 2.0) + 2.0 * R + 2.0) - \
                 ((R + 1.0) ** 2.0) * (np.cos(self.theta) ** 2.0))
            C = 0.25 * (K ** 2.0) * (np.cos(self.theta) ** (2.0 * R)) * \
                ((R + 1.0) * (np.cos(self.theta) ** 2.0) - (R + 2.0))

            h = h0 + ((self.a ** 2.0) * A + \
                      (self.a ** 2.0) * B * np.cos(R * self.phi) + \
                      (self.a ** 2.0) * C * np.cos(2.0 * R * self.phi)) / self.g

            # Compute initial wind
            u = self.a * w * np.cos(self.theta) + \
                self.a * K * (np.cos(self.theta) ** (R - 1.0)) * \
                (R * (np.sin(self.theta) ** 2.0) - (np.cos(self.theta) ** 2.0)) * \
                np.cos(R * self.phi)
            v = - self.a * K * R * (np.cos(self.theta) ** (R - 1.0)) * \
                  np.sin(self.theta) * np.sin(R * self.phi)

        # --- IC 1: second test case taken from Williamson's suite --- #
        # ----      Steady State Nonlinear Zonal Geostrophic Flow  --- #

        elif (self.IC == 1):
            # Suggested values for $\alpha$ for second
            # test cases of Williamson's suite:
            #	- 0
            #	- 0.05
            #	- pi/2 - 0.05
            #	- pi/2
            alpha = math.pi/2

            # Set constants
            u0 = 2.0 * math.pi * self.a / (12.0 * 24.0 * 3600.0)
            h0 = 2.94e4 / self.g

            # Make Coriolis parameter dependent on longitude and latitude
            self.f = 2.0 * self.omega * \
                     (- np.cos(self.phi) * np.cos(self.theta) * np.sin(alpha) + \
                      np.sin(self.theta) * np.cos(alpha))

            # Compute initial height
            h = h0 - (self.a * self.omega * u0 + 0.5 * (u0 ** 2.0)) * \
                     ((- np.cos(self.phi) * np.cos(self.theta) * np.sin(alpha) + \
                       np.sin(self.theta) * np.cos(alpha)) ** 2.0) / self.g

            # Compute initial wind
            u = u0 * (np.cos(self.theta) * np.cos(alpha) + \
                      np.cos(self.phi) * np.sin(self.theta) * np.sin(alpha))
            self.uMidx = u0 * (np.cos(0.5 * (self.theta[:-1,:] + self.theta[1:,:])) * np.cos(alpha) + \
                                 np.cos(0.5 * (self.phi[:-1,:] + self.phi[1:,:])) * \
                                 np.sin(0.5 * (self.theta[:-1,:] + self.theta[1:,:])) * np.sin(alpha))
            self.uMidy = u0 * (np.cos(0.5 * (self.theta[:,:-1] + self.theta[:,1:])) * np.cos(alpha) + \
                                 np.cos(0.5 * (self.phi[:,:-1] + self.phi[:,1:])) * \
                                 np.sin(0.5 * (self.theta[:,:-1] + self.theta[:,1:])) * np.sin(alpha))

            v = - u0 * np.sin(self.phi) * np.sin(alpha)
            self.vMidx = - u0 * np.sin(0.5 * (self.phi[:-1,:] + self.phi[1:,:])) * np.sin(alpha)
            self.vMidy = - u0 * np.sin(0.5 * (self.phi[:,:-1] + self.phi[:,1:])) * np.sin(alpha)

        self.h = h
        self.u = u
        self.v = v


    def computeLaplacian(self, q):
        """
        Auxiliary methods evaluating the Laplacian of a given quantity
        in all interior points of the grid. The approximations is given
        by applying twice a centre finite difference formula along
        both axis.

        :param	q:	conserved quantity

        :return qlap:	Laplacian of q
        """
        
        # --- TO DO --- #
        # turn into gt4py function
        # --- TO DO --- #
        
        # Compute second order derivative along longitude
        qxx = self.Ax[1:-1,:] * (self.Ax[1:-1,:] * q[2:-2,2:-2] + \
                                 self.Bx[1:-1,:] * q[3:-1,2:-2] + \
                                 self.Cx[1:-1,:] * q[1:-3,2:-2]) + \
              self.Bx[1:-1,:] * (self.Ax[2:,:] * q[3:-1,2:-2] + \
                                   self.Bx[2:,:] * q[4:,2:-2] + \
                                   self.Cx[2:,:] * q[2:-2,2:-2]) + \
              self.Cx[1:-1,:] * (self.Ax[:-2,:] * q[1:-3,2:-2] + \
                                   self.Bx[:-2,:] * q[2:-2,2:-2] + \
                                   self.Cx[:-2,:] * q[:-4,2:-2])

        # Compute second order derivative along latitude
        qyy = self.Ay[:,1:-1] * (self.Ay[:,1:-1] * q[2:-2,2:-2] + \
                                 self.By[:,1:-1] * q[2:-2,3:-1] + \
                                 self.Cy[:,1:-1] * q[2:-2,1:-3]) + \
              self.By[:,1:-1] * (self.Ay[:,2:] * q[2:-2,3:-1] + \
                                   self.By[:,2:] * q[2:-2,4:] + \
                                   self.Cy[:,2:] * q[2:-2,2:-2]) + \
              self.Cy[:,1:-1] * (self.Ay[:,:-2] * q[2:-2,1:-3] + \
                                   self.By[:,:-2] * q[2:-2,2:-2] + \
                                   self.Cy[:,:-2] * q[2:-2,:-4])

        # Compute Laplacian
        qlap = qxx + qyy

        return qlap


    def LaxWendroff(self, h, u, v):
        """
        Update solution through finite difference Lax-Wendroff scheme.
        Note that Coriolis effect is taken into account in Lax-Wendroff step,
        while diffusion is separately added afterwards.

        :param	h:	fluid height at current timestep
        :param	u:	longitudinal velocity at current timestep
        :param	v:	latitudinal velocity at current timestep

        :return	hnew:	updated fluid height
        :return	unew:	updated longitudinal velocity
        :return	vnew:	updated latitudinal velocity
        """

        # --- Auxiliary variables --- #

        v1	= v * self.c
        hu	= h * u
        hv	= h * v
        hv1 = h * v1

        # --- TO DO --- #
        # turn the computations into gt4py functions
        # and LaxWendroff into a stencil
        # --- TO DO --- #
        
        # --- Compute mid-point values after half timestep --- #

        # Mid-point value for h along x
        hMidx = 0.5 * (h[1:,1:-1] + h[:-1,1:-1]) - \
             0.5 * self.dt / self.dx[:,1:-1] * (hu[1:,1:-1] - hu[:-1,1:-1])

        # Mid-point value for h along y
        hMidy = 0.5 * (h[1:-1,1:] + h[1:-1,:-1]) - \
             0.5 * self.dt / self.dy1[1:-1,:] * (hv1[1:-1,1:] - hv1[1:-1,:-1])

        # Mid-point value for hu along x
        Ux = hu * u + 0.5 * self.g * h * h
        huMidx = 0.5 * (hu[1:,1:-1] + hu[:-1,1:-1]) - \
                0.5 * self.dt / self.dx[:,1:-1] * (Ux[1:,1:-1] - Ux[:-1,1:-1]) + \
                0.5 * self.dt * \
                (0.5 * (self.f[1:,1:-1] + self.f[:-1,1:-1]) + \
                0.5 * (self.u[1:,1:-1] + self.u[:-1,1:-1]) * self.tgMidx / self.a) * \
                (0.5 * (hv[1:,1:-1] + hv[:-1,1:-1]))

        # Mid-point value for hu along y
        Uy = hu * v1
        huMidy = 0.5 * (hu[1:-1,1:] + hu[1:-1,:-1]) - \
                0.5 * self.dt / self.dy1[1:-1,:] * (Uy[1:-1,1:] - Uy[1:-1,:-1]) + \
                0.5 * self.dt * \
                (0.5 * (self.f[1:-1,1:] + self.f[1:-1,:-1]) + \
                0.5 * (u[1:-1,1:] + u[1:-1,:-1]) * self.tgMidy / self.a) * \
                (0.5 * (hv[1:-1,1:] + hv[1:-1,:-1]))

        # Mid-point value for hv along x
        Vx = hu * v
        hvMidx = 0.5 * (hv[1:,1:-1] + hv[:-1,1:-1]) - \
                0.5 * self.dt / self.dx[:,1:-1] * (Vx[1:,1:-1] - Vx[:-1,1:-1]) - \
                0.5 * self.dt * \
                (0.5 * (self.f[1:,1:-1] + self.f[:-1,1:-1]) + \
                0.5 * (u[1:,1:-1] + u [:-1,1:-1]) * self.tgMidx / self.a) * \
                (0.5 * (hu[1:,1:-1] + hu[:-1,1:-1]))

        # Mid-point value for hv along y
        Vy1 = hv * v1
        Vy2 = 0.5 * self.g * h * h
        hvMidy = 0.5 * (hv[1:-1,1:] + hv[1:-1,:-1]) - \
                0.5 * self.dt / self.dy1[1:-1,:] * (Vy1[1:-1,1:] - Vy1[1:-1,:-1]) - \
                0.5 * self.dt / self.dy[1:-1,:] * (Vy2[1:-1,1:] - Vy2[1:-1,:-1]) - \
                0.5 * self.dt * \
                (0.5 * (self.f[1:-1,1:] + self.f[1:-1,:-1]) + \
                0.5 * (u[1:-1,1:] + u[1:-1,:-1]) * self.tgMidy / self.a) * \
                (0.5 * (hu[1:-1,1:] + hu[1:-1,:-1]))

        # --- Compute solution at next timestep --- #

        # Update fluid height
        hnew = h[1:-1,1:-1] - \
               self.dt / self.dxc * (huMidx[1:,:] - huMidx[:-1,:]) - \
               self.dt / self.dy1c * (hvMidy[:,1:]*self.cMidy[:,1:] - hvMidy[:,:-1]*self.cMidy[:,:-1])

        # Update longitudinal moment
        UxMid = np.where(hMidx > 0.0, \
                            huMidx * huMidx / hMidx + 0.5 * self.g * hMidx * hMidx, \
                            0.5 * self.g * hMidx * hMidx)
        UyMid = np.where(hMidy > 0.0, \
                            hvMidy * self.cMidy * huMidy / hMidy, \
                            0.0)
        hunew = hu[1:-1,1:-1] - \
                self.dt / self.dxc * (UxMid[1:,:] - UxMid[:-1,:]) - \
                self.dt / self.dy1c * (UyMid[:,1:] - UyMid[:,:-1]) + \
                self.dt * (self.f[1:-1,1:-1] + \
                            0.25 * (huMidx[:-1,:] / hMidx[:-1,:] + \
                                    huMidx[1:,:] / hMidx[1:,:] + \
                                    huMidy[:,:-1] / hMidy[:,:-1] + \
                                    huMidy[:,1:] / hMidy[:,1:]) * \
                            self.tg / self.a) * \
                0.25 * (hvMidx[:-1,:] + hvMidx[1:,:] + hvMidy[:,:-1] + hvMidy[:,1:]) - \
                self.dt * self.g * \
                0.25 * (hMidx[:-1,:] + hMidx[1:,:] + hMidy[:,:-1] + hMidy[:,1:]) * \
                (self.hs[2:,1:-1] - self.hs[:-2,1:-1]) / (self.dx[:-1,1:-1] + self.dx[1:,1:-1])

        # Update latitudinal moment
        VxMid = np.where(hMidx > 0.0, \
                            hvMidx * huMidx / hMidx, \
                            0.0)
        Vy1Mid = np.where(hMidy > 0.0, \
                            hvMidy * hvMidy / hMidy * self.cMidy, \
                            0.0)
        Vy2Mid = 0.5 * self.g * hMidy * hMidy
        hvnew = hv[1:-1,1:-1] - \
                self.dt / self.dxc * (VxMid[1:,:] - VxMid[:-1,:]) - \
                self.dt / self.dy1c * (Vy1Mid[:,1:] - Vy1Mid[:,:-1]) - \
                self.dt / self.dyc * (Vy2Mid[:,1:] - Vy2Mid[:,:-1]) - \
                self.dt * (self.f[1:-1,1:-1] + \
                            0.25 * (huMidx[:-1,:] / hMidx[:-1,:] + \
                                    huMidx[1:,:] / hMidx[1:,:] + \
                                    huMidy[:,:-1] / hMidy[:,:-1] + \
                                    huMidy[:,1:] / hMidy[:,1:]) * \
                            self.tg / self.a) * \
                0.25 * (huMidx[:-1,:] + huMidx[1:,:] + huMidy[:,:-1] + huMidy[:,1:]) - \
                self.dt * self.g * \
                0.25 * (hMidx[:-1,:] + hMidx[1:,:] + hMidy[:,:-1] + hMidy[:,1:]) * \
                (self.hs[1:-1,2:] - self.hs[1:-1,:-2]) / (self.dy1[1:-1,:-1] + self.dy1[1:-1,1:])

        # Come back to original variables
        unew = hunew / hnew
        vnew = hvnew / hnew

        # --- Add diffusion --- #

        if (self.diffusion):
            # Extend fluid height
            hext = np.concatenate((h[-4:-3,:], h, h[3:4,:]), axis = 0)
            hext = np.concatenate((hext[:,0:1], hext, hext[:,-1:]), axis = 1)

            # Add the Laplacian
            hnew += self.dt * self.nu * self.computeLaplacian(hext)

            # Extend longitudinal velocity
            uext = np.concatenate((u[-4:-3,:], u, u[3:4,:]), axis = 0)
            uext = np.concatenate((uext[:,0:1], uext, uext[:,-1:]), axis = 1)

            # Add the Laplacian
            unew += self.dt * self.nu * self.computeLaplacian(uext)

            # Extend fluid height
            vext = np.concatenate((v[-4:-3,:], v, v[3:4,:]), axis = 0)
            vext = np.concatenate((vext[:,0:1], vext, vext[:,-1:]), axis = 1)

            # Add the Laplacian
            vnew += self.dt * self.nu * self.computeLaplacian(vext)

        return hnew, unew, vnew


    def solve(self, verbose, save):
        """
        Solver.

        :param	verbose:	if positive, print to screen information about the solution
                            every 'verbose' timesteps
        :param	save:	if positive, store the solution every 'save' timesteps

        :return	h:	if save <= 0, fluid height at final time
        :return	u:	if save <= 0, fluid longitudinal velocity at final time
        :return	v:	if save <= 0, fluid latitudinal velocity at final time
        :return tsave:	if save > 0, stored timesteps
        :return	phi:	if save > 0, longitudinal coordinates of grid points
        :return theta:	if save > 0, latitudinal coordinates of grid points
        :return	hsave:	if save > 0, stored fluid height
        :return	usave:	if save > 0, stored longitudinal velocity
        :return	vsave:	if save > 0, stored latitudinal velocity
        """

        verbose = int(verbose)
        save = int(save)

        # --- Print and save --- #

        # Print to screen
        if (verbose > 0):
            norm = np.sqrt(self.u*self.u + self.v*self.v)
            umax = norm.max()
            print("Time = %6.2f hours (max %i); max(|u|) = %8.8f" \
                    % (0.0, int(self.T / 3600.0), umax))

        # Save
        if (save > 0):
            tsave = np.array([[0.0]])
            hsave = self.h[1:-1, :, np.newaxis]
            usave = self.u[1:-1, :, np.newaxis]
            vsave = self.v[1:-1, :, np.newaxis]

        # --- Time marching --- #

        n = 0
        t = 0.0

        while (t < self.T):

            # Update number of iterations
            n += 1

            # --- Compute timestep through CFL condition --- #

            # Compute flux Jacobian eigenvalues
            eigenx = (np.maximum(np.absolute(self.u - np.sqrt(self.g * np.absolute(self.h))),
                                 np.maximum(np.absolute(self.u),
                                             np.absolute(self.u + np.sqrt(self.g * np.absolute(self.h)))))).max()

            eigeny = (np.maximum(np.absolute(self.v - np.sqrt(self.g * np.absolute(self.h))),
                                 np.maximum(np.absolute(self.v),
                                             np.absolute(self.v + np.sqrt(self.g * np.absolute(self.h)))))).max()

            # Compute timestep
            dtmax = np.minimum(self.dxmin/eigenx, self.dymin/eigeny)
            self.dt = self.CFL * dtmax

            # If needed, adjust timestep not to exceed final time
            if (t + self.dt > self.T):
                self.dt = self.T - t
                t = self.T
            else:
                t += self.dt

            # --- Update solution --- #
            
            # --- TO DO --- # 
            # call our function
            # --- TO DO --- # 
            
            hnew, unew, vnew = self.LaxWendroff(self.h, self.u, self.v)

            # --- Update solution applying BCs --- #

            self.h[:,1:-1] = np.concatenate((hnew[-2:-1,:], hnew, hnew[1:2,:]), axis = 0)
            self.h[:,0]  = self.h[:,1]
            self.h[:,-1] = self.h[:,-2]

            self.u[:,1:-1] = np.concatenate((unew[-2:-1,:], unew, unew[1:2,:]), axis = 0)
            self.u[:,0]  = self.u[:,1]
            self.u[:,-1] = self.u[:,-2]

            self.v[:,1:-1] = np.concatenate((vnew[-2:-1,:], vnew, vnew[1:2,:]), axis = 0)
            self.v[:,0]  = self.v[:,1]
            self.v[:,-1] = self.v[:,-2]

            # --- Print and save --- #

            if (verbose > 0 and (n % verbose == 0)):
                norm = np.sqrt(self.u*self.u + self.v*self.v)
                umax = norm.max()
                print("Time = %6.2f hours (max %i); max(|u|) = %16.16f" \
                        % (t / 3600.0, int(self.T / 3600.0), umax))

            if (save > 0 and (n % save == 0)):
                tsave = np.concatenate((tsave, np.array([[t]])), axis = 0)
                hsave = np.concatenate((hsave, self.h[1:-1, :, np.newaxis]), axis = 2)
                usave = np.concatenate((usave, self.u[1:-1, :, np.newaxis]), axis = 2)
                vsave = np.concatenate((vsave, self.v[1:-1, :, np.newaxis]), axis = 2)

        # --- Return --- #

        if (save > 0):
            return tsave, self.phi, self.theta, hsave, usave, vsave
        else:
            return self.h, self.u, self.v

