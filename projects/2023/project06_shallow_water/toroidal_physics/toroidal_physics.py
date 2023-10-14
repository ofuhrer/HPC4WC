"""
implementation of toroidal planetary physics to be used in shallow water model

the toroidal coordinate system with r, theta and phi is equivalent to the unwrapped & flattened torus (rectangle) with cartesian coordinates z, y and x.

cylindrical coordinates with r, theta and z are used, 
as well as toroidal coordinates with theta and theta, along mayor and minor radii respectively, where theta=0 is the outer equator
theta is not used for fields that are symmetric about the z axis
"""

import numpy as np


def setup_toroidal_planet(theta, r_major, r_minor, g_0, omega):
    """
    returns the constant parameters of the toroidal planet

    input parameters:
    theta : angle of the toroidal coordinate system
    r_major : major radius of the torus
    r_minor : minor radius of the torus
    g_0 : gravitational acceleration at the equator
    omega : angular velocity of the planet

    returns:
    g_t_r: gravitational acceleration in the toroidal r direction
    g_t_theta: gravitational acceleration in the toroidal theta direction
    """

    r, z = toroidal2cylindrical(theta, r_major, r_minor)
    print('r',r[0,:])
    print('z',z[0,:])
    g_r, g_z = toroidal_gravity(r, z, r_major)
    print("g_r",g_r[0,:])
    print("g_z",g_z[0,:])
    g_r, g_z = scale_gravity(g_r, g_z, theta, g_0)
    print("g_r",g_r[0,:])
    print("g_z",g_z[0,:])
    centrifugal_r = centrifugal_acceleration(r, omega)
    print("cf_r",centrifugal_r[0,:])
    g_t_r, g_t_theta = vector_cylindrical2toroidal(theta, g_r + centrifugal_r, g_z)
    print("g_t_r",g_t_r[0,:])
    print("g_t_t",g_t_theta[0,:])
    return g_t_r, g_t_theta


def toroidal_gravity(r, z, r_major, rho=1, grav_const=1, integration_points=100, potential=False):
    """
    returns the gravitational acceleration vector in cylindrical coordinates
    works by approximating the torus as a series of point masses along a circle and integrating over them

    input parameters:
    r : cylindrical radius
    z : height above the equatorial plane
    rho : mass density of the planet
    r_major : major radius of the torus
    grav_const : gravitational constant
    integration_points : number of integration points to use in the numerical integration of the gravitational acceleration
    potential: switch to output gravitational potential

    returns:
    a_r : acceleration in the r direction
    a_z : acceleration in the z direction
    optional returns:
    v_g : gravitational potential of the input locations
    
    """

    rho = 1
    d_theta = np.pi / integration_points
    point_mass = rho * r_major * d_theta

    a_r = 0
    a_z = 0
    a_theta = 0
    v_g = 0

    for theta in np.linspace(0, 2 * np.pi, 2 * integration_points):
        # distance between point of interest and point mass on circle on the z=0 plane
        distance_z0 = np.sqrt(
            np.square(r_major) + np.square(r) - 2 * r_major * r * np.cos(theta)
        )

        # 3d distance between point of interest and point mass
        distance = np.sqrt(np.square(distance_z0) + np.square(z))

        # total acceleration
        a = grav_const * point_mass / np.square(distance)

        # gravitational potential
        v_g += -1.0 * grav_const * point_mass / distance
        
        # force in the r direction
        distance_r = r - r_major * np.cos(theta)

        a_r += -a * (distance_r / distance)

        # force in the theta direction
        distance_theta = r_major * np.sin(theta)

        a_theta += -a * (distance_theta / distance)

        # force in the z direction
        a_z += -a * (z / distance)

    if potential:
        return a_r, a_z, v_g
    else:
        return a_r, a_z


def scale_gravity(a_r, a_z, theta, g_0):
    """
    scale the gravitational acceleration by the set gravitational acceleration at the equator

    input parameters:
    a_r : acceleration in the r direction
    a_z : acceleration in the z direction
    theta : toroidal angle, theta=0 is the outer equator
    g_0 : gravitational acceleration at the equator
    """
    # index where theta = 0
    if theta.ndim == 1:
        theta_0 = np.argmin(np.abs(theta))
        print('1d:',theta[theta_0])
        a_equ = np.sqrt(np.square(a_r[theta_0]) + np.square(a_z[theta_0]))
    elif theta.ndim == 2:
        theta_0 = np.argmin(np.abs(theta[0,:]))
        print('2d:',theta[0,theta_0])
        a_equ = np.sqrt(np.square(a_r[0,theta_0]) + np.square(a_z[0,theta_0]))
    a_r = a_r * g_0 / a_equ
    a_z = a_z * g_0 / a_equ

    return a_r, a_z


def centrifugal_acceleration(r, omega):
    """
    returns the centrifugal acceleration vector in cylindrical coordinates

    input parameters:
    r : cylindrical radius
    omega : angular velocity

    returns:
    a_r : acceleration in the r direction
    """

    v = omega * r
    a_r = np.square(v) / r

    return a_r


def toroidal_coriolis_acceleration(theta, v_r, v_theta, v_phi):
    """
    returns the coriolis acceleration vector in toroidal coordinates

    input parameters:
    theta: toroidal angle, theta=0 is the outer equator
    v_r: velocity in the toroidal r direction
    v_theta: velocity in the toroidal theta direction
    v_phi: velocity in the toroidal phi direction

    returns:
    a_r: acceleration in toroidal the r direction
    a_theta: acceleration in the toroidal theta direction
    a_phi: acceleration in the toroidal phi direction
    """
    omega = 2 * np.pi / 24 / 3600  # 1 rotation per day

    a_phi = 2 * omega * (v_theta * np.sin(theta) - v_r * np.cos(theta))
    a_theta = 2 * omega * (-v_phi * np.sin(theta))
    a_r = 2 * omega * (v_phi * np.cos(theta))

    return a_r, a_theta, a_phi


def toroidal2cylindrical(theta, r_major, r_minor, phi=None):
    """
    input parameters:
    converts from toroidal to cylindrical coordinates
    theta: toroidal angle, theta=0 is the outer equator
    r_major: major radius of the torus
    r_minor: minor radius of the torus
    phi: cylindrical angle

    returns:
    r: cylindrical radius
    z: height above the equatorial plane
    phi: cylindrical angle (if phi is given)
    """

    r = r_major + r_minor * np.cos(theta)
    z = r_minor * np.sin(theta)

    if phi is None:
        return r, z
    else:
        return r, z, phi

def toroidal3d2cylindrical(theta, r_major, r_minor, phi=None):
    """
    input parameters:
    converts from toroidal to cylindrical coordinates
    theta: toroidal angle, theta=0 is the outer equator
    r_major: major radius of the torus
    r_minor: minor radius of the torus (1d-array)
    phi: cylindrical angle

    returns:
    r: cylindrical radius
    z: height above the equatorial plane
    phi: cylindrical angle (if phi is given)
    """
    rhead,thetahead = np.meshgrid(r_minor,theta,indexing='ij')
    #print("rhead",rhead.shape,rhead[:,0])
    #print("thead",thetahead.shape,thetahead[:,0])

    r = r_major + rhead * np.cos(thetahead)
    z = rhead * np.sin(thetahead)

    if phi is None:
        return r, z
    else:
        return r, z, phi

def toroidal_slice_width(theta, r_major, r_minor):
    """
    input parameters:
    theta: toroidal angle, theta=0 is the outer equator
    r_major: major radius of the torus
    r_minor: minor radius of the torus

    returns:
    width of a toroidal slice at the given theta relative to the width at the equator
    """

    w_0 = 1
    w = (r_major + r_minor * np.cos(theta)) / (r_minor + r_major) * w_0

    return w


def vector_cylindrical2toroidal(theta, a_r, a_z, a_theta=None):
    """
    cordinate transformation of a vector a from cylindrical (r,z,theta) to toroidal coordinates (r,theta,theta)
    input parameters:
    theta: toroidal angle, theta=0 is the outer equator
    r: cylindrical radius
    z: height above the equatorial plane
    theta: cylindrical angle
    a_r: acceleration in the cylindrical r direction
    a_z: acceleration in the cylindrical z direction
    a_theta: acceleration in the cylindrical theta direction

    returns:
    a_t_r: acceleration in the r_minor direction (normal to the torus surface)
    a_t_theta: acceleration in the theta direction (horizontal to the torus surface, along the minor radius)
    a_t_phi: acceleration in the phi direction (along the major radius)
    """

    a_t_r = a_r * np.cos(theta) + a_z * np.sin(theta)
    a_t_theta = -a_r * np.sin(theta) + a_z * np.cos(theta)
    a_t_phi = a_theta
    if a_theta is None:
        return a_t_r, a_t_theta
    else:
        return a_t_r, a_t_theta, a_t_phi
