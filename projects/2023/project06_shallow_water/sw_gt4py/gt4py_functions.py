#!/usr/bin/env python3
import time
import sys
import math
import numpy as np
import gt4py as gt
from gt4py import gtscript
from scipy.stats import norm

# --- STEP 0 --- #

def compute_temp_variables(
    u: gtscript.Field[float], 
    v: gtscript.Field[float],
    h: gtscript.Field[float],
    c: gtscript.Field[float],

    hu: gtscript.Field[float],
    hv: gtscript.Field[float],
    v1: gtscript.Field[float]):

    with computation(PARALLEL), interval(...):
        hu = h * u
        v1 = v * c
        hv = h * v
        
# --- STEP 1 --- #

@gtscript.function
def compute_hMidx(dx, dt, h, hu):
    return 0.5 * (h[1, 0,0] + h[0,0,0]) - 0.5 * dt / dx[0,0,0] * (hu[1,0,0] - hu[0,0,0])

@gtscript.function
def compute_hMidy(dy1, dt, h, hv1):
    return 0.5 * (h[0,1,0] + h[0,0,0]) - 0.5 * dt / dy1[0,0,0] * (hv1[0,1,0] - hv1[0,0,0])

@gtscript.function
def compute_huMidx(dx, dt, hu, hv, Ux, f, u, tgMidx, a):
    return (0.5 * (hu[1, 0,0] + hu[0,0,0]) - 0.5 * dt / dx[0,0,0] * (Ux[1,0,0] - Ux[0,0,0]) + \
            0.5 * dt * \
            (0.5 * (f[1,0,0] + f[0,0,0]) + \
            0.5 * (u[1,0,0] + u[0,0,0]) * tgMidx / a) * \
            (0.5 * (hv[1,0,0] + hv[0,0,0])))           

@gtscript.function
def compute_huMidy(dy1, dt, hu, hv, Uy, f, u, tgMidy, a):
    return (0.5 * (hu[0,1,0] + hu[0,0,0]) - 0.5 * dt / dy1[0,0,0] * (Uy[0,1,0] - Uy[0,0,0]) + \
        0.5 * dt * \
        (0.5 * (f[0,1,0] + f[0,0,0]) + \
        0.5 * (u[0,1,0] + u[0,0,0]) * tgMidy / a) * \
        (0.5 * (hv[0,1,0] + hv[0,0,0])))

@gtscript.function
def compute_hvMidx(dx, dt, hu, hv, Vx, f, u, tgMidx, a):
    first= 0.5 * (hv[1, 0,0] + hv[0,0,0])
    second = 0.5 * dt / dx[0,0,0] * (Vx[1,0,0] - Vx[0,0,0])
    third = 0.5 * dt * (0.5 * (f[1,0,0] + f[0,0,0]) + 0.5 * (u[1,0,0] + u[0,0,0]) * tgMidx / a) * (0.5 * (hu[1,0,0] + hu[0,0,0]))
                        
    return first - second - third 

@gtscript.function
def compute_hvMidy(dy1, dy, dt, hu, hv,  Vy1, Vy2, f, u, tgMidy, a):
    return (0.5 * (hv[0,1,0] + hv[0,0,0]) \
            - 0.5 * dt / dy1[0,0,0] * (Vy1[0,1,0] - Vy1[0,0,0]) -  \
            0.5 * dt / dy[0,0,0] * (Vy2[0,1,0] - Vy2[0,0,0]) -\
            0.5 * dt * \
            (0.5 * (f[0,1,0] + f[0,0,0]) + \
            0.5 * (u[0,1,0] + u[0,0,0]) * tgMidy / a) * \
            (0.5 * (hu[0,1,0] + hu[0,0,0])))

def x_staggered_first_step(
    u: gtscript.Field[float], 
    v: gtscript.Field[float],
    h: gtscript.Field[float],

    hu: gtscript.Field[float],
    hv: gtscript.Field[float],
    f: gtscript.Field[float],

    dx: gtscript.Field[float],
    tgMidx: gtscript.Field[float],

    hMidx: gtscript.Field[float],
    huMidx: gtscript.Field[float],
    hvMidx: gtscript.Field[float],
    *,
    dt: float,
    g: float,
    a: float):
    from __gtscript__ import PARALLEL, computation, interval
    from __externals__ import compute_hMidx, compute_huMidx, compute_hvMidx

    with computation(PARALLEL), interval(...):
            Ux = hu * u + 0.5 * g * h * h
            Vx = hu * v

            # Mid-point value for h along x
            hMidx = compute_hMidx(dx=dx, dt=dt, h=h, hu=hu)
            huMidx = compute_huMidx(dx=dx, dt=dt, hu=hu, hv=hv, Ux=Ux, f=f, u=u, tgMidx=tgMidx, a=a)
            hvMidx = compute_hvMidx(dx=dx, dt=dt, hu=hu, hv=hv, Vx=Vx, f=f, u=u, tgMidx=tgMidx, a=a)

def y_staggered_first_step(
    u: gtscript.Field[float], 
    v: gtscript.Field[float],
    h: gtscript.Field[float],

    hu: gtscript.Field[float],
    hv: gtscript.Field[float],
    v1: gtscript.Field[float],
    f: gtscript.Field[float],

    dy: gtscript.Field[float],
    dy1: gtscript.Field[float],
    tgMidy: gtscript.Field[float],

    hMidy: gtscript.Field[float],
    huMidy: gtscript.Field[float],
    hvMidy: gtscript.Field[float],   
    *,
    dt: float,
    g: float,
    a: float):
    from __gtscript__ import PARALLEL, computation, interval
    from __externals__ import compute_hMidy, compute_huMidy, compute_hvMidy
    
    with computation(PARALLEL), interval(...):
            hv1 = h * v1
            Uy = hu * v1
            Vy1 = hv * v1
            Vy2 = 0.5 * g * h * h

            # Mid-point value for h along y
            hMidy = compute_hMidy(dy1=dy1, dt=dt, h=h, hv1=hv1)
            huMidy = compute_huMidy(dy1=dy1, dt=dt, hu=hu, hv=hv, Uy=Uy, f=f, u=u, tgMidy=tgMidy, a=a)
            hvMidy = compute_hvMidy(dy1=dy1, dy=dy, dt=dt, hu=hu, hv=hv,  Vy1=Vy1, Vy2=Vy2, f=f, u=u, tgMidy=tgMidy, a=a)       
                
# --- STEP 2 --- #

@gtscript.function
def compute_hnew(h, dt, dxc, huMidx, dy1c, hvMidy, cMidy):
    return h[1,1,0] -  dt / dxc[0,0,0] * (huMidx[1,1,0] - huMidx[0,1,0]) - \
                       dt /dy1c[0,0,0] * (hvMidy[1,1,0]*cMidy[1,1,0] - hvMidy[1,0,0]*cMidy[1,0,0])

@gtscript.function
def compute_hunew(hu, dt, dxc, UxMid, dy1c, UyMid, f, huMidx, hMidx, huMidy,hMidy,tg, a, hvMidx,hvMidy,g, hs,dx):
    first= dt / dxc * (UxMid[1,1,0] - UxMid[0,1,0])

    second= dt / dy1c * (UyMid[1,1,0] - UyMid[1,0,0])

    third= dt * (f[1,1,0] +  0.25 * (huMidx[0,1,0] / hMidx[0,1,0] + \
                                    huMidx[1,1,0] / hMidx[1,1,0] + \
                                    huMidy[1,0,0] / hMidy[1,0,0] + \
                                    huMidy[1,1,0] / hMidy[1,1,0]) * \
                                    tg /a) * \
                                    0.25 * (hvMidx[0,1,0] + hvMidx[1,1,0] + hvMidy[1,0,0] + hvMidy[1,1,0])

    fourth= dt * g * 0.25 * (hMidx[0,1,0] + hMidx[1,1,0] + hMidy[1,0,0] + hMidy[1,1,0]) * \
                (hs[2,1,0] - hs[0,1,0]) / (dx[0,1,0] + dx[1,1,0])

    return hu[1,1,0] - first - second + third - fourth

@gtscript.function
def compute_hvnew(hv, dt, dxc, VxMid, dy1c, Vy1Mid, dyc, Vy2Mid, f, huMidx, hMidx, huMidy, hMidy, tg, a, g, hs, dy1):

    first  = dt / dxc * (VxMid[1,1,0] - VxMid[0,1,0])
    second = dt / dy1c * (Vy1Mid[1,1,0] - Vy1Mid[1,0,0])
    third  = dt / dyc * (Vy2Mid[1,1,0] - Vy2Mid[1,0,0])

    fourth = dt * (f[1,1,0] + 0.25 * (huMidx[0,1,0] / hMidx[0,1,0] + \
                                    huMidx[1,1,0] / hMidx[1,1,0] + \
                                    huMidy[1,0,0] / hMidy[1,0,0] + \
                                    huMidy[1,1,0] / hMidy[1,1,0]) * \
                                    tg / a) * \
                                    0.25 * (huMidx[0,1,0] + huMidx[1,1,0] + huMidy[1,0,0] + huMidy[1,1,0])#

    fifth  = dt * g * 0.25 * (hMidx[0,1,0] + hMidx[1,1,0] + hMidy[1,0,0] + hMidy[1,1,0]) * (hs[1,2,0] - hs[1,0,0]) / (dy1[1,1,0] + dy1[1,0,0]) 

    return hv[1,1,0] - first - second - third - fourth - fifth

def combined_last_step(
    h: gtscript.Field[float], 

    hu: gtscript.Field[float],
    hv: gtscript.Field[float],
    hs: gtscript.Field[float],

    f: gtscript.Field[float],
    tg: gtscript.Field[float],

    huMidx: gtscript.Field[float],
    huMidy: gtscript.Field[float],
    hvMidx: gtscript.Field[float],
    hvMidy: gtscript.Field[float],
    hMidx: gtscript.Field[float],
    hMidy: gtscript.Field[float],
    cMidy: gtscript.Field[float],

    dx: gtscript.Field[float],
    dy1: gtscript.Field[float],
    dxc: gtscript.Field[float],
    dyc: gtscript.Field[float],
    dy1c: gtscript.Field[float],

    hnew: gtscript.Field[float],
    unew: gtscript.Field[float],
    vnew: gtscript.Field[float],
    
    VxMidnew: gtscript.Field[float],
    Vy1Midnew: gtscript.Field[float],
    Vy2Midnew: gtscript.Field[float],
    
    *,
    
    dt: float,
    g: float,
    a: float
):
    from __gtscript__ import PARALLEL, computation, interval
    from __externals__ import compute_hnew, compute_hunew, compute_hvnew

    with computation(PARALLEL), interval(...):
        
        # Update fluid height
        hnew=compute_hnew(h, dt, dxc, huMidx, dy1c, hvMidy, cMidy)

        
        # Update longitudinal moment

        #---RECPLACED NUMPY WHERE----------------------------------------
        temp_bool=hMidx > 0.0        
        UxMid_temp = temp_bool* (huMidx * huMidx / hMidx + 0.5 * g * hMidx * hMidx)
        temp_bool=hMidx <= 0.0
        UxMid=UxMid_temp +  temp_bool*(0.5 * g * hMidx * hMidx)

        if hMidy > 0.0:        
            UyMid = hvMidy * cMidy * huMidy / hMidy
        else:
            UyMid = 0.0
        #---RECPLACED NUMPY WHERE----------------------------------------

        hunew = compute_hunew(
            hu=hu, dt=dt, dxc=dxc, UxMid=UxMid, dy1c=dy1c, UyMid=UyMid, 
            f=f, huMidx=huMidx, hMidx=hMidx, huMidy=huMidy,hMidy=hMidy,tg=tg,
            a=a, hvMidx=hvMidx,hvMidy=hvMidy,g=g, hs=hs,dx=dx
        )



        # Update latitudinal moment
            
        #---RECPLACED NUMPY WHERE----------------------------------------
        VxMid = (hvMidx[0,0,0] * huMidx[0,0,0] / hMidx[0,0,0]) if (hMidx[0,0,0] > 0.0) else 0.0   
        if hMidy > 0.0:
            Vy1Mid = hvMidy * hvMidy / hMidy * cMidy
        else:
            Vy1Mid = 0.0
        Vy2Mid = 0.5 * g * hMidy * hMidy
        #---RECPLACED NUMPY WHERE----------------------------------------

        hvnew = compute_hvnew(
            hv=hv, dt=dt, dxc=dxc, VxMid=VxMid, dy1c=dy1c, Vy1Mid=Vy1Mid,
            dyc=dyc, Vy2Mid=Vy2Mid, f=f, huMidx=huMidx, hMidx=hMidx, huMidy=huMidy,
            hMidy=hMidy, tg=tg, a=a, g=g, hs=hs, dy1=dy1) 


        # Come back to original variables
        unew = hunew / hnew
        vnew = hvnew / hnew
        

# --- DIFFUSION --- #
def compute_Lapacian(
    q: gtscript.Field[float],
    Ax: gtscript.Field[float],
    Bx: gtscript.Field[float],
    Cx: gtscript.Field[float],
    
    Ay: gtscript.Field[float],
    By: gtscript.Field[float],
    Cy: gtscript.Field[float],
    
    qtemp: gtscript.Field[float],
    qnew: gtscript.Field[float],
    *,
    dt: float,
    nu: float,
    ):
    
    from __gtscript__ import PARALLEL, computation, interval
    
    with computation(PARALLEL), interval(...):
        # Compute second order derivative along longitude
        qxx=Ax[1,0,0]*(Ax[1,0,0]*q[2,2,0]+Bx[1,0,0]*q[3,2,0]+Cx[1,0,0]*q[1,2,0]) + \
            Bx[1,0,0]*(Ax[2,0,0]*q[3,2,0]+Bx[2,0,0]*q[4,2,0]+Cx[2,0,0]*q[2,2,0]) + \
            Cx[1,0,0]*(Ax[0,0,0]*q[1,2,0]+Bx[0,0,0]*q[2,2,0]+Cx[0,0,0]*q[0,2,0])

        qyy=Ay[0,1,0]*(Ay[0,1,0]*q[2,2,0]+By[0,1,0]*q[2,3,0]+Cy[0,1,0]*q[2,1,0]) + \
            By[0,1,0]*(Ay[0,2,0]*q[2,3,0]+By[0,2,0]*q[2,4,0]+Cy[0,2,0]*q[2,2,0]) + \
            Cy[0,1,0]*(Ay[0,0,0]*q[2,1,0]+By[0,0,0]*q[2,2,0]+Cy[0,0,0]*q[2,0,0])

        qnew=qtemp+dt*nu*(qxx+qyy)

        
#-------------------TORUS_SPECIFIC-------------------------------------------------------------------------------------------------------------------------
#---------------------------------TORUS_SPECIFIC-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------TORUS_SPECIFIC---------------------------------------------------------------------------------------------
#--------------------------------------------------------------TORUS_SPECIFIC------------------------------------------------------------------------------


@gtscript.function
def compute_huMidx_torus(dx, dt, hu, hv, Ux, f, u, sgMidx, cgMidx, r_major, r_minor):
    return (0.5 * (hu[1, 0,0] + hu[0,0,0]) - 0.5 * dt / dx[0,0,0] * (Ux[1,0,0] - Ux[0,0,0]) + \
            0.5 * dt * \
            (0.5 * (f[1,0,0] + f[0,0,0]) + \
            0.5 * (u[1,0,0] + u[0,0,0]) * sgMidx / (r_major + r_minor * cgMidx) ) * \
            (0.5 * (hv[1,0,0] + hv[0,0,0])))   

@gtscript.function
def compute_hvMidx_torus(dx, dt, hu, hv, Vx, f, u, sgMidx, cgMidx, r_major, r_minor):
    first= 0.5 * (hv[1, 0,0] + hv[0,0,0])
    second = 0.5 * dt / dx[0,0,0] * (Vx[1,0,0] - Vx[0,0,0])
    third = 0.5 * dt * (0.5 * (f[1,0,0] + f[0,0,0]) + 0.5 * (u[1,0,0] + u[0,0,0]) * sgMidx / (r_major + r_minor * cgMidx)) * (0.5 * (hu[1,0,0] + hu[0,0,0]))
                        
    return first - second - third 


def x_staggered_first_step_torus(
    u: gtscript.Field[float], 
    v: gtscript.Field[float],
    h: gtscript.Field[float],

    hu: gtscript.Field[float],
    hv: gtscript.Field[float],
    f: gtscript.Field[float],
    g_torus_r: gtscript.Field[float],

    dx: gtscript.Field[float],
    sgMidx: gtscript.Field[float],
    cgMidx: gtscript.Field[float],

    hMidx: gtscript.Field[float],
    huMidx: gtscript.Field[float],
    hvMidx: gtscript.Field[float],
    *,
    dt: float,
    r_major: float,
    r_minor: float
    ):
    from __gtscript__ import PARALLEL, computation, interval
    from __externals__ import compute_hMidx, compute_huMidx_torus, compute_hvMidx_torus

    with computation(PARALLEL), interval(...):
            Ux = hu * u + 0.5 * g_torus_r * h * h
            Vx = hu * v

            # Mid-point value for h along x
            hMidx = compute_hMidx(dx=dx, dt=dt, h=h, hu=hu)
            huMidx = compute_huMidx_torus(dx=dx, dt=dt, hu=hu, hv=hv, Ux=Ux, f=f, u=u, sgMidx=sgMidx, cgMidx=cgMidx, r_major=r_major, r_minor=r_minor)
            hvMidx = compute_hvMidx_torus(dx=dx, dt=dt, hu=hu, hv=hv, Vx=Vx, f=f, u=u, sgMidx=sgMidx, cgMidx=cgMidx, r_major=r_major, r_minor=r_minor)

            
@gtscript.function
def compute_huMidy_torus(dy1, dt, hu, hv, Uy, f, u, sgMidy, cgMidy, r_major, r_minor):
    return (0.5 * (hu[0,1,0] + hu[0,0,0]) - 0.5 * dt / dy1[0,0,0] * (Uy[0,1,0] - Uy[0,0,0]) + \
        0.5 * dt * \
        (0.5 * (f[0,1,0] + f[0,0,0]) + \
        0.5 * (u[0,1,0] + u[0,0,0]) * sgMidy / (r_major + r_minor*cgMidy)) * \
        (0.5 * (hv[0,1,0] + hv[0,0,0])))
            
            
@gtscript.function
def compute_hvMidy_torus(dy1, dy, dt, hu, hv,  Vy1, Vy2, f, u, sgMidy, cgMidy, r_major, r_minor):
    return (0.5 * (hv[0,1,0] + hv[0,0,0]) \
            - 0.5 * dt / dy1[0,0,0] * (Vy1[0,1,0] - Vy1[0,0,0]) -  \
            0.5 * dt / dy[0,0,0] * (Vy2[0,1,0] - Vy2[0,0,0]) -\
            0.5 * dt * \
            (0.5 * (f[0,1,0] + f[0,0,0]) + \
            0.5 * (u[0,1,0] + u[0,0,0]) * sgMidy / (r_major + r_minor*cgMidy)) * \
            (0.5 * (hu[0,1,0] + hu[0,0,0])))



def y_staggered_first_step_torus(
    u: gtscript.Field[float], 
    v: gtscript.Field[float],
    h: gtscript.Field[float],

    hu: gtscript.Field[float],
    hv: gtscript.Field[float],
    v1: gtscript.Field[float],
    f: gtscript.Field[float],
    g_torus_r: gtscript.Field[float],

    dy: gtscript.Field[float],
    dy1: gtscript.Field[float],
    sgMidy: gtscript.Field[float],
    cgMidy: gtscript.Field[float],

    hMidy: gtscript.Field[float],
    huMidy: gtscript.Field[float],
    hvMidy: gtscript.Field[float],   
    *,
    dt: float,
    r_major: float,
    r_minor: float):
    from __gtscript__ import PARALLEL, computation, interval
    from __externals__ import compute_hMidy, compute_huMidy_torus, compute_hvMidy_torus
    
    with computation(PARALLEL), interval(...):
            hv1 = h * v1
            Uy = hu * v1
            Vy1 = hv * v1
            Vy2 = 0.5 * g_torus_r * h * h

            # Mid-point value for h along y
            hMidy = compute_hMidy(dy1=dy1, dt=dt, h=h, hv1=hv1)
            huMidy = compute_huMidy_torus(dy1=dy1, dt=dt, hu=hu, hv=hv, Uy=Uy, f=f, u=u, sgMidy=sgMidy, cgMidy=cgMidy, r_major=r_major, r_minor=r_minor)
            hvMidy = compute_hvMidy_torus(dy1=dy1, dy=dy, dt=dt, hu=hu, hv=hv,  Vy1=Vy1, Vy2=Vy2, f=f, u=u, sgMidy=sgMidy, cgMidy=cgMidy, r_major=r_major, r_minor=r_minor)    





@gtscript.function
def compute_hunew_torus(hu, dt, dxc, UxMid, dy1c, UyMid, f, huMidx, hMidx, huMidy,hMidy,sg, cg, r_major, r_minor, hvMidx,hvMidy,g_torus_r, hs,dx):
    first= dt / dxc * (UxMid[1,1,0] - UxMid[0,1,0])

    second= dt / dy1c * (UyMid[1,1,0] - UyMid[1,0,0])

    third= dt * (f[1,1,0] +  0.25 * (huMidx[0,1,0] / hMidx[0,1,0] + \
                                    huMidx[1,1,0] / hMidx[1,1,0] + \
                                    huMidy[1,0,0] / hMidy[1,0,0] + \
                                    huMidy[1,1,0] / hMidy[1,1,0]) * \
                                    sg / (r_major+r_minor*cg)) * \
                                    0.25 * (hvMidx[0,1,0] + hvMidx[1,1,0] + hvMidy[1,0,0] + hvMidy[1,1,0])

    fourth= dt * g_torus_r[1,1,0] * 0.25 * (hMidx[0,1,0] + hMidx[1,1,0] + hMidy[1,0,0] + hMidy[1,1,0]) * \
                (hs[2,1,0] - hs[0,1,0]) / (dx[0,1,0] + dx[1,1,0])

    return hu[1,1,0] - first - second + third - fourth

@gtscript.function
def compute_hvnew_torus(hv, dt, dxc, VxMid, dy1c, Vy1Mid, dyc, Vy2Mid, f, huMidx, hMidx, huMidy, hMidy, sg, cg, r_major, r_minor, g_torus_r, hs, dy1, g_torus_theta):

    first  = dt / dxc * (VxMid[1,1,0] - VxMid[0,1,0])
    second = dt / dy1c * (Vy1Mid[1,1,0] - Vy1Mid[1,0,0])
    third  = dt / dyc * (Vy2Mid[1,1,0] - Vy2Mid[1,0,0])

    fourth = dt * (f[1,1,0] + 0.25 * (huMidx[0,1,0] / hMidx[0,1,0] + \
                                    huMidx[1,1,0] / hMidx[1,1,0] + \
                                    huMidy[1,0,0] / hMidy[1,0,0] + \
                                    huMidy[1,1,0] / hMidy[1,1,0]) * \
                                    sg / (r_major+r_minor*cg)) * \
                                    0.25 * (huMidx[0,1,0] + huMidx[1,1,0] + huMidy[1,0,0] + huMidy[1,1,0])#

    fifth  = dt * g_torus_r[1,1,0] * 0.25 * (hMidx[0,1,0] + hMidx[1,1,0] + hMidy[1,0,0] + hMidy[1,1,0]) * (hs[1,2,0] - hs[1,0,0]) / (dy1[1,1,0] + dy1[1,0,0]) 
    
    sixth = dt * g_torus_theta[1,1,0] * 0.25 * (hMidx[0,1,0] + hMidx[1,1,0] + hMidy[1,0,0] + hMidy[1,1,0])
    
    return hv[1,1,0] - first - second - third - fourth - fifth + sixth

def combined_last_step_torus(
    h: gtscript.Field[float], 

    hu: gtscript.Field[float],
    hv: gtscript.Field[float],
    hs: gtscript.Field[float],

    f: gtscript.Field[float],
    sg: gtscript.Field[float],
    cg: gtscript.Field[float],
    g_torus_r: gtscript.Field[float],
    g_torus_theta: gtscript.Field[float],
    grav_r_Midy: gtscript.Field[float],

    huMidx: gtscript.Field[float],
    huMidy: gtscript.Field[float],
    hvMidx: gtscript.Field[float],
    hvMidy: gtscript.Field[float],
    hMidx: gtscript.Field[float],
    hMidy: gtscript.Field[float],
    cMidy: gtscript.Field[float],

    dx: gtscript.Field[float],
    dy1: gtscript.Field[float],
    dxc: gtscript.Field[float],
    dyc: gtscript.Field[float],
    dy1c: gtscript.Field[float],

    hnew: gtscript.Field[float],
    unew: gtscript.Field[float],
    vnew: gtscript.Field[float],
    
    VxMidnew: gtscript.Field[float],
    Vy1Midnew: gtscript.Field[float],
    Vy2Midnew: gtscript.Field[float],
    
    *,
    
    dt: float,
    r_major: float,
    r_minor: float):
    from __gtscript__ import PARALLEL, computation, interval
    from __externals__ import compute_hnew, compute_hunew_torus, compute_hvnew_torus

    with computation(PARALLEL), interval(...):
        
        # Update fluid height
        hnew=compute_hnew(h, dt, dxc, huMidx, dy1c, hvMidy, cMidy)

        
        # Update longitudinal moment

        #---RECPLACED NUMPY WHERE----------------------------------------
        temp_bool=hMidx > 0.0        
        UxMid_temp = temp_bool* (huMidx * huMidx / hMidx + 0.5 * g_torus_r[1,1,0] * hMidx * hMidx) #IS THIS CORRECT???
        temp_bool=hMidx <= 0.0
        UxMid=UxMid_temp +  temp_bool*(0.5 * g_torus_r[1,1,0] * hMidx * hMidx)

        if hMidy > 0.0:        
            UyMid = hvMidy * cMidy * huMidy / hMidy
        else:
            UyMid = 0.0
        #---RECPLACED NUMPY WHERE----------------------------------------

        hunew = compute_hunew_torus(
            hu=hu, dt=dt, dxc=dxc, UxMid=UxMid, dy1c=dy1c, UyMid=UyMid, 
            f=f, huMidx=huMidx, hMidx=hMidx, huMidy=huMidy,hMidy=hMidy,sg=sg, cg=cg, r_major=r_major, r_minor=r_minor, hvMidx=hvMidx,hvMidy=hvMidy,g_torus_r=g_torus_r, hs=hs,dx=dx
        )


        # Update latitudinal moment
            
        #---RECPLACED NUMPY WHERE----------------------------------------
        VxMid = (hvMidx[0,0,0] * huMidx[0,0,0] / hMidx[0,0,0]) if (hMidx[0,0,0] > 0.0) else 0.0   
        if hMidy > 0.0:
            Vy1Mid = hvMidy * hvMidy / hMidy * cMidy
        else:
            Vy1Mid = 0.0
        Vy2Mid = 0.5 * grav_r_Midy[1,0,0] * hMidy * hMidy #IS THIS CORRECT???
        #---RECPLACED NUMPY WHERE----------------------------------------

        hvnew = compute_hvnew_torus(
            hv=hv, dt=dt, dxc=dxc, VxMid=VxMid, dy1c=dy1c, Vy1Mid=Vy1Mid,
            dyc=dyc, Vy2Mid=Vy2Mid, f=f, huMidx=huMidx, hMidx=hMidx, huMidy=huMidy,
            hMidy=hMidy, sg=sg, cg=cg, r_major=r_major, r_minor=r_minor, g_torus_r=g_torus_r, hs=hs, dy1=dy1, g_torus_theta=g_torus_theta) 


        # Come back to original variables
        unew = hunew / hnew
        vnew = hvnew / hnew
