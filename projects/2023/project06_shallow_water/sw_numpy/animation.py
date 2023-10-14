"""
Script to plot results coming from finite difference SWES solver.
The user may specify all the solver options and let the code look for
the correspondent dataset or directly give the path to the dataset.
In any case, the user should specify all the settings.
"""

import os
import pickle
import numpy as np
import math
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from matplotlib.animation import PillowWriter 
import cartopy.crs as ccrs
import gt4py
#-----------------------------------------------------------
#from mpl_toolkits.basemap import Basemap
#-----------------------------------------------------------


# --- SETTINGS --- #

# Choose what to plot:
# 	* h (fluid height)
#	* u (longitudinal velocity)
#	* v (latitudinal velocity)
# 	* quiver (velocity quiver-plot)
#	* vorticity (relative vorticity)
#	* mesh
what_to_plot = 'h'

# Choose projection for height plot:
#	* cyl (cylindrical)
#	* ortho (orthogonal)
projection = 'cyl'

# Choose if you want to save the movie and if yes which format do you prefer:
#	* mp4
#	* mpg
# and the frames per seconds
save_movie = True
movie_format = 'gif'
fps = 15


# --- LOAD DATA --- #

GRIDTOOLS_ROOT = '.' #os.environ.get('GRIDTOOLS_ROOT')
#baseName = GRIDTOOLS_ROOT + '/data/swes-numpy-0-M180-N90-T5-1-'
baseName = GRIDTOOLS_ROOT + '/data/swes-numpy-0-M180-N90-T2-1-'

# Load h
with open(baseName + 'h', 'rb') as f:
    M, N, t, phi, theta, h = pickle.load(f)

# Load u
with open(baseName + 'u', 'rb') as f:
    M, N, t, phi, theta, u = pickle.load(f)

# Load v
with open(baseName + 'v', 'rb') as f:
    M, N, t, phi, theta, v = pickle.load(f)

# Apply wrap-around boundary conditions on data coming from
# any original solver implementation
phi = np.concatenate((phi, phi[0:1,:]), axis = 0)
theta = np.concatenate((theta, theta[0:1,:]), axis = 0)
h = np.concatenate((h, h[0:1,:]), axis = 0)
u = np.concatenate((u, u[0:1,:]), axis = 0)
v = np.concatenate((v, v[0:1,:]), axis = 0)
Nt = h.shape[2]


# --- PLOT HEIGHT --- #

if (what_to_plot == 'h'):

    fig1 = plt.figure(figsize=[15,8])

    if (save_movie):

        # Instantiate writer class
        
        metadata = dict(artist='Matplotlib', comment='')
        #writer = manimation.FFMpegWriter(fps=fps, metadata=metadata, bitrate=3500)
        
        writer = manimation.PillowWriter(fps=fps)#, bitrate=3500)
        
        #PillowWriter
        #FFMpegWriter = manimation.writers["ffmpeg"]
        #writer = FFMpegWriter(fps = fps)

        with writer.saving(fig1, baseName + 'h.' + movie_format, Nt):

            for n in range(Nt):

                if (n == 0):
                    if (projection == 'cyl'):
                        #m1 = Basemap(projection = 'cyl',
                        #             llcrnrlat = -90,
                        #             urcrnrlat = 90,
                        #             llcrnrlon = 0,
                        #             urcrnrlon = 360)
                        proj=ccrs.PlateCarree(central_longitude=0.0, globe=None)
                        
                    elif (projection == 'ortho'):
                        proj=ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0, globe=None)
                    
                    #m1=plt.subplot(1,1,1, projection=proj)
                    m1=plt.subplot(1,1,1)
                    
                    #x1, y1 = m1(phi*180.0/math.pi, theta*180.0/math.pi)
                    x1, y1 = phi*180.0/math.pi, theta*180.0/math.pi
                    x1[1,:] = 0.0

                    #m1.drawcoastlines()
                    #m1.drawparallels(np.arange(-80.,81.,20.))
                    #m1.drawmeridians(np.arange(0.,360.,20.))
                    #m1.drawmapboundary(fill_color='white')
                else:
                    for coll in surf.collections:
                        plt.gca().collections.remove(coll)

                x1[1,:] = 0.0
                x1[-2,:] = 360.0
                color_scale = np.linspace(h.min(), h.max(), 30, endpoint = True)
                #surf = m1.contourf(x1[1:-1,:], y1[1:-1,:], h[:,:,n], color_scale,  transform = ccrs.PlateCarree())
                surf = m1.contourf(x1[1:-1,:], y1[1:-1,:], h[:,:,n], color_scale)

                
                # Comment the following line to hide the title
                plt.title('Fluid height [m]: time = %5.2f hours\n' % (t[n] / 3600.0))

                #Uncomment the following lines to show the colorbar on the left side of the plot
                #if (n == 0):
               # 	fig1.colorbar(surf)
               # else:
               # 	surf.autoscale()

                writer.grab_frame()

    else:

        for n in range(Nt):

            if (n == 0):
                if (projection == 'cyl'):
                    m1 = Basemap(projection = 'cyl',
                                 llcrnrlat = -90,
                                 urcrnrlat = 90,
                                 llcrnrlon = 0,
                                 urcrnrlon = 360)
                elif (projection == 'ortho'):
                    m1 = Basemap(projection = 'ortho',
                                 lat_0 = 45,
                                 lon_0 = 8.9511, # Lugano longitude
                                 resolution='l')

                x1, y1 = m1(phi*180.0/math.pi, theta*180.0/math.pi)

                m1.drawcoastlines()
                m1.drawparallels(np.arange(-80.,81.,20.))
                m1.drawmeridians(np.arange(0.,360.,20.))
                m1.drawmapboundary(fill_color='white')
            else:
                for coll in surf.collections:
                    plt.gca().collections.remove(coll)

            x1[1,:] = 0.0
            x1[-2,:] = 360.0
            color_scale = np.linspace(h[:,:,:].min(), h[:,:,:].max(), 30, endpoint = True)
            surf = m1.contourf(x1[:,:], y1[:,:], h[:,:,n], color_scale)

            # Comment the following line to hide the title
            plt.title('Fluid height [m]: time = %5.2f hours\n' % (t[n] / 3600.0))

            # Uncomment the following lines to show the colorbar on the left side of the plot
            #if (n == 0):
            #	fig1.colorbar(surf)
            #else:
            #	surf.autoscale()

            plt.draw()
            plt.pause(0.1)

    plt.show()


# --- PLOT LONGITUDINAL VELOCITY --- #

if (what_to_plot == 'u'):

    fig2 = plt.figure(figsize=[15,8])

    if (save_movie):

        # Instantiate writer class
        FFMpegWriter = manimation.writers["ffmpeg"]
        writer = FFMpegWriter(fps = fps)

        with writer.saving(fig2, baseName + 'u.' + movie_format, Nt):

            for n in range(Nt):

                if (n == 0):
                    if (projection == 'cyl'):
                        m2 = Basemap(projection = 'cyl',
                                     llcrnrlat = -90,
                                     urcrnrlat = 90,
                                     llcrnrlon = 0,
                                     urcrnrlon = 360)
                    elif (projection == 'ortho'):
                        m2 = Basemap(projection = 'ortho',
                                     lat_0 = 45,
                                     lon_0 = 8.9511, # Lugano longitude
                                     resolution='l')

                    x2, y2 = m2(phi*180.0/math.pi, theta*180.0/math.pi)

                    m2.drawcoastlines()
                    m2.drawparallels(np.arange(-80.,81.,20.))
                    m2.drawmeridians(np.arange(0.,360.,20.))
                    m2.drawmapboundary(fill_color='white')
                else:
                    for coll in surf.collections:
                        plt.gca().collections.remove(coll)

                color_scale = np.linspace(u.min(), u.max(), 30, endpoint = True)
                surf = m2.contourf(x2[1:-1,:], y2[1:-1,:], u[:,:,n], color_scale)

                # Comment the following line to hide the title
                plt.title('Longitudinal velocity [m/s]: time = %5.2f hours\n' % (t[n] / 3600.0))

                # Uncomment the following lines to show the colorbar on the left side of the plot
                #if (n == 0):
                #	fig2.colorbar(surf)
                #else:
                #	surf.autoscale()

                writer.grab_frame()

    else:

        for n in range(Nt):

            if (n == 0):
                if (projection == 'cyl'):
                    m2 = Basemap(projection = 'cyl',
                                 llcrnrlat = -90,
                                 urcrnrlat = 90,
                                 llcrnrlon = 0,
                                 urcrnrlon = 360)
                elif (projection == 'ortho'):
                    m2 = Basemap(projection = 'ortho',
                                 lat_0 = 45,
                                 lon_0 = 8.9511, # Lugano longitude
                                 resolution='l')

                x2, y2 = m2(phi*180.0/math.pi, theta*180.0/math.pi)

                m2.drawcoastlines()
                m2.drawparallels(np.arange(-80.,81.,20.))
                m2.drawmeridians(np.arange(0.,360.,20.))
                m2.drawmapboundary(fill_color='white')
            else:
                for coll in surf.collections:
                    plt.gca().collections.remove(coll)

            color_scale = np.linspace(u.min(), u.max(), 30, endpoint = True)
            surf = m2.contourf(x2[1:-1,:], y2[1:-1,:], u[:,:,n], color_scale)

            # Comment the following line to hide the title
            plt.title('Longitudinal velocity [m/s]: time = %5.2f hours\n' % (t[n] / 3600.0))

            # Uncomment the following lines to show the colorbar on the left side of the plot
            #if (n == 0):
            #	fig2.colorbar(surf)
            #else:
            #	surf.autoscale()

            plt.draw()
            plt.pause(0.1)

    plt.show()


# --- PLOT LATITUDINAL VELOCITY --- #

if (what_to_plot == 'v'):

    fig3 = plt.figure(figsize=[15,8])

    if (save_movie):

        # Instantiate writer class
        FFMpegWriter = manimation.writers["ffmpeg"]
        writer = FFMpegWriter(fps = fps)

        with writer.saving(fig3, baseName + 'v.' + movie_format, Nt):

            for n in range(Nt):

                if (n == 0):
                    if (projection == 'cyl'):
                        m3 = Basemap(projection = 'cyl',
                                     llcrnrlat = -90,
                                     urcrnrlat = 90,
                                     llcrnrlon = 0,
                                     urcrnrlon = 360)
                    elif (projection == 'ortho'):
                        m3 = Basemap(projection = 'ortho',
                                     lat_0 = 45,
                                     lon_0 = 8.9511, # Lugano longitude
                                     resolution='l')

                    x3, y3 = m3(phi*180.0/math.pi, theta*180.0/math.pi)

                    m3.drawcoastlines()
                    m3.drawparallels(np.arange(-80.,81.,20.))
                    m3.drawmeridians(np.arange(0.,360.,20.))
                    m3.drawmapboundary(fill_color='white')
                else:
                    for coll in surf.collections:
                        plt.gca().collections.remove(coll)

                color_scale = np.linspace(v.min(), v.max(), 30, endpoint = True)
                surf = m3.contourf(x3[1:-1,:], y3[1:-1,:], v[:,:,n], color_scale)

                # Comment the following line to hide the title
                plt.title('Latitudinal velocity [m/s]: time = %5.2f hours\n' % (t[n] / 3600.0))

                # Uncomment the following lines to show the colorbar on the left side of the plot
                #if (n == 0):
                #	fig3.colorbar(surf)
                #else:
                #	surf.autoscale()

                writer.grab_frame()

    else:

        for n in range(Nt):

            if (n == 0):
                if (projection == 'cyl'):
                    m3 = Basemap(projection = 'cyl',
                                 llcrnrlat = -90,
                                 urcrnrlat = 90,
                                 llcrnrlon = 0,
                                 urcrnrlon = 360)
                elif (projection == 'ortho'):
                    m3 = Basemap(projection = 'ortho',
                                 lat_0 = 45,
                                 lon_0 = 8.9511, # Lugano longitude
                                 resolution='l')

                x3, y3 = m3(phi*180.0/math.pi, theta*180.0/math.pi)

                m3.drawcoastlines()
                m3.drawparallels(np.arange(-80.,81.,20.))
                m3.drawmeridians(np.arange(0.,360.,20.))
                m3.drawmapboundary(fill_color='white')
            else:
                for coll in surf.collections:
                    plt.gca().collections.remove(coll)

            color_scale = np.linspace(v.min(), v.max(), 30, endpoint = True)
            surf = m3.contourf(x3[1:-1,:], y3[1:-1,:], v[:,:,n], color_scale)

            # Comment the following line to hide the title
            plt.title('Latitudinal velocity [m/s]: time = %5.2f hours\n' % (t[n] / 3600.0))

            # Uncomment the following lines to show the colorbar on the left side of the plot
            #if (n == 0):
            #	fig3.colorbar(surf)
            #else:
            #	surf.autoscale()

            plt.draw()
            plt.pause(0.1)

    plt.show()


# --- PLOT VELOCITY QUIVER PLOT --- #

if (what_to_plot == 'quiver'):

    fig4 = plt.figure(figsize=(15,7))

    if (save_movie):

        # Instantiate writer class
        FFMpegWriter = manimation.writers["ffmpeg"]
        writer = FFMpegWriter(fps = fps)

        with writer.saving(fig4, baseName + 'quiver.' + movie_format, Nt):

            for n in range(Nt):

                if (n == 0):
                    m4 = Basemap(projection = 'cyl',
                                 llcrnrlat = -90,
                                 urcrnrlat = 90,
                                 llcrnrlon = 0,
                                 urcrnrlon = 360)

                    x4, y4 = m4(phi*180.0/math.pi, theta*180.0/math.pi)

                    m4.drawcoastlines()
                    m4.drawparallels(np.arange(-80.,81.,20.))
                    m4.drawmeridians(np.arange(0.,360.,20.))
                    m4.drawmapboundary(fill_color='white')

                    Q = m4.quiver(x4[1:-1:10,0:-1:10], y4[1:-1:10,0:-1:10], u[1:-1:10,0:-1:10,0].transpose(), v[1:-1:10,0:-1:10,0].transpose(), color = 'blue')

                else:
                    Q.set_UVC(u[1:-1:10,0:-1:10,n].transpose(), v[1:-1:10,0:-1:10,n].transpose())

                # Comment the following line to hide the title
                plt.title('Velocity field [m/s]: time = %5.2f hours\n' % (t[n] / 3600.0))

                writer.grab_frame()

    else:

        for n in range(Nt):

            if (n == 0):
                m4 = Basemap(projection = 'cyl',
                             llcrnrlat = -90,
                             urcrnrlat = 90,
                             llcrnrlon = 0,
                             urcrnrlon = 360)

                x4, y4 = m4(phi*180.0/math.pi, theta*180.0/math.pi)

                m4.drawcoastlines()
                m4.drawparallels(np.arange(-80.,81.,20.))
                m4.drawmeridians(np.arange(0.,360.,20.))
                m4.drawmapboundary(fill_color='white')

                Q = m4.quiver(x4[1:-1:10,0:-1:10], y4[1:-1:10,0:-1:10], u[1:-1:10,0:-1:10,0].transpose(), v[1:-1:10,0:-1:10,0].transpose(), color = 'blue')

            else:
                Q.set_UVC(u[1:-1:10,0:-1:10,n].transpose(), v[1:-1:10,0:-1:10,n].transpose())

            # Comment the following line to hide the title
            plt.title('Velocity field [m/s]: time = %5.2f hours\n' % (t[n] / 3600.0))

            plt.draw()
            plt.pause(0.1)

    plt.show()


# --- PLOT VORTICITY --- #

if (what_to_plot == 'vorticity'):

    # Compute vorticity
    vorticity = np.zeros((M+3, N, 1), float)
    dphi = 360.0 / M
    dtheta = 180.0 / (N + 1.0)

    vorticity[:,1:-1,:] = - (u[:,2:,:] - u[:,:-2,:]) / (2.0 * dtheta)
    vorticity[:,0,:] = - (-1.5 * u[:,0,:] + 2.0 * u[:,1,:] - 0.5 * u[:,2,:]) / dtheta
    vorticity[:,-1,:] = - (1.5 * u[:,-1,:] - 2.0 * u[:,-2,:] + 0.5 * u[:,-3,:]) / dtheta

    vorticity[1:-1,:,:] += (v[2:,:,:] - v[:-2,:,:]) / (2.0 * dphi)
    vorticity[0,:,:] += v[1,:,:] - v[-2,:,:] / (2.0 * dphi)
    vorticity[-1,:,:] = vorticity[0,:,:]

    fig5 = plt.figure()

    if (save_movie):

        # Instantiate writer class
        FFMpegWriter = manimation.writers["ffmpeg"]
        writer = FFMpegWriter(fps = fps)

        with writer.saving(fig5, baseName + 'vorticity.' + movie_format, Nt):

            for n in range(Nt):

                if (n == 0):
                    if (projection == 'cyl'):
                        m5 = Basemap(projection = 'cyl',
                                     llcrnrlat = -90,
                                     urcrnrlat = 90,
                                     llcrnrlon = 0,
                                     urcrnrlon = 360)
                    elif (projection == 'ortho'):
                        m5 = Basemap(projection = 'ortho',
                                     lat_0 = 45,
                                     lon_0 = 8.9511, # Lugano longitude
                                     resolution = 'l')

                    x5, y5 = m5(phi*180.0/math.pi, theta*180.0/math.pi)

                    m5.drawcoastlines()
                    m5.drawparallels(np.arange(-80.,81.,20.))
                    m5.drawmeridians(np.arange(0.,360.,20.))
                    m5.drawmapboundary(fill_color='white')
                else:
                    for coll in surf.collections:
                        plt.gca().collections.remove(coll)

                color_scale = np.linspace(vorticity.min(), vorticity.max(), 30, endpoint = True)
                surf = m5.contourf(x5[1:-1,:], y5[1:-1,:], vorticity[:,:,n], color_scale)

                # Comment the following line to hide the title
                plt.title('Relative vorticity magnitude [m/s]: time = %5.2f hours\n' % (t[n] / 3600.0))

                # Uncomment the following lines to show the colorbar on the left side of the plot
                #if (n == 0):
                #	fig5.colorbar(surf)
                #else:
                #	surf.autoscale()

                writer.grab_frame()

    else:

        for n in range(Nt):

            if (n == 0):
                if (projection == 'cyl'):
                    m5 = Basemap(projection = 'cyl',
                                 llcrnrlat = -90,
                                 urcrnrlat = 90,
                                 llcrnrlon = 0,
                                 urcrnrlon = 360)
                elif (projection == 'ortho'):
                    m5 = Basemap(projection = 'ortho',
                                 lat_0 = 45,
                                 lon_0 = 8.9511, # Lugano longitude
                                 resolution = 'l')

                x5, y5 = m5(phi*180.0/math.pi, theta*180.0/math.pi)

                m5.drawcoastlines()
                m5.drawparallels(np.arange(-80.,81.,20.))
                m5.drawmeridians(np.arange(0.,360.,20.))
                m5.drawmapboundary(fill_color='white')
            else:
                for coll in surf.collections:
                    plt.gca().collections.remove(coll)

            color_scale = np.linspace(vorticity.min(), vorticity.max(), 30, endpoint = True)
            surf = m5.contourf(x5[1:-1,:], y5[1:-1,:], vorticity[:,:,n])

            # Comment the following line to hide the title
            plt.title('Relative vorticity magnitude [m/s]: time = %5.2f hours\n' % (t[n] / 3600.0))

            # Uncomment the following lines to show the colorbar on the left side of the plot
            #if (n == 0):
            #	fig5.colorbar(surf)
            #else:
            #	surf.autoscale()

            plt.draw()
            plt.pause(0.1)

    plt.show()


# --- PLOT THE MESH --- #

if (what_to_plot == 'mesh'):

    fig6 = plt.figure()

    if (projection == 'cyl'):
        m = Basemap(projection = 'cyl',
                     llcrnrlat = -90,
                     urcrnrlat = 90,
                     llcrnrlon = -22,
                     urcrnrlon = 382)
    elif (projection == 'ortho'):
        m = Basemap(projection = 'ortho',
                     lat_0 = 45,
                     lon_0 = 8.9511, # Lugano longitude
                     resolution='l')
    elif (projection == 'npstere'):
        m = Basemap(projection='npstere',
                    boundinglat=45,
                    lon_0=8.9511,
                    resolution='l')

    phi1D = np.linspace(-20.,380.,21)
    theta1D = np.array([-88, -70, -50, -30, -10, 10, 30, 50, 70, 90])
    phi, theta = np.meshgrid(phi1D, theta1D, indexing = 'ij')

    x, y = m(phi, theta)

    m.drawcoastlines()
    m.drawparallels([-88, -70, -50, -30, -10, 10, 30, 50, 70, 90])
    m.drawmeridians(np.arange(-20.,380.,20.))
    m.drawmapboundary(fill_color='white')

    m.scatter(x, y, 80, marker = 'o', color = 'r')
    #m.scatter(x[:,-3:-1], y[:,-3:-1], 70, marker = 'o', color = 'g')
    #m.scatter(x[:,-1], y[:,-1], 55, marker = 'o', color = 'b')
    m.scatter(x[1:-1,1:-1], y[1:-1,1:-1], 80, marker = 'o', color = 'g')
    #m.scatter(x[0,:], y[0,:], 60, marker = 'o', color = 'b')
    #m.scatter(x[-1,:], y[-1,:], 60, marker = 'o', color = 'b')

    plt.show()


