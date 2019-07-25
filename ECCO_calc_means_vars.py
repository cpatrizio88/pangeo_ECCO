#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:40:34 2019

@author: cpatrizio
"""

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

import sys
sys.path.append('/Users/cpatrizio/repos/ECCO/ECCOv4-py')
sys.path.append('/Users/cpatrizio/repos/')
import ecco_v4_py
import ecco_v4_py as ecco
#import xscale
import numpy as np
import xarray as xr
import pandas as pd
import cdms2 as cdms2
import cdutil
import xgcm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
from ocean_atmosphere.misc_fns import linear_detrend, detrend, detrend_ECCO, an_ave, spatial_ave, calc_AMO, running_mean, calc_NA_globeanom, detrend_separate, detrend_common, butter_lowpass_filter, corr2_coeff, cov2_coeff



#add to change plotting parameters
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'axes.titlesize': 20})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 18})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
matplotlib.rcParams.update({'axes.labelsize': 16})
matplotlib.rcParams.update({'ytick.labelsize': 16})
matplotlib.rcParams.update({'xtick.labelsize': 16})

fpath = '/Users/cpatrizio/data/ECCO'
fout = '/Users/cpatrizio/figures/ECCO/'

fgrid = fpath + '/nctiles_grid/'
fsnap = fpath + '/nctiles_monthly_snapshots/'
fmonthly = fpath + '/nctiles_monthly/'

#fname = 'THETA/THETA.0002.nc'
ftheta = 'THETA/'
fQs = 'TFLUX/'
fADVr_TH = 'ADVr_TH/'
fADVx_TH = 'ADVx_TH/'
fADVy_TH = 'ADVy_TH/'


#fname_grid = 'grid.0002.nc'


#theta_dataset = xr.open_dataset(fmonthly + fname).load()

#fnames = glob.glob(fmonthly + ftheta + 'THETA*.nc')

#Note: variables should be in the format VARNAME_NN.nc where NN is the tile number. This needs to be fixed in ECCO tutorial.

#Tiles for regions: 


#Northern Hemisphere
#tiles_load = (2,5,7,10)

#tiles_load = (2,10)

tiles_load = tuple(range(13))
num_levs_load = 10

grid = ecco.load_ecco_grid_from_tiles_nc(fgrid, k_subset=range(num_levs_load), dask_chunk = False)
grid.attrs = []


#Note: for some reason grid has dimension names that are inconsistent with other data
grid = grid.rename({'i1': 'k', 'i2': 'j', 'i3': 'i'})



#i1 : time
#i2 : z
#i3 : y
#i4 : x 


theta_dataset = ecco.load_ecco_var_from_tiles_nc(fmonthly + ftheta,\
                                                 'THETA', \
                                                 tiles_to_load = tiles_load, \
                                                 k_subset = range(num_levs_load), \
                                                 dask_chunk = True, \
                                                 less_output=False)

Qs_dataset = ecco.load_ecco_var_from_tiles_nc(fmonthly + fQs,\
                                                 'TFLUX', \
                                                 tiles_to_load = tiles_load, \
                                                 #k_subset = range(20), \
                                                 dask_chunk = False, \
                                                 less_output=False)

ADVr_TH_dataset = ecco.load_ecco_var_from_tiles_nc(fmonthly + fADVr_TH, \
                                                 'ADVr_TH', \
                                                 tiles_to_load = tiles_load, \
                                                 k_subset = range(num_levs_load), \
                                                 dask_chunk = False, \
                                                 less_output=False)

ADVx_TH_dataset = ecco.load_ecco_var_from_tiles_nc(fmonthly + fADVx_TH,\
                                                 'ADVx_TH', \
                                                 tiles_to_load = tiles_load, \
                                                 k_subset = range(num_levs_load), \
                                                 dask_chunk = False, \
                                                 less_output=False)

ADVy_TH_dataset = ecco.load_ecco_var_from_tiles_nc(fmonthly + fADVy_TH,\
                                                 'ADVy_TH', \
                                                 tiles_to_load = tiles_load, \
                                                 k_subset = range(num_levs_load), \
                                                 dask_chunk = False, \
                                                 less_output=False)

#ADVr_TH = ADVr_TH.rename({'i1' : 'k', 'i2')

#Qs_plt = Qs_dataset.TFLUX.isel(i1=10,i2=1)

#To get advective terms in W/m^2 need to multiply by (rho*c_p_deltaz_{k})/(v_{i,j,k})
# v_{i,j,k} = h_{i,j,k}*A_{i,j}*deltaz_{k}

#Also need to divide Qs by h_{i,j,k} to compare with other terms (grid-cell relative thickness? hFacC)

rho=1029
c_p=3994

deltaz = grid.DRF
h = grid.hFacC
A = grid.RAC
v = h*A*deltaz



#theta = theta_dataset.THETA.isel(i2=range(20))
#theta = theta.mean(axis=1)
#theta_dataset.attrs =[]




#tmp_plt = tmp_plt.where(ecco_ds.hFacC.isel(k=0) !=0)

#ecco.plot_tiles(tmp_plt, cmin=-2, cmax=28, \
#                show_colorbar=True, fig_size=8,\
#               #layout='latlon',rotate_to_latlon=True,\
#               cmap = plt.cm.cubehelix_r, show_tile_labels=True, \
#               Arctic_cap_tile_location=10)
#plt.figure(figsize=(14,8), dpi= 90)
#ecco.plot_tiles(tmp_plt, cmin=5, cmax=20, \
#                show_colorbar=True, fig_size=8,\
#               layout='latlon',rotate_to_latlon=True,\
#               show_tile_labels=True, \
#               Arctic_cap_tile_location=10)
if len(tiles_load) <=2: 
    lon0=0
else:
    lon0=180.

SSTunits = r'$^{\circ}$C'
Qsunits = r'W m$^{-2}$' 

SST = theta_dataset.sel(i2=1).THETA
Qs = Qs_dataset.TFLUX
#ADVr_TH = ADVr_TH_dataset.sel(i2=[1,2]).ADVr_TH
#ADVx_TH = ADVx_TH_dataset.sel(i2=[1]).ADVx_TH
#ADVy_TH = ADVy_TH_dataset.sel(i2=[1]).ADVy_TH
#
#h = h.isel(i2=1)
#h = 1
#v = v.isel(i2=1)
#deltaz = deltaz.isel(i2=1)


ADVr_TH = ADVr_TH_dataset.ADVr_TH
ADVx_TH = ADVx_TH_dataset.ADVx_TH
ADVy_TH = ADVy_TH_dataset.ADVy_TH

ADVr_TH = ((rho*c_p*deltaz)/v)*(ADVr_TH.diff('i2'))
ADVy_TH = ((rho*c_p*deltaz)/v)*(-ADVy_TH.diff('i3'))
ADVx_TH = ((rho*c_p*deltaz)/v)*(-ADVx_TH.diff('i4'))

ADVr_TH = ADVr_TH.mean('i2')
ADVx_TH = ADVx_TH.mean('i2')
ADVy_TH = ADVy_TH.mean('i2')




#difference ADVr_TH, ADVx_TH, ADVy_TH along spatial dimension before analyzing?




#ADVr_TH = ((rho*c_p*deltaz)/v)*(ADVr_TH)
#ADVy_TH = ((rho*c_p*deltaz)/v)*(ADVy_TH)
#ADVx_TH = ((rho*c_p*deltaz)/v)*(ADVx_TH)

print 'persisting diff...'
ADVr_TH.persist()
ADVy_TH.persist()
ADVx_TH.persist()


#Qs=Qs/h


meanSST = SST.mean('i1')
meanQs = Qs.mean('i1')
meanADVr_TH = ADVr_TH.mean('i1')
meanADVx_TH = ADVx_TH.mean('i1')
meanADVy_TH = ADVy_TH.mean('i1')

print 'persisting means...'
meanSST.persist()
meanQs.persist()
meanADVr_TH.persist()
meanADVx_TH.persist()
meanADVy_TH.persist()


print 'plotting means...'
plt.figure(figsize=(14,8), dpi= 90)
f, ax, p, cbar = ecco.plot_proj_to_latlon_grid(meanSST.lon, meanSST.lat, meanSST, plot_type = 'contourf', \
                              user_lon_0 = lon0, \
                              levels = 41, \
                              dx=1.0,\
                              dy=1.0,\
                              units=SSTunits,\
                              projection_type = 'PlateCaree',\
                              cmap = plt.cm.CMRmap_r, \
                              cmin=0, cmax=30, \
                              show_colorbar=True, \
                              less_output = False);
f.savefig(fout + 'meanSST.png')
                              
                              
                              
plt.figure(figsize=(14,8), dpi= 90)
f, ax, p, cbar = ecco.plot_proj_to_latlon_grid(meanQs.lon, meanQs.lat, meanQs, plot_type = 'contourf', \
                              user_lon_0 = lon0, \
                              levels = 41, \
                              dx=1.0,\
                              dy=1.0,\
                              units=Qsunits,\
                              projection_type = 'PlateCaree',\
                              cmap = plt.cm.RdBu_r, \
                              cmin=-150, cmax=150, \
                              show_colorbar=True, \
                              less_output = False);
f.savefig(fout + 'meanQs.png')

                              
plt.figure(figsize=(14,8), dpi= 90)
f, ax, p, cbar = ecco.plot_proj_to_latlon_grid(meanADVr_TH.lon, meanADVr_TH.lat, meanADVr_TH, plot_type = 'contourf', \
                              user_lon_0 = lon0, \
                              levels = 81, \
                              dx=1.0,\
                              dy=1.0,\
                              units=Qsunits,\
                              projection_type = 'PlateCaree',\
                              cmap = plt.cm.RdBu_r, \
                              cmin=-150, cmax=150, \
                              show_colorbar=True, \
                              less_output = False);
f.savefig(fout + 'meanADVr_TH.png')

                              
plt.figure(figsize=(14,8), dpi= 90)
f, ax, p, cbar = ecco.plot_proj_to_latlon_grid(meanADVx_TH.lon, meanADVx_TH.lat, meanADVx_TH, plot_type = 'contourf', \
                              user_lon_0 = lon0, \
                              levels = 81, \
                              dx=1.0,\
                              dy=1.0,\
                              units=Qsunits,\
                              projection_type = 'PlateCaree',\
                              cmap = plt.cm.RdBu_r, \
                              cmin=-150, cmax=150, \
                              show_colorbar=True, \
                              less_output = False);
f.savefig(fout + 'meanADVx_TH.png')

                              
plt.figure(figsize=(14,8), dpi= 90)
f, ax, p, cbar = ecco.plot_proj_to_latlon_grid(meanADVy_TH.lon, meanADVy_TH.lat, meanADVy_TH, plot_type = 'contourf', \
                              user_lon_0 = lon0, \
                              levels = 81, \
                              dx=1.0,\
                              dy=1.0,\
                              units=Qsunits,\
                              projection_type = 'PlateCaree',\
                              cmap = plt.cm.RdBu_r, \
                              cmin=-150, cmax=150, \
                              show_colorbar=True, \
                              less_output = False);
f.savefig(fout + 'meanADVy_TH.png')

#remove climatological monthly mean SST
monthlymeans = SST.groupby('tim.month').mean('i1')
SST = SST.groupby('tim.month') - monthlymeans

monthlymeans = Qs.groupby('tim.month').mean('i1')
Qs = Qs.groupby('tim.month') - monthlymeans

monthlymeans = ADVr_TH.groupby('tim.month').mean('i1')
ADVr_TH = ADVr_TH.groupby('tim.month') - monthlymeans

monthlymeans = ADVx_TH.groupby('tim.month').mean('i1')
ADVx_TH = ADVx_TH.groupby('tim.month') - monthlymeans

monthlymeans = ADVy_TH.groupby('tim.month').mean('i1')
ADVy_TH = ADVy_TH.groupby('tim.month') - monthlymeans

#detrend
#SST = linear_detrend(SST)

SST_var= SST.std('i1')
Qs_var = Qs.std('i1')
ADVr_TH_var = ADVr_TH.std('i1')
ADVx_TH_var = ADVx_TH.std('i1')
ADVy_TH_var = ADVy_TH.std('i1')

print 'persisting variance...'
SST_var.persist()
Qs_var.persist()
ADVr_TH_var.persist()
ADVx_TH_var.persist()
ADVy_TH_var.persist()
                              
plt.figure(figsize=(14,8), dpi= 90)
f, ax, p, cbar = ecco.plot_proj_to_latlon_grid(SST_var.lon, SST_var.lat, SST_var, plot_type = 'contourf', \
                              user_lon_0 = lon0, \
                              levels = 41, \
                              dx=1.0,\
                              dy=1.0,\
                              units=SSTunits,\
                              projection_type = 'PlateCaree',\
                              cmap = plt.cm.cubehelix_r, \
                              cmin=0, cmax=1.5, \
                              show_colorbar=True, \
                              less_output = False);
f.savefig(fout + 'varSST.png')
 
                              
plt.figure(figsize=(14,8), dpi= 90)
f, ax, p, cbar = ecco.plot_proj_to_latlon_grid(Qs_var.lon, Qs_var.lat, Qs_var, plot_type = 'contourf', \
                              user_lon_0 = lon0, \
                              levels = 41, \
                              dx=1.0,\
                              dy=1.0,\
                              units=Qsunits,\
                              projection_type = 'PlateCaree',\
                              cmap = plt.cm.cubehelix_r, \
                              cmin=0, cmax=60, \
                              show_colorbar=True, \
                              less_output = False);
f.savefig(fout + 'stdQs.png')  

plt.figure(figsize=(14,8), dpi= 90)
f, ax, p, cbar = ecco.plot_proj_to_latlon_grid(ADVr_TH_var.lon, ADVr_TH_var.lat, ADVr_TH_var, plot_type = 'contourf', \
                              user_lon_0 = lon0, \
                              levels = 41, \
                              dx=1.0,\
                              dy=1.0,\
                              units=Qsunits,\
                              projection_type = 'PlateCaree',\
                              cmap = plt.cm.cubehelix_r, \
                              cmin=0, cmax=60, \
                              show_colorbar=True, \
                              less_output = False);
f.savefig(fout + 'stdADVr_TH.png')   

plt.figure(figsize=(14,8), dpi= 90)
f, ax, p, cbar = ecco.plot_proj_to_latlon_grid(ADVx_TH_var.lon, ADVx_TH_var.lat, ADVx_TH_var, plot_type = 'contourf', \
                              user_lon_0 = lon0, \
                              levels = 41, \
                              dx=1.0,\
                              dy=1.0,\
                              units=Qsunits,\
                              projection_type = 'PlateCaree',\
                              cmap = plt.cm.cubehelix_r, \
                              cmin=0, cmax=60, \
                              show_colorbar=True, \
                              less_output = False);
f.savefig(fout + 'stdADVx_TH.png')   

plt.figure(figsize=(14,8), dpi= 90)
f, ax, p, cbar = ecco.plot_proj_to_latlon_grid(ADVy_TH_var.lon, ADVy_TH_var.lat, ADVy_TH_var, plot_type = 'contourf', \
                              user_lon_0 = lon0, \
                              levels = 41, \
                              dx=1.0,\
                              dy=1.0,\
                              units=Qsunits,\
                              projection_type = 'PlateCaree',\
                              cmap = plt.cm.cubehelix_r, \
                              cmin=0, cmax=60, \
                              show_colorbar=True, \
                              less_output = False);
f.savefig(fout + 'stdADVy_TH.png')                              
                              



#grid_dataset = xr.open_dataset(fgrid + fname_grid).load()
#fig=plt.figure(figsize=(8, 6.5))
#td_masked = theta_dataset.THETA.where(theta_dataset.thic > 0, np.nan)
#theta = theta_dataset.THETA.mean(axis=0)
#theta.isel(i2=0).plot(cmap=plt.cm.cubehelix_r, vmin=-2,vmax=40)
#plt.xlabel('test')