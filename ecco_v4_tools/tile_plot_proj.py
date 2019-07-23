"""
ECCO v4 Python: tile_plot_proj

This module includes routines for plotting arrays in different
projections.

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py

"""

from __future__ import division,print_function
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .resample_to_latlon import resample_to_latlon
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#import matplotlib.ticker as mticker
#from mpl_toolkits.axes_grid1 import make_axes_locatable

#add to change plotting parameters
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams.update({'axes.titlesize': 22})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 18})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
matplotlib.rcParams.update({'axes.labelsize': 20})
matplotlib.rcParams.update({'ytick.labelsize': 16})
matplotlib.rcParams.update({'xtick.labelsize': 16})

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
def plot_proj_to_latlon_grid(lons, lats, data, 
                             projection_type = 'robin', 
                             plot_type = 'pcolormesh', 
                             user_lon_0 = 0,
                             lat_lim = 50, 
                             levels = 21, 
                             cmap='jet', 
                             dx=.25, 
                             dy=.25,
                             units='',
                             show_colorbar = False, 
                             show_grid_lines = False,
                             show_grid_labels = False,
		 	     grid_linewidth = 1, 
	   	 	     grid_linestyle = '--', 
                             subplot_grid=None,
                             less_output=True,
                             **kwargs):
    """Generate a plot of llc data, resampled to lat/lon grid, on specified 
    projection.

    Parameters
    ----------
    lons, lats, data : xarray DataArray    : 
        give the longitude, latitude values of the grid, and the 2D field to 
        be plotted
    projection_type : string, optional
        denote the type of projection, options include
            'robin' - Robinson
            'PlateCaree' - flat 2D projection
            'Mercator'
            'cyl' - Lambert Cylindrical
            'ortho' - Orthographic
            'stereo' - polar stereographic projection, see lat_lim for choosing
            'InterruptedGoodeHomolosine'
                North or South
    user_lon_0 : float, optional, default 0 degrees
        denote central longitude
    lat_lim : int, optional
        for stereographic projection, denote the Southern (Northern) bounds for 
        North (South) polar projection
    levels : int, optional
        number of contours to plot
    cmap : string or colormap object, optional
        denotes to colormap
    dx, dy : float, optional
        latitude, longitude spacing for grid resampling
    units : string, optional
        units of data
    show_colorbar : logical, optional, default False
	show a colorbar or not,
    show_grid_lines : logical, optional
        True only possible for Mercator or PlateCarree projections
    grid_linewidth : float, optional, default 1.0
	width of grid lines
    grid_linestyle : string, optional, default = '--'
	pattern of grid lines,
    cmin, cmax : float, optional
        minimum and maximum values for colorbar, default is min/max of data
    subplot_grid : dict or list, optional
        specifying placement on subplot as
            dict:
                {'nrows': rows_val, 'ncols': cols_val, 'index': index_val}

            list:
                [nrows_val, ncols_val, index_val]

            equates to

                matplotlib.pyplot.subplot(
                    row=nrows_val, col=ncols_val,index=index_val)
    less_output : string, optional
        debugging flag, don't print if True
    """

    #%%    
    cmin = np.nanmin(data)
    cmax = np.nanmax(data)

    for key in kwargs:
        if key == "cmin":
            cmin = kwargs[key]
        elif key == "cmax":
            cmax =  kwargs[key]
        else:
            print("unrecognized argument ", key)

    #%%
    # To avoid plotting problems around the date line, lon=180E, -180W 
    # plot the data field in two parts, A and B.  
    # part 'A' spans from starting longitude to 180E 
    # part 'B' spans the from 180E to 360E + starting longitude.  
    # If the starting  longitudes or 0 or 180 it is a special case.
    if user_lon_0 > -180 and user_lon_0 < 180:
        A_left_limit = user_lon_0
        A_right_limit = 180
        B_left_limit =  -180
        B_right_limit = user_lon_0
        #center_lon = A_left_limit + 180
       
        if not less_output:
            print ('-180 < user_lon_0 < 180')
    
    #correction here to get desired behavior.
    elif user_lon_0 == 180 or user_lon_0 == -180:
        A_left_limit = -180
        A_right_limit = 180
        B_left_limit =  180
        B_right_limit = -180
        #center_lon = 0
	
        if not less_output:
            print('user_lon_0 ==-180 or 180')
   
    else:
        raise ValueError('invalid starting longitude')

    #%%
    # the number of degrees spanned in part A and part B
    num_deg_A =  (int((A_right_limit - A_left_limit)/dx))
    num_deg_B =  (int((B_right_limit - B_left_limit)/dx))

    # find the longitudal limits of part A and B
    lon_tmp_d = dict()
    if num_deg_A > 0:
        lon_tmp_d['A'] = [A_left_limit, A_right_limit]
            
    if num_deg_B > 0:
        lon_tmp_d['B'] = [B_left_limit, B_right_limit]

    # Make projection axis
    (ax,show_grid_labels) = _create_projection_axis(
            projection_type, user_lon_0, lat_lim, subplot_grid, less_output)
    

    #%%
    # loop through different parts of the map to plot (if they exist), 
    # do interpolation and plot
    f = plt.gcf()
    if not less_output:
        print('len(lon_tmp_d): ',len(lon_tmp_d))
    for key, lon_tmp in lon_tmp_d.items():

        new_grid_lon, new_grid_lat, data_latlon_projection = \
            resample_to_latlon(lons, lats, data, 
                               -90+dy, 90-dy, dy,
                               lon_tmp[0], lon_tmp[1], dx, 
                               mapping_method='nearest_neighbor')
            
        if isinstance(ax.projection, ccrs.NorthPolarStereo) or \
           isinstance(ax.projection, ccrs.SouthPolarStereo) :
            p, gl, cbar = \
                plot_pstereo(new_grid_lon,
                             new_grid_lat, 
                             data_latlon_projection,
                             4326, lat_lim, 
                             cmin, cmax, ax,
                             plot_type = plot_type,
                             show_colorbar=False, 
                             circle_boundary=True,
                             cmap=cmap, 
                             show_grid_lines=False,
                             levels=levels,
                             less_output=less_output)

        else: # not polar stereo
            p, gl, cbar = \
                plot_global(new_grid_lon,
                            new_grid_lat, 
                            data_latlon_projection,
                            4326, 
                            cmin, cmax, ax,
                            plot_type = plot_type,                                       
                            show_colorbar = False,
                            cmap=cmap, 
			    show_grid_lines = False,
                            levels=levels,
                            show_grid_labels = False)
			    
                    
        if show_grid_lines :
            gl=ax.gridlines(crs=ccrs.PlateCarree(), 
                                  linewidth=grid_linewidth,
				  color='black', 	
                                  alpha=0.3, 
				  linestyle=grid_linestyle, 
                                  draw_labels = show_grid_labels)

            #format latitude and longitude labels
            #gl.ylabels_right=False 
            #gl.xlabels_top=False
#            gl.xformatter = LONGITUDE_FORMATTER
#            gl.yformatter = LATITUDE_FORMATTER
#            ax.get_yaxis().set_tick_params(direction='out')
#            ax.get_xaxis().set_tick_params(direction='out')
            #ax.get_xticklabels()
#            xlabels=ax.get_xticklabels()
#            ylabels = ax.get_yticklabels()
#            print ('xlabels', xlabels[:])
#            print ('ylabels', xlabels[:])
        
         #%%
        ax.add_feature(cfeature.LAND, edgecolor='k', facecolor='grey')
        #ax.coastlines()
        ax.add_feature(cfeature.COASTLINE,linewidth=0.1)
    
    #Add this for better lat/lon labeling
    ax= plt.gca()
                    
    pardiff = 30.
    merdiff = 60.
    par = np.arange(-90.,91.,pardiff)
    mer = np.arange(-180.,180.,merdiff)
    
    ax.set_xticks(mer, crs=ax.projection)
    ax.set_yticks(par, crs=ax.projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    
    #Add this for fixed plotting bounds
    bnds = [lons.min().values, lons.max().values, lats.min().values, lats.max().values]
    
    ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=ax.projection)
    
    #Adjust colorbar depending on longitude limits
    if show_colorbar:
        if np.abs(A_left_limit - A_right_limit) >= 180:
            orient='horizontal'
            #orient='bottom'
        else:
            orient='vertical'
            #orient='right'
        sm = plt.cm.ScalarMappable(cmap=discrete_cmap(levels, cmap), norm=plt.Normalize(cmin,cmax))
        sm._A = []
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes(orient, size="5%", pad=0.05)
        cbar = plt.colorbar(sm,ax=ax,orientation=orient,fraction=0.078, pad=0.09, label=units)
        #cbar = plt.colorbar(sm,cax=cax)
        #cbar = plt.colorbar(ax=ax)   
    
   

    #%%
    return f, ax, p, cbar
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    

def plot_pstereo(xx,yy, data, 
                 data_projection_code, \
                 lat_lim, 
                 cmin, cmax, ax, 
                 plot_type = 'pcolormesh', 
                 show_colorbar=False, 
                 circle_boundary = False, 
		 grid_linewidth = 1, 
		 grid_linestyle = '--', 
                 cmap='jet', 
                 show_grid_lines=False,
                 levels = 20,
                 less_output=True):

                            
    if isinstance(ax.projection, ccrs.NorthPolarStereo):
        ax.set_extent([-180, 180, lat_lim, 90], ccrs.PlateCarree())
        if not less_output:
            print('North Polar Projection')
    elif isinstance(ax.projection, ccrs.SouthPolarStereo):
        ax.set_extent([-180, 180, -90, lat_lim], ccrs.PlateCarree())
        if not less_output:
            print('South Polar Projection')
    else:
        raise ValueError('ax must be either ccrs.NorthPolarStereo or ccrs.SouthPolarStereo')

    if not less_output:
        print('lat_lim: ',lat_lim)
    
    if circle_boundary:
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    if show_grid_lines :
        gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                          linewidth=grid_linewidth, color='black', 
                          alpha=0.5, linestyle=grid_linestyle)
        gl.ylabels_right=False
        gl.xlabels_top=False
    else:
        gl = []

    if data_projection_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        # reproject the data if necessary
        data_crs=ccrs.epsg(data_projection_code)
    

    p=[]    
    if plot_type == 'pcolormesh':
        p = ax.pcolormesh(xx, yy, data, transform=data_crs, \
                          vmin=cmin, vmax=cmax, cmap=cmap)

    elif plot_type =='contourf':
        p = ax.contourf(xx, yy, data, levels, transform=data_crs,  \
                 vmin=cmin, vmax=cmax, cmap=cmap)

    else:
        raise ValueError('plot_type  must be either "pcolormesh" or "contourf"')

         
    ax.add_feature(cfeature.LAND)
    ax.coastlines('110m', linewidth=0.5)

    cbar = []
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=discrete_cmap(levels, cmap), norm=plt.Normalize(cmin,cmax))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax)
    
    return p, gl, cbar

#%%    

def plot_global(xx,yy, data, 
                data_projection_code,
                cmin, cmax, ax, 
                plot_type = 'pcolormesh', 
                show_colorbar=False, 
                cmap='jet', 
                show_grid_lines = True,
                show_grid_labels = True,
		grid_linewidth = 1, 
                levels=20):

    if show_grid_lines :
        gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                          linewidth=1, color='black', 
                          draw_labels = show_grid_labels,
                          alpha=0.5, linestyle='--')
        gl.ylabels_right=False
        gl.xlabels_top=False
    else:
        gl = []
        
    if data_projection_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        data_crs =ccrs.epsg(data_projection_code)
        
    if plot_type == 'pcolormesh':
        p = ax.pcolormesh(xx, yy, data, transform=data_crs, 
                          vmin=cmin, vmax=cmax, cmap=cmap)
    elif plot_type =='contourf':
        p = ax.contourf(xx, yy, data, levels, transform=data_crs,
                        vmin=cmin, vmax=cmax, cmap=cmap, extend='both')
    else:
        raise ValueError('plot_type  must be either "pcolormesh" or "contourf"') 
                         
    ax.coastlines('110m', linewidth=0.5)
    ax.add_feature(cfeature.LAND)

    cbar = []
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=discrete_cmap(levels, cmap), norm=plt.Normalize(cmin,cmax))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax) 
    return p, gl, cbar

# -----------------------------------------------------------------------------

def _create_projection_axis(projection_type, 
                            user_lon_0, 
                            lat_lim, 
                            subplot_grid, 
                            less_output):

    """Set appropriate axis for projection type
    See plot_proj_to_latlon_grid for input parameter definitions.

    Returns
    -------
    ax :  matplotlib axis object
        defined with the correct projection
    show_grid_labels : logical
        True = show the grid labels, only currently
        supported for PlateCarree and Mercator projections
    """

    # initialize (optional) subplot variables
    row = []
    col = []
    ind = []

    if subplot_grid is not None:

        if type(subplot_grid) is dict:
            row = subplot_grid['nrows']
            col = subplot_grid['ncols']
            ind = subplot_grid['index']

        elif type(subplot_grid) is list:
            row = subplot_grid[0]
            col = subplot_grid[1]
            ind = subplot_grid[2]

        else:
            raise TypeError('Unexpected subplot_grid type: ',type(subplot_grid))


    if projection_type == 'Mercator':
        if subplot_grid is not None:
            ax = plt.subplot(row, col, ind,
                    projection=ccrs.Mercator(central_longitude=user_lon_0))
        else:
            ax = plt.axes(projection=ccrs.Mercator(central_longitude=user_lon_0))
        show_grid_labels = True

    elif projection_type == 'PlateCaree':
        if subplot_grid is not None   :
            ax = plt.subplot(row, col, ind,
                    projection=ccrs.PlateCarree(central_longitude=    user_lon_0))
        else:
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=user_lon_0))
        show_grid_labels = True

    elif projection_type == 'cyl':
        if subplot_grid is not None:
            ax = plt.subplot(row, col, ind,
                    projection=ccrs.LambertCylindrical(central_longitude=user_lon_0))
        else:
            ax = plt.axes(projection=ccrs.LambertCylindrical(central_longitude=user_lon_0))
        show_grid_labels = False

    elif projection_type == 'robin':    
        if subplot_grid is not None:
            ax = plt.subplot(row, col, ind,
                    projection=ccrs.Robinson(central_longitude=user_lon_0))
        else:
            ax = plt.axes(projection=ccrs.Robinson(central_longitude=user_lon_0))
        show_grid_labels = False

    elif projection_type == 'ortho':
        if subplot_grid is not None:
            ax = plt.subplot(row, col, ind,
                    projection=ccrs.Orthographic(central_longitude=user_lon_0))
        else:
            ax = plt.axes(projection=ccrs.Orthographic(central_longitude=user_lon_0))
        show_grid_labels = False

    elif projection_type == 'stereo':    
        if lat_lim > 0:
            stereo_proj = ccrs.NorthPolarStereo()
        else:
            stereo_proj = ccrs.SouthPolarStereo()

        if subplot_grid is not None:
            ax = plt.subplot(row, col, ind,
                    projection=stereo_proj)
        else:
            ax = plt.axes(projection=stereo_proj)

        show_grid_labels = False

    elif projection_type == 'InterruptedGoodeHomolosine':
        if subplot_grid is not None:
            ax = plt.subplot(row, col, ind,
                    projection=ccrs.InterruptedGoodeHomolosine(central_longitude=user_lon_0))
        else:
            ax = plt.axes(projection=ccrs.InterruptedGoodeHomolosine(central_longitude=user_lon_0))
        show_grid_labels = False
        
    else:
        raise NotImplementedError('projection type must be either "Mercator", "PlateCaree",  "cyl", "robin", "ortho", "stereo", or "InterruptedGoodeHomolosine"')

    if not less_output:
        print('Projection type: ', projection_type)

    return (ax,show_grid_labels)
