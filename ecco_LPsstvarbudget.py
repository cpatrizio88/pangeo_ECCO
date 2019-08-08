#!/usr/bin/env python
# coding: utf-8

# In[2]:


import xarray as xr
import gcsfs
import intake
import numpy as np
import matplotlib
#import cmocean
import stats
import stats as st
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import ecco_v4_tools as ecco
# Import plotting libraries
import importlib
import llcmapping
importlib.reload(llcmapping)
from llcmapping import LLCMapper
importlib.reload(ecco)
importlib.reload(ecco.tile_plot_proj)
import glob


# In[3]:


fin = '/glade/scratch/patrizio/data/'
fout = '/glade/scratch/patrizio/figs/'


# In[4]:


fcoords = glob.glob(fin + '*coords.nc')[0]
fsnp = glob.glob(fin + '*snp.nc')
fvars = set(glob.glob(fin + '*.nc')) - set(glob.glob(fin + '*coords.nc')) - set(glob.glob(fin + '*snp.nc'))
fvars = list(fvars)
fvars 


# In[5]:


coords = xr.open_dataset(fcoords)
coords


# In[6]:


ds_snp = xr.open_mfdataset(fsnp,concat_dim=None)


# In[7]:


ds = xr.open_mfdataset(fvars,concat_dim=None)


# In[8]:


# Flag for low-pass filtering
lowpass=False

# Filter requirements
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 12*1.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)

# Face numbers to analyze
# 0: Southern Ocean (Atlantic)
# 1: South Atlantic Ocean / Africa 
# 2: East North Atlantic / Europe
# 3: Southern Ocean (Indian)
# 4: Indian Ocean
# 5: Asia
# 6: Arctic
# 7: North Pacific (central)
# 8: West South Pacific
# 9: Southern Ocean (West Pacific)
# 10: North America / West North Atlantic
# 11: East South Pacific / South America
# 12: Southern Ocean(East Pacific)
#facen = [5,7]

#Note: longitude bounds can either be 0 < bounds < 360, or -180 < bounds < 180. 
#The only requirement is that the left longitude bound is less than the right bound 
#(along date line must use 0 < bounds < 360).
#(along prime meridian must use -180 < bounds < 180)

# Complete global 
#facen=[0,1,2,3,4,5,6,7,8,9,10,11,12]
#bnds = [0,359.9,-90,90]

#facen=[]
#bnds = [0,359.9,-90,90]

# Global (excluding polar regions)
#facen=[1,2,4,5,7,8,10,11]
#bnds = [0,359.9,-58,70]

#Southern Ocean (Atlantic)
#facen=[0]
#bnds = [-20,20,-58,-90]

#1: South Atlantic Ocean / Africa
#facen=[1]
#bnds = [-38,30,-58,10]

#2: East North Atlantic 
#facen=[2]
#bnds = [-38,30,10,70]

#3: Southern Ocean (Indian)
#facen=[3]
#bnds = [60,143,-58,-90]

#4: Indian Ocean
#facen=[4]
#bnds = [60,143,-58,10]

#7: North Pacific (central)
#facen=[7]
#bnds = [145,230,10,70]

#8: West South Pacific
#facen=[8]
#bnds = [145,230,-58,10]

#11: East South Pacific
#facen=[11]
#bnds = [-128,-38,-58,10]

#2, 10: North Atlantic
facen=[2,10]
bnds = [-80,0,10,70]

#5,7,10: North Pacific
#facen=[5,7,10]
#bnds = [100,270,10,70]

#4,5,7,8,10,11: Pacific
#facen=[4,5,7,8,10,11]
#bnds = [100,300,-70,70]

#5,7,8,10,11: Tropical Pacific
#facen=[5,7,8,10,11]
#bnds = [145,290,-15,15]

#5,7: KOE
#facen=[5,7]
#bnds = [120,180,15,60]


# In[9]:


rho0 = 1029 #sea-water density (kg/m^3)
c_p = 3994 #sea-water heat capacity (J/kg/K)


# In[10]:


coords.isel(face=facen)


# In[11]:


# Vertical grid spacing
drF = coords.drF
hFacC = coords.hFacC
#rA = coords.rA.isel(face=facen).load()
#vol = drF*hFacC*rA.load()


# In[12]:


c_o = rho0*c_p*drF*hFacC


# In[13]:


T = ds_snp.T.isel(face=facen)
adv_ConvH = ds.adv_ConvH.isel(face=facen)
dif_ConvH = ds.dif_ConvH.isel(face=facen)
forcH = ds.forcH.isel(face=facen)


# In[14]:


dt = coords.time_snp[1:].load()
dt = dt.rename({'time_snp': 'time'})
# delta t in seconds. Note: divide by 10**9 to convert nanoseconds to seconds
dt.values = [float(t)/10**9 for t in np.diff(coords.time_snp)]

# time axis of dt should be the same as of the monthly averages
dt.time.values = coords.time[1:-1].values


# In[15]:


lons = coords.XC
lats = coords.YC


# In[16]:


T_anom, T_clim = st.anom(T) 
C_adv_anom, C_adv_anom = st.anom(adv_ConvH)
C_dif_anom, C_dif_anom = st.anom(dif_ConvH)
C_forc_anom, C_forc_anom = st.anom(forcH)
totalH_anom = C_adv_anom + C_dif_anom + C_forc_anom


# In[17]:


if lowpass:
    
    T_anom = T_anom.chunk({'time':288, 'j':10, 'i':10})
    
    C_adv_anom = C_adv_anom.chunk({'time':288, 'j':10, 'i':10})
    C_dif_anom = C_dif_anom.chunk({'time':288, 'j':10, 'i':10})
    C_forc_anom = C_forc_anom.chunk({'time':288, 'j':10, 'i':10})
    
    T_anom  = stats.butter_lowpass_filter_xr(T_anom, cutoff, fs, order)
    
    C_adv_anom  = stats.butter_lowpass_filter_xr(C_adv_anom, cutoff, fs, order)
    C_dif_anom = stats.butter_lowpass_filter_xr(C_dif_anom, cutoff, fs, order)
    C_forc_anom  = stats.butter_lowpass_filter_xr(C_forc_anom, cutoff, fs, order)
    
    totalH_anom = C_adv_anom + C_dif_anom + C_forc_anom


# In[18]:


tendH_perMonth = (T_anom.shift(time=-1)-T_anom)[:-1]


# In[20]:


# Make sure time axis is the same as for the monthly variables
tendH_perMonth.time.values = coords.time[1:-1].values

# Convert tendency from 1/month to 1/s
tendH_perSec = tendH_perMonth/dt
tendH_perSec = tendH_perSec.transpose('face','time', 'k', 'j', 'i')


# In[21]:


# Define tendH array with correct dimensions
tendH_anom = xr.DataArray(np.nan*np.zeros([len(facen),np.shape(tendH_perSec)[1]+2,50,90,90]),
                     coords={'face': facen, 'time': range(np.shape(tendH_perSec)[1]+2),'k': np.array(range(0,50)),
                             'j': np.array(range(0,90)),'i': np.array(range(0,90))},dims=['face', 'time','k', 'j','i'])

tendH_anom.time.values = coords.time.values


# In[22]:


tendH_anom


# In[23]:


tendH_anom.nbytes/1e9


# In[ ]:


# Add coordinates#
tendH_anom['XC'] = lons
tendH_anom['YC'] = lats
tendH_anom['Z'] = coords.Z

# Total tendency (degC/s)
tendH_anom.values[:,1:-1,:] = tendH_perSec.values
get_ipython().run_line_magic('time', 'tendH_anom.load()')
#%time tendH.persist()

# Convert from degC/s to W/m^2
tendH_anom = c_o*tendH_anom
tendH_anom = tendH_anom.transpose('time','face', 'k', 'j', 'i')


# In[ ]:


face=0
k = 0
j = 15
i = 15

plt.figure(figsize=(14,10))
plt.subplot(2, 1, 1)
plt.plot(tendH_anom.time, tendH_anom.isel(face=face,k=k,j=j,i=i), lw=4, color='K', marker='.',label='total tendency')
plt.plot(C_forc_anom.time, C_forc_anom.isel(face=face,k=k,j=j,i=i), lw=2, color='C0', marker='.',label='forcing')
plt.plot(C_adv_anom.time, C_adv_anom.isel(face=face,k=k,j=j,i=i), lw=2, color='C1', marker='.',label='advection')
plt.axhline(0,color='k',lw=1)
plt.plot(C_dif_anom.time, C_dif_anom.isel(face=face,k=k,j=j,i=i), lw=2, color='C2',label='diffusion')
plt.setp(plt.gca(), 'xticklabels',[])
plt.legend(loc='best',frameon=False,fontsize=14)

plt.subplot(2, 1, 2)
plt.plot(totalH_anom.time, totalH_anom.isel(face=face,k=k,j=j,i=i), lw=4, color='red', marker='.',label='RHS')
plt.plot(tendH_anom.time, tendH_anom.isel(face=face,k=k,j=j,i=i), lw=2, color='blue', marker='.',label='LHS')
plt.plot(tendH_anom.time, (totalH_anom-tendH_anom).isel(face=face,k=k,j=j,i=i), lw=2, color='k', marker='.',label='RHS - LHS')
plt.legend(loc='best',frameon=False,fontsize=14)
plt.savefig(fout + 'sstbudget_anom_ts.png')


# In[ ]:


T_var = T_anom.var(dim='time')
get_ipython().run_line_magic('time', 'T_var.load()')
#%time T_var.persist()


# In[ ]:


tendH_anom = tendH_anom/c_o


# In[ ]:


#tendH_anom = tendH_anom.transpose('time','face', 'k', 'j', 'i')
cov_adv = st.cov(tendH_anom, C_adv_anom)
cov_dif = st.cov(tendH_anom, C_dif_anom)
cov_forc = st.cov(tendH_anom, C_forc_anom)


# In[ ]:


cov_adv.nbytes/1e9


# In[ ]:


get_ipython().run_line_magic('time', 'cov_adv.load()')
get_ipython().run_line_magic('time', 'cov_dif.load()')
get_ipython().run_line_magic('time', 'cov_forc.load()')


# In[ ]:


deltat = dt.mean()
deltat.compute()


# In[ ]:


r_1 = st.cor(T_anom, T_anom,lagx=1).compute()
r_1


# In[ ]:


fac = (deltat**2/(2*c_o*(1-r_1)))
fac.load()


# In[ ]:


T_var_sum = fac*(cov_adv + cov_dif + cov_forc)


# In[ ]:


get_ipython().run_line_magic('time', 'T_var_sum.load()')
#%time T_var_sum.persist()


# In[ ]:


k=0
mapper(T_var.isel(k=k), bnds=bnds, cmap='cubehelix_r', vmin=0,vmax=1.0)
mapper(T_var_sum.isel(k=k), bnds=bnds, cmap='cubehelix_r', vmin=0,vmax=1.0)


# The temperature variance budget is clearly balanced! Let's take a look at the contribution due to each term.

# In[ ]:


T_var_adv = fac*cov_adv
T_var_dif = fac*cov_dif
T_var_forc = fac*cov_forc


# ### Contributions to temperature variance from advection, diffusion and surface forcing

# In[ ]:


k=0
mapper(T_var_sum.isel(k=k), bnds=bnds, cmap='cubehelix_r', vmin=0,vmax=1.0)
plt.title(r'temperature variance (K$^2$)')
plt.savefig(fout + 'Tvar_sum.png')
mapper(T_var_adv.isel(k=k), bnds=bnds, cmap='RdBu_r', vmin=-1.0,vmax=1.0)
plt.title(r'advective contribution (K$^2$)')
plt.savefig(fout + 'Tvar_adv.png')
mapper(T_var_dif.isel(k=k), bnds=bnds, cmap='RdBu_r', vmin=-1.0,vmax=1.0)
plt.title(r'diffusive contribution (K$^2$)')
plt.savefig(fout + 'Tvar_dif.png')
mapper(T_var_forc.isel(k=k), bnds=bnds, cmap='RdBu_r', vmin=-1.0,vmax=1.0)
plt.title(r'surface forcing contribution (K$^2$)')
plt.savefig(fout + 'Tvar_forc.png')


# ### Contributions to ocean mixed layer temperature variance from advection, diffusion and surface forcing

# In[ ]:


mxlpoints = mxlpoints.isel(face=facen)
delz = drF*hFacC
delz=delz.where(mxlpoints)
delz_sum = delz.sum(dim='k')


# In[ ]:


mxlpoints


# In[ ]:


weights = delz/delz_sum


# In[ ]:


T_var_mxl = (weights*T_var).where(mxlpoints).sum(dim='k')


# In[ ]:


T_var_adv_mxl = (weights*T_var_adv).where(mxlpoints).sum(dim='k')
T_var_dif_mxl = (weights*T_var_dif).where(mxlpoints).sum(dim='k')
T_var_forc_mxl = (weights*T_var_forc).where(mxlpoints).sum(dim='k')


# In[ ]:


T_var_sum_mxl = T_var_adv_mxl + T_var_dif_mxl + T_var_forc_mxl


# In[ ]:


#f, axes = plt.subplots(2,2,figsize=(16,12))
#f.tight_layout()
mapper(T_var_sum_mxl, bnds=bnds, cmap='cubehelix_r', vmin=0,vmax=1.0)
plt.title(r'temperature variance (K$^2$)')
plt.savefig(fout + 'Tmxlvar_sum.png')

mapper(T_var_adv_mxl, bnds=bnds, cmap='RdBu_r', vmin=-1.0,vmax=1.0)
plt.title(r'advective contribution (K$^2$)')
plt.savefig(fout + 'Tmxlvar_adv.png')

mapper(T_var_dif_mxl, bnds=bnds, cmap='RdBu_r', vmin=-1.0,vmax=1.0)
plt.title(r'diffusive contribution (K$^2$)')
plt.savefig(fout + 'Tmxlvar_dif.png')

mapper(T_var_forc_mxl, bnds=bnds, cmap='RdBu_r', vmin=-1.0,vmax=1.0)
plt.title(r'surface forcing contribution (K$^2$)')
plt.savefig(fout + 'Tmxlvar_forc.png')


# In[ ]:


#mapper(T_var_sum_mxl, bnds=bnds, cmap='cubehelix_r', vmin=0,vmax=1.0)
#plt.title(r'temperature variance (K$^2$)')
#plt.savefig(fout + 'Tmxlvar_sum.png')
mapper(T_var_adv_mxl + T_var_dif_mxl, bnds=bnds, cmap='RdBu_r', vmin=-1.0,vmax=1.0)
plt.title(r'ocean dynamics (advective + diffusive) contribution (K$^2$)')
plt.savefig(fout + 'Tmxlvar_ocndyn.png')
#mapper(T_var_forc_mxl, bnds=bnds, cmap='RdBu_r', vmin=-1.0,vmax=1.0)
#plt.title(r'surface forcing contribution (K$^2$)')
#plt.savefig(fout + 'Tmxlvar_forc.png')

