#%% libraries
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord 
import numpy as np
from pyorbital.astronomy import observer_position, sun_ra_dec

#%% variables
lon = 119.526
lat = 6.97588
alt = 530.411
utc = '2022-11-22 02:43:26'

ra_sat = SkyCoord(lon,lat,unit='deg').ra.deg
dec_sat = SkyCoord(lon,lat,unit='deg').dec.deg

dec_nadir = -1*dec_sat
if (ra_sat < 180):
    ra_nadir = ra_sat + 180
elif (ra_sat > 180):
    ra_nadir = ra_sat - 180

ra_grb = 121.5800
dec_grb = -67.6299

ra_sun, dec_sun = sun_ra_dec(utc)
if (ra_sun < 0):
    ra_sun = 360 + ra_sun

Erad = np.arctan(6378/alt)

#%% plot
fig, ax = plt.subplots(figsize=(10,5),dpi=200)

ax.scatter(ra_sat,dec_sat)
ax.scatter(ra_nadir,dec_nadir,c='grey')
ax.scatter(ra_grb,dec_grb,marker='x',c='red')
ax.scatter(ra_sun,dec_sun,c='yellow')

ax.set_xlim(0,360)
ax.set_ylim(-90,90)
ax.set_xlabel('Ra')
ax.set_ylabel('Dec')

# %%
