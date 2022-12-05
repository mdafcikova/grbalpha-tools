#%% libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.coordinates import ICRS, AltAz, EarthLocation, get_sun
import astropy.units as u
from astropy.time import Time
import numpy as np

#%% function definition
def make_oc_map(lon,lat,alt,utc,ra_grb,dec_grb):

    location = EarthLocation(lon=lon*u.deg, lat=lat*u.deg)#, height=alt*u.km)
    altaz = AltAz(obstime=Time(utc), location=location, alt=90*u.deg, az=180*u.deg)

    ra_sat = altaz.transform_to(ICRS).ra.deg
    dec_sat = altaz.transform_to(ICRS).dec.deg

    dec_nadir = -1*lat #-1*dec_sat
    if (ra_sat < 180):
        ra_nadir = ra_sat + 180
    elif (ra_sat > 180):
        ra_nadir = ra_sat - 180

    ra_sun = get_sun(Time(utc)).ra.deg
    dec_sun = get_sun(Time(utc)).dec.deg

    Erad = np.arcsin(6378/(6378+alt))*180/np.pi

    # plot
    fig, ax = plt.subplots(figsize=(10,5),dpi=200)
    fig.suptitle(f'trigger: {utc}')
    fig.tight_layout()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(15))
    ax.set_xlim(360,0)
    ax.set_ylim(-90,90)
    ax.set_xlabel('Ra')
    ax.set_ylabel('Dec')
    ax.grid()

    # ax.scatter(ra_sat,dec_sat)
    # ax.scatter(ra_nadir,dec_nadir,c='grey')
    ax.scatter(ra_grb,dec_grb,marker='x',c='red')
    ax.scatter(ra_sun,dec_sun,c='yellow')

    ra_vals = np.linspace(ra_nadir - Erad, ra_nadir + Erad, 50)
    dec_vals = np.linspace(dec_nadir - Erad, dec_nadir + Erad, 50)
    vals = np.array(np.meshgrid(ra_vals, dec_vals)).T.reshape(-1,2)
    for ra, dec in vals:
        if ((ra - ra_nadir)**2 + (dec - dec_nadir)**2 <= Erad**2):
            if (ra < 0):
                ra = 360 + ra
            elif (ra > 360):
                ra = ra - 360
            
            if (dec > 90):
                dec = -90 + (dec - 90)
            elif (dec < -90):
                dec = (dec + 90) + 90
            
            ax.scatter(ra,dec,c='b',s=0.5)

#%% variables 
lon = 268.404
lat = -82.1738
alt = 554.946
utc = '2022-12-02 21:41:52'
ra_grb = 346.390
dec_grb = -3.140

make_oc_map(lon,lat,alt,utc,ra_grb,dec_grb)


# %%
