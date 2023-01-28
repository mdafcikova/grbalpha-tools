#%% libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.coordinates import ICRS, AltAz, EarthLocation, get_sun
import astropy.units as u
from astropy.time import Time
import numpy as np

#%% function definition
def make_oc_map(lon,lat,alt,utc,ra_grb,dec_grb,projection=False):

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
    if (projection == False):
        fig, ax = plt.subplots(figsize=(10,5),dpi=200)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(15))
        ax.set_xlim(360,0)
        ax.set_ylim(-90,90)

        # ax.scatter(ra_sat,dec_sat)
        ax.scatter(ra_nadir,dec_nadir,c='grey')
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

    elif (projection == True):
        fig = plt.figure(figsize=(10,5),dpi=200)
        ax = plt.axes(projection="mollweide")
        ax.scatter(ra_sat*np.pi/180,dec_sat*np.pi/180)
        ax.scatter(ra_nadir*np.pi/180,dec_nadir*np.pi/180,c='grey')
        ax.scatter(ra_grb*np.pi/180,dec_grb*np.pi/180,marker='x',c='red')
        ax.scatter(ra_sun*np.pi/180,dec_sun*np.pi/180,c='yellow')

        ra_vals = np.linspace(ra_nadir - Erad, ra_nadir + Erad, 50)*np.pi/180
        dec_vals = np.linspace(dec_nadir - Erad, dec_nadir + Erad, 50)*np.pi/180
        vals = np.array(np.meshgrid(ra_vals, dec_vals)).T.reshape(-1,2)
        for ra, dec in vals:
            if ((ra - ra_nadir*np.pi/180)**2 + (dec - dec_nadir*np.pi/180)**2 <= (Erad*np.pi/180)**2):
                if (ra < 0):
                    ra = 2*np.pi + ra
                elif (ra > 2*np.pi):
                    ra = ra - 2*np.pi
                
                if (dec > np.pi/2):
                    dec = -1*np.pi/2 + (dec - np.pi/2)
                elif (dec < -1*np.pi/2):
                    dec = (dec + np.pi/2) + np.pi/2
                
                ax.scatter(ra,dec,c='b',s=0.5)

    fig.suptitle(f'trigger: {utc}')
    fig.tight_layout()
    ax.set_xlabel('Ra')
    ax.set_ylabel('Dec')
    ax.grid(True)
    fig.show()

#%% variables 
lon = 157.545
lat = -15.1291
alt =  556.762
utc = '2023-01-02 00:22:12'
ra_grb = 272.8
dec_grb = -35.1

make_oc_map(lon,lat,alt,utc,ra_grb,dec_grb,projection=False)


# %%
