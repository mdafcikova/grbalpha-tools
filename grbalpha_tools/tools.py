from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from astropy.coordinates import ICRS, AltAz, EarthLocation, SkyCoord, get_sun
from astropy.visualization.wcsaxes import SphericalCircle
import astropy.units as u
from astropy.time import Time
from pyorbital.orbital import Orbital
import scipy.optimize as opt
import os 
import warnings
warnings.filterwarnings('ignore')

def plot_skymap(event_time, event_type, event_ra, event_dec,
                lon,lat,alt=550,
                save_path=None
                ):
    '''
    Plots a skymap with marked event, Sun and the Earth's shadow.

    Parameters
    ----------
    event_time: str
        time of the event in UTC
    event_type: str
        type of the event to be used in plot title and filename, e.g. 'GRB'
    event_ra: float
        right ascension of the event in degrees
    event_dec: float
        declination of the event in degrees
    lon: float
        satellite's longitude in degrees
    lat: float
        satellite's latitude in degrees
    alt: float
        satellite's altitude in km
    save_path: str
        path to folder where the skymap will be saved
        if None (default), the skymap will not be saved 
    '''
    event_time = pd.Timestamp(event_time).round('ms')
    location = EarthLocation(lon=lon*u.deg, lat=lat*u.deg, height=alt*u.km)
    altaz = AltAz(obstime=Time(event_time), location=location, alt=90*u.deg, az=180*u.deg)

    ra_sat = altaz.transform_to(ICRS()).ra.deg
    dec_sat = altaz.transform_to(ICRS()).dec.deg

    dec_nadir = -1*lat #-1*dec_sat
    if (ra_sat < 180):
        ra_nadir = ra_sat + 180
    elif (ra_sat > 180):
        ra_nadir = ra_sat - 180

    ra_sun = get_sun(Time(event_time)).ra.deg
    dec_sun = get_sun(Time(event_time)).dec.deg

    # Earth's angular radius from the satellite
    Erad = np.arcsin(6378/(6378+alt))*180/np.pi
    # Earth's shaddow
    Earth_coord = SphericalCircle(center=SkyCoord(ra=ra_nadir*u.deg,dec=dec_nadir*u.deg),
                                radius=u.Quantity(value=Erad,unit=u.deg),
                                resolution=500).get_xy()
    Earth_ra = Earth_coord.T[0]
    Earth_dec = Earth_coord.T[1]


    # skymap
    fig, ax = plt.subplots(figsize=(10,5),dpi=200)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(15))
    ax.set_xlim(360,0)
    ax.set_ylim(-90,90)
    ax.grid(True)

    # ax.scatter(ra_sat,dec_sat)
    ax.scatter(ra_nadir,dec_nadir,c='b',label='Earth')
    ax.scatter(Earth_ra,Earth_dec,c='b',s=3)

    ax.scatter(event_ra,event_dec,marker='x',c='red',label='event')
    ax.scatter(ra_sun,dec_sun,marker='*',c='yellow',label='Sun')
    ax.legend()

    fig.suptitle(f"{event_type}: {event_time.strftime(format='%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    ax.set_xlabel('Ra')
    ax.set_ylabel('Dec')
    fig.tight_layout()
    if (save_path != None):
        filepath = save_path + f"{pd.to_datetime(event_time).strftime(format='%Y%m%d-%H%M%S')}_{event_type}/skymap.png"
        fig.savefig(filepath)
    fig.show()

    return # skymap

def ADC_to_keV(ADC,cutoff,gain=4.31,offset=154.0):
    '''
    Converts ADC value to energy keV.

    Parameters
    ----------
    ADC: int
        ADC value to be converted
    cutoff: int
        cutoff value in ADC
    gain: float
        gain value (4.08 keV/ch pre April 2023; 4.31 keV/ch since April 2023)
    offset: float
        offset value to use (154 keV)
    '''
    keV = gain*ADC - offset
    keV_cutoff = gain*cutoff - offset
    if (keV < keV_cutoff):
        keV = keV_cutoff
    return int(round(keV,-1))


class Event():
    '''
    Event is a class which allows you to check whether an event was in satellite's FoV.

    Parameters
    ----------
    time: str
        time of the event in UTC
    event_type: str
        type of the event, e.g. 'GRB'
        if event_type='SF' or event_type='Sun', RA and Dec of the Sun will be calculated automatically
    ra: float
        right ascension of the event in degrees
    dec: float
        declination of the event in degrees
    '''

    def __init__(self,time:str, event_type:str, ra=None, dec=None):
        self.time = time
        self.event_type = event_type
        self.ra = ra
        self.dec = dec

    def get_sun_coord(self):
        '''
        Returns RA and Dec of the Sun.
        '''
        # sun coordinates
        ra_sun = get_sun(Time(self.time)).ra.deg
        dec_sun = get_sun(Time(self.time)).dec.deg
        return ra_sun, dec_sun

    def get_Earth_coord(self,lon,lat,alt):
        '''
        Returns RA and Dec of the Earth's nadir and the contours of the Earth's shadow from satellite's position. 

        Parameters
        ----------
        lon: float
            satellite's longitude in degrees
        lat: float
            satellite's latitude in degrees
        alt: float
            satellite's altitude in km
        '''
        # satellite's ra, dec
        location = EarthLocation(lon=lon*u.deg, lat=lat*u.deg)#, height=alt*u.km)
        altaz = AltAz(obstime=Time(self.time), location=location, alt=90*u.deg, az=180*u.deg)
        ra_sat = altaz.transform_to(ICRS()).ra.deg
        dec_sat = altaz.transform_to(ICRS()).dec.deg

        # nadir coordinates
        dec_nadir = -1*lat #-1*dec_sat
        if (ra_sat < 180):
            ra_nadir = ra_sat + 180
        elif (ra_sat > 180):
            ra_nadir = ra_sat - 180

        # Earth's angular radius from the satellite
        Erad = np.arcsin(6378/(6378+alt))*180/np.pi

        # Earth's shaddow
        Earth_coord = SphericalCircle(center=SkyCoord(ra=ra_nadir*u.deg,dec=dec_nadir*u.deg),
                                      radius=u.Quantity(value=Erad,unit=u.deg),
                                      resolution=500).get_xy()
        Earth_ra = Earth_coord.T[0]
        Earth_dec = Earth_coord.T[1]

        return ra_nadir, dec_nadir, Earth_ra, Earth_dec

    def in_fov(self,sat='GRBAlpha',tle_file_path=None,map=False):
        '''
        Checks if event is in satellite's FoV.

        Parameters
        ----------
        sat: str
            name of the satellite to calculate the fov
        map: bool
            if map=True, a skymap will be created
            default is False
        '''
        # check if one of Earth_coord points is between nadir and Sun;
        # if True = Sun is in FoV
        # if False = Sun in NOT in FoV

        if tle_file_path != None:
            orb = Orbital(sat,tle_file=tle_file_path)
        else:
            orb = Orbital(sat)
        lon, lat, alt = orb.get_lonlatalt(self.time)

        ra_nadir, dec_nadir, Earth_ra, Earth_dec = self.get_Earth_coord(lon,lat,alt)

        if np.logical_or(self.event_type == 'Sun', self.event_type == 'SF'):
            ra_event, dec_event = self.get_sun_coord()
        
        else:
            ra_event, dec_event = self.ra, self.dec
        
        cond_ra = np.logical_or(np.logical_and(Earth_ra > ra_nadir, Earth_ra < ra_event),
                                np.logical_and(Earth_ra < ra_nadir, Earth_ra > ra_event))
        cond_dec = np.logical_or(np.logical_and(Earth_dec > dec_nadir, Earth_dec < dec_event),
                                np.logical_and(Earth_dec < dec_nadir, Earth_dec > dec_event))
        cond = np.logical_and(cond_ra,cond_dec)
        result = any(cond)

        if (map == True):
            plot_skymap(event_time=self.time,event_type=self.event_type,event_ra=ra_event,event_dec=dec_event,
                        lon=lon,lat=lat,alt=alt,save_path=None)

        return print(f'{self.event_type} at {self.time} in FoV: {result}')


@dataclass(frozen=True)
class Chunk():
    exp_time: int
    bin_mode: int
    cutoff: int
    time_utc: str
    lon: float
    lat: float
    alt: float
    data: list

class Observation():#MutableSequence):
    '''
    Parameters
    ----------
    filepath:str
        path to the file with data
    '''
    # filepath: str
    time_format: str = '%Y-%m-%dZ%H:%M:%S.%f'
    
    def __init__(self,filepath:str):
        if (filepath[-4:] == 'json'):
            df = pd.read_json(filepath,lines=True)
            mid = df[df['type']=='spectrum'].mid.reset_index(drop=True)
            mid = pd.DataFrame.from_records(mid.to_list())
        
            self.exp_time = df[df['type']=='meta']['exptime'].reset_index(drop=True)[0]
            self.bin_mode = df[df['type']=='spectrum']['bin_mode'].reset_index(drop=True)[0]
            self.cutoff = df[df['type']=='meta']['cutoff'].reset_index(drop=True)[0]

            self.time_utc = pd.to_datetime(mid.utc,format='%Y-%m-%dZ%H:%M:%S.%f')
            self.time_utc = self.time_utc.set_axis(self.time_utc.round('ms'))
            self.longitude = mid.lon.set_axis(self.time_utc.round('ms'))
            self.latitude = mid.lat.set_axis(self.time_utc.round('ms'))
            self.altitude = mid.alt.set_axis(self.time_utc.round('ms'))

            self.data = df[df['type']=='spectrum'].data.set_axis(self.time_utc.round('ms'))
            self.data = pd.DataFrame.from_records(self.data,index=self.data.index)
            
        elif (filepath[-3:] == 'txt'):
            df = pd.read_csv(filepath,skiprows=8,sep='\s+\s+')

            self.exp_time = pd.read_csv(filepath,skiprows=8,sep='\s+\s+',usecols=['exposure(s)'])['exposure(s)'][0]
            nbins = pd.read_csv(filepath,skiprows=8,sep='\s+\s+',usecols=['spec_nbins']).spec_nbins[0]
            self.bin_mode = np.log(2**8/nbins)/np.log(2)
            self.cutoff = pd.read_csv(filepath,skiprows=6,nrows=1,header=None,sep='\s+',usecols=[1])[1][0]
            
            self.time_utc = pd.to_datetime(df.exp_end_time)- pd.Timedelta(self.exp_time/2,unit='s')
            self.time_utc = self.time_utc.set_axis(self.time_utc.round('ms'))
            self.longitude = round((df.lon_end+df.lon_start)/2,3).set_axis(self.time_utc.round('ms'))
            self.latitude = round((df.lat_end+df.lat_start)/2,3).set_axis(self.time_utc.round('ms'))
            self.altitude = round((df.alt_end+df.alt_start)/2,3).set_axis(self.time_utc.round('ms'))
            
            keys = [col for col in df.columns if col.startswith('cnt_band')]
            vals = np.arange(len(keys))
            names = {keys[i]:vals[i] for i in range(len(keys))}
            self.data = df[keys].rename(columns=names).set_axis(self.time_utc.round('ms'))#/self.exp_time
        else:
            print('ERROR: wrong filename')

    def get_chunk(self,time:str):
        '''
        Returns a specific chunk of data at specified time.
        '''
        return Chunk(exp_time=self.exp_time,
                     bin_mode=self.bin_mode,
                     cutoff=self.cutoff,
                     time_utc=self.time_utc[time],
                     lon=self.longitude[time],
                     lat=self.latitude[time],
                     alt=self.altitude[time],
                     data=self.data[time]
                     )

    def is_GRB_in_file(self,path):
        '''
        Checks if specified file contains a GRB.
        
        Parameters
        ----------
        path: str
            path to the file to be checked
            only .csv files with columns 'grb_date' and 'mission' are supported
        '''
        trig = pd.read_csv(path,usecols=['grb_date','mission'])
        trig_date = pd.to_datetime(trig.grb_date)
        cond_trig_in_data = np.logical_and(trig_date > self.time_utc[0], trig_date < self.time_utc[len(self.time_utc)-1])
        trig_mission = trig.mission[cond_trig_in_data].reset_index(drop=True)
        trig_date = trig_date[cond_trig_in_data].reset_index(drop=True)
        if len(trig_date)==0:
            return 'No GRB in file.'
        else:
            return [x for x in zip(trig_date,trig_mission)]

    def is_SGR_in_file(self,path):
        '''
        Checks if specified file contains a SGR.
        
        Parameters
        ----------
        path: str
            path to the file to be checked
            only .csv files with columns 'time' and 'mission' are supported
        '''
        trig = pd.read_csv(path,usecols=['time','mission'])
        trig_date = pd.to_datetime(trig.time)
        cond_trig_in_data = np.logical_and(trig_date > self.time_utc[0], trig_date < self.time_utc[len(self.time_utc)-1])
        trig_mission = trig.mission[cond_trig_in_data].reset_index(drop=True)
        trig_date = trig_date[cond_trig_in_data].reset_index(drop=True)
        if len(trig_date)==0:
            return 'No SGR in file.'
        else:
            return [x for x in zip(trig_date,trig_mission)]

    def is_SF_in_file(self,path):
        '''
        Checks if specified file contains a SF.
        
        Parameters
        ----------
        path: str
            path to the file to be checked
            only .csv files with columns 'DateTime' and 'Class' are supported
        '''
        trig = pd.read_csv(path,usecols=['DateTime','Class'])
        trig_date = pd.to_datetime(trig.DateTime)
        cond_trig_in_data = np.logical_and(trig_date > self.time_utc[0], trig_date < self.time_utc[len(self.time_utc)-1])
        sf_class = trig.Class[cond_trig_in_data].reset_index(drop=True)
        trig_date = trig_date[cond_trig_in_data].reset_index(drop=True)
        if len(trig_date)==0:
            return 'No SF in file.'
        else:
            return [x for x in zip(trig_date,sf_class)]
        
    def cut_data_around_trigger(self,event_time,
                                dtvalue_left:float=1, dtvalue_right:float=1, tunit:str='min'):
        '''
        Returns pd.DataFrame containing only data around the specified trigger.

        Parameters
        ----------

        '''
        
        event_time = pd.to_datetime(event_time)

        dt_left = pd.Timedelta(dtvalue_left,tunit)
        dt_right = pd.Timedelta(dtvalue_right,tunit)
        start = event_time - dt_left
        end = event_time + dt_right 
        
        dt_start = self.data.index - start
        # print((dt_start==min(abs(dt_start)))[200:400])

        index_start = pd.Series(dt_start.where(abs(dt_start)==min(abs(dt_start)))).dropna().index[0]
        dt_end = self.data.index - end
        index_end = pd.Series(dt_end.where(abs(dt_end)==min(abs(dt_end)))).dropna().index[0]
        df = self.data[index_start:index_end]

        return df
    
    def select_eband(self, df, eband):
        '''
        
        df: 

        Returns beginning, xdata and ydata of selected eband.

        beginning: timestamp of the beginning of the dataset 
        xdata: seconds from beginning
        ydata: count rate in specified energy band 

        '''

        beginning = df.index[0]

        xdata = self.exp_time * np.arange(0,len(df))

        if (eband == 'all'):
            ydata = 0
            for i in range(len(df.columns)):
                ydata += df[i]
        elif (eband == 'batse'):
            ydata = 0
            for i in range(int(128/(2**self.bin_mode))):
                ydata += df[i]
        elif (eband == '0-64'):
            ydata = 0
            for i in range(int(64/(2**self.bin_mode))):
                ydata += df[i]
        elif (eband == '64-128'):
            ydata_128 = 0
            for i in range(int(128/(2**self.bin_mode))):
                ydata_128 += df[i]
            ydata_64 = 0
            for i in range(int(64/(2**self.bin_mode))):
                ydata_64 += df[i]
            ydata = ydata_128-ydata_64
        else:
            ydata = df[eband]

        return beginning, xdata, np.array(ydata)

    def count_rate_data(self,data):
        '''
        data: (raw) ydata - counts per exposure time 
        '''

        return data/self.exp_time


    def fit_data(self, xdata, ydata, llim, rlim, fit_function='linear'or'polynom'):
        '''
        Returns background fit to the data = ydata from the fit.
        
        Parameters
        ----------

        ydata: count rate data! (not counts in exp time)

        '''

        if (fit_function == 'linear'):
            def function(x,a1,a0):
                return a1*x + a0
        elif (fit_function == 'polynom'):
            def function(x,a2,a1,a0):
                return a2*x*x + a1*x + a0
        else:
            print('ERROR: incorrect fit function')

        bg_xdata = np.concatenate((xdata[:llim],xdata[rlim:]))
        bg_ydata = np.concatenate((ydata[:llim],ydata[rlim:]))

        popt, pcov = opt.curve_fit(function,bg_xdata,bg_ydata)
            
        def bgd_data(x):
            return function(x,*popt)
        
        fit = [bgd_data(x) for x in xdata]

        return fit
    
    def bg_sub_ydata(self,ydata,fit):
        '''
        Returns ydata for background subtracted lightcurves.

        '''
        return (ydata - fit)

    
    def peak_stats(self,xdata,ydata,beginning,llim,rlim,fit_function='linear'or'polynom'):
        '''
        ydata: raw NOT background subtracted, COUNT RATE ydata
        '''

        y_fit = self.fit_data(xdata,ydata,llim,rlim,fit_function)
        y_bg_sub = self.bg_sub_ydata(ydata,y_fit)

        peak_cr = max(y_bg_sub)
        idx = np.argmax(y_bg_sub)
        peak_cr_error = np.sqrt(ydata[idx]*self.exp_time)/self.exp_time
        peak_snr = peak_cr/peak_cr_error

        x = xdata[idx]

        peak_time = beginning + pd.Timedelta(value=x,unit='second')

        return peak_time, peak_cr, peak_cr_error, peak_snr

    def t90_stats(self,xdata,ydata,beginning,llim,rlim,fit_function='linear'or'polynom'):
        '''
        ydata: raw NOT background subtracted, COUNT RATE ydata
        '''

        x_event = xdata[llim:rlim]

        y_fit = self.fit_data(xdata,ydata,llim,rlim,fit_function)
        y_bg_sub = self.bg_sub_ydata(ydata,y_fit)
        y_event = y_bg_sub[llim:rlim]

        cum = np.cumsum(y_event)
        cum_5 = cum[-1] * 0.05
        cum_95 = cum[-1] * 0.95

        # def get_x_from_cum_y(cum,y,x):
            
        #     lim1,lim2 = np.sort(abs(cum-y))[:2]

        #     mins = [np.argwhere(abs(cum-y)==lim1)[0][0],np.argwhere(abs(cum-y)==lim2)[0][0]]
        #     min1 = min(mins)
        #     min2 = max(mins)

        #     def f(x,a1,a0):
        #         return a1*x + a0

        #     popt, pcov = opt.curve_fit(f,[x[min1],x[min2]],[y[min1],y[min2]])

        #     def inv_f(y,a1,a0):
        #         return (y-a0)/a1

        #     new_x = inv_f(y,*popt)

        #     return new_x 
        
        # t90_start = get_x_from_cum_y(cum,cum_5,x_event)
        # t90_end = get_x_from_cum_y(cum,cum_95,x_event)

        minimum = np.sort(abs(cum-cum_5))[0]
        index_min = np.argwhere(abs(cum-cum_5)==minimum)[0][0]
        t90_start = x_event[index_min]

        maximum = np.sort(abs(cum-cum_95))[0]
        index_max = np.argwhere(abs(cum-cum_95)==maximum)[0][0]
        t90_end = x_event[index_max]

        t90 = t90_end - t90_start

        total_counts = sum(y_event[index_min:index_max]*self.exp_time)
        total_counts_error = np.sqrt(sum(ydata[index_min:index_max]*self.exp_time))
        t90_snr = total_counts/total_counts_error

        t90_start_time = beginning + pd.Timedelta(t90_start,unit='second')
        t90_end_time = beginning + pd.Timedelta(t90_end,unit='second')

        return t90_start_time, t90_end_time, t90, total_counts, total_counts_error, t90_snr
    
    def get_statistics(self,xdata,ydata,beginning,llim,rlim,fit_function='linear'or'polynom',
                       ADC_low=None,ADC_high=None,gain=4.31,event_type=None,event_time=None,save_path=None):
        '''

        E_low, E_high: limits of the studied energy band in ADC values
        '''
        
        peak_time, peak_cr, peak_cr_error, peak_snr = self.peak_stats(xdata,ydata,beginning,llim,rlim,fit_function)
        t90_start_time, t90_end_time, t90, total_counts, total_counts_error, t90_snr = self.t90_stats(xdata,ydata,beginning,llim,rlim,fit_function)

        output = (f"trigger: {event_type} at {event_time}:\n"+
                  f"energy band [ADC]: {int(ADC_low)} - {int(ADC_high)}\n"+
                  f"energy band [keV]: {ADC_to_keV(ADC_low,self.cutoff,gain)} - {ADC_to_keV(ADC_high,self.cutoff,gain)}\n"+
                  f"gain used for ADC - keV conversion [keV/ch]: {gain}\n"+
                  f"cutoff [ADC]: {self.cutoff}\n"+
                  f"fit function: {fit_function}\n\n"+
                  f"peak time [UTC]: {peak_time}\n"+
                  f"SNR at peak: {round(peak_snr,3)}\n"+
                  f"count rate [cnt/s] above background at peak: {round(peak_cr,3)} +- {round(peak_cr_error,3)}\n\n"+
                  f"T90 start time [UTC]: {t90_start_time}\n"+
                  f"T90 end time [UTC]: {t90_end_time}\n"+
                  f"T90 duration [s]: {t90}\n"+
                  f"SNR in T90: {round(t90_snr,3)}\n"+
                  f"counts above background in T90: {round(total_counts,3)} +- {round(total_counts_error,3)}\n")

        if (save_path != None):
            dirpath = save_path + f"{pd.to_datetime(event_time).strftime(format='%Y%m%d-%H%M%S')}_{event_type}/"

            if (os.path.exists(dirpath)==False):
                os.makedirs(dirpath)

            filename = f"statistics_{int(ADC_low)}-{int(ADC_high)}ADC.txt"
            with open(dirpath+filename, "w") as text_file:
                text_file.write(output)

        return print(output)        



    def lightcurve(self, event_time, event_type, dtvalue_left=1, dtvalue_right=1, tunit='min', figsize=(9,13), gain=4.31, save_path=None):
        '''
        Returns lightcurve around the specified trigger.

        '''
        df = self.cut_data_around_trigger(event_time,dtvalue_left,dtvalue_right,tunit)

        # number of energy bands
        ncols = int(2**8/2**self.bin_mode)
        # do not plot energy bands below cutoff
        empty_bins = int(self.cutoff/(2**self.bin_mode))
        
        fig, ax = plt.subplots(nrows=ncols-empty_bins+1,figsize=figsize,dpi=200,sharex=True)

        beginning, xdata, ydata = self.select_eband(df,'all')
        ydata = self.count_rate_data(ydata)
        x_event_time = int((pd.Timestamp(event_time) - beginning).to_numpy())*1e-9

        ax[-1].axvline(x_event_time,c='k',ls='--',lw=0.7)
        ax[-1].step(xdata,ydata,c='C0',where='mid',lw=0.75,label=f'{ADC_to_keV(0,cutoff=self.cutoff,gain=gain)} - {ADC_to_keV(256,cutoff=self.cutoff,gain=gain)} keV')
        ax[-1].errorbar(xdata,ydata,yerr=np.sqrt(ydata*self.exp_time)/self.exp_time,c='C0',lw=0.5,fmt=' ')
        ax[-1].legend(loc='lower left')

        i = 0
        for band in reversed(range(empty_bins,ncols)):
            ADC_low = band*256/ncols
            E_low = ADC_to_keV(ADC_low,cutoff=self.cutoff,gain=gain)
            ADC_high = (band+1)*256/ncols
            E_high = ADC_to_keV(ADC_high,cutoff=self.cutoff,gain=gain)

            beginning, xdata, ydata = self.select_eband(df,band)
            ydata = self.count_rate_data(ydata)

            if (E_low != E_high):
                ax[i].axvline(x_event_time,c='k',ls='--',lw=0.7)
                ax[i].step(xdata,ydata,where='mid',lw=0.75,c='C0',label=f'{E_low} - {E_high} keV')
                ax[i].errorbar(xdata,ydata,yerr=np.sqrt(ydata*self.exp_time)/self.exp_time,lw=0.5,c='C0',fmt=' ')
                ax[i].legend(loc='lower left')

                i += 1

        ax[-1].set_xlim(min(xdata),max(xdata))
        ax[-1].set_xlabel(f"seconds from {beginning.round('1s').strftime(format='%Y-%m-%d-%H:%M:%S')} UTC")
        fig.supylabel('count rate [counts/s]')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.show()

        if (save_path != None):
            dirpath = save_path + f"{pd.to_datetime(event_time).strftime(format='%Y%m%d-%H%M%S')}_{event_type}/"

            if (os.path.exists(dirpath)==False):
                os.makedirs(dirpath)

            filepath = dirpath + "lightcurve.png"
            fig.savefig(filepath)

    def bg_sub_lightcurve(self, event_time, event_type, llim, rlim, fit_function='linear'or'polynom', dtvalue_left=1, dtvalue_right=1, tunit='min', figsize=(9,13), gain=4.31, save_path=None):
        '''
        description
        '''

        df = self.cut_data_around_trigger(event_time,dtvalue_left,dtvalue_right,tunit)

        # number of energy bands
        ncols = int(2**8/2**self.bin_mode)
        # do not plot energy bands below cutoff
        empty_bins = int(self.cutoff/(2**self.bin_mode))

        # lightcurve with fit         
        fig, ax = plt.subplots(nrows=ncols-empty_bins+1,figsize=figsize,dpi=200,sharex=True)

        beginning, xdata, ydata = self.select_eband(df,'all')
        x_event_time = int((pd.Timestamp(event_time) - beginning).to_numpy())*1e-9

        ydata = self.count_rate_data(ydata)
        y_fit = self.fit_data(xdata,ydata,llim,rlim,fit_function)

        ax[-1].step(xdata,ydata,c='C0',where='mid',lw=0.75,label=f'{ADC_to_keV(0,cutoff=self.cutoff,gain=gain)} - {ADC_to_keV(256,cutoff=self.cutoff,gain=gain)} keV')
        ax[-1].errorbar(xdata,ydata,yerr=np.sqrt(ydata*self.exp_time)/self.exp_time,c='C0',lw=0.5,fmt=' ')

        ax[-1].axvline(xdata[llim]-self.exp_time/2,c='k',lw=0.6,alpha=0.5)
        ax[-1].axvline(xdata[rlim]-self.exp_time/2,c='k',lw=0.6,alpha=0.5)        
        ax[-1].plot(xdata,y_fit,c='C0',lw=0.6)

        ax[-1].legend(loc='lower left')

        # background subtracted lightcurve
        y_bg_sub = self.bg_sub_ydata(ydata,y_fit)
        fig_sub, ax_sub = plt.subplots(nrows=ncols-empty_bins+1,figsize=figsize,dpi=200,sharex=True)
        ax_sub[-1].axhline(0,c='k',ls='--',lw=0.7)

        ax_sub[-1].axvline(xdata[llim]-self.exp_time/2,c='k',lw=0.6,alpha=0.5)
        ax_sub[-1].axvline(xdata[rlim]-self.exp_time/2,c='k',lw=0.6,alpha=0.5)        
        ax_sub[-1].step(xdata,y_bg_sub,c='C0',where='mid',lw=0.75,label=f'{ADC_to_keV(0,cutoff=self.cutoff,gain=gain)} - {ADC_to_keV(256,cutoff=self.cutoff,gain=gain)} keV')
        ax_sub[-1].legend(loc='lower left')


        i = 0
        for band in reversed(range(empty_bins,ncols)):
            ADC_low = band*256/ncols
            E_low = ADC_to_keV(ADC_low,cutoff=self.cutoff,gain=gain)
            ADC_high = (band+1)*256/ncols
            E_high = ADC_to_keV(ADC_high,cutoff=self.cutoff,gain=gain)

            beginning, xdata, ydata = self.select_eband(df,band)
            ydata = self.count_rate_data(ydata)
            y_fit = self.fit_data(xdata,ydata,llim,rlim,fit_function)
            y_bg_sub = self.bg_sub_ydata(ydata,y_fit)

            if (E_low != E_high):
                ax[i].step(xdata,ydata,where='mid',lw=0.75,c='C0',label=f'{E_low} - {E_high} keV')
                ax[i].errorbar(xdata,ydata,yerr=np.sqrt(ydata*self.exp_time)/self.exp_time,lw=0.5,c='C0',fmt=' ')

                ax[i].axvline(xdata[llim]-self.exp_time/2,c='k',lw=0.6,alpha=0.5)
                ax[i].axvline(xdata[rlim]-self.exp_time/2,c='k',lw=0.6,alpha=0.5)        
                ax[i].plot(xdata,y_fit,c='C0',lw=0.6)

                ax[i].legend(loc='lower left')

                ### bg_sub plot
                ax_sub[i].axhline(0,c='k',ls='--',lw=0.7,alpha=0.5)
                ax_sub[i].axvline(xdata[llim]-self.exp_time/2,c='k',lw=0.6,alpha=0.5)
                ax_sub[i].axvline(xdata[rlim]-self.exp_time/2,c='k',lw=0.6,alpha=0.5)        
                ax_sub[i].step(xdata,y_bg_sub,where='mid',lw=0.75,c='C0',label=f'{E_low} - {E_high} keV')
                ax_sub[i].legend(loc='lower left')

                i += 1

        ax[-1].set_xlim(min(xdata),max(xdata))
        ax[-1].set_xlabel(f"seconds from {beginning.round('1s').strftime(format='%Y-%m-%d-%H:%M:%S')} UTC")
        fig.supylabel('count rate [counts/s]')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.show()

        ### bg_sub plot
        ax_sub[-1].set_xlim(min(xdata),max(xdata))
        ax_sub[-1].set_xlabel(f"seconds from {beginning.round('1s').strftime(format='%Y-%m-%d-%H:%M:%S')} UTC")
        fig_sub.supylabel('count rate [counts/s]')
        fig_sub.tight_layout()
        fig_sub.subplots_adjust(hspace=0)
        fig_sub.show()

        if (save_path != None):
            dirpath = save_path + f"{pd.to_datetime(event_time).strftime(format='%Y%m%d-%H%M%S')}_{event_type}/"

            if (os.path.exists(dirpath)==False):
                os.makedirs(dirpath)

            filepath = dirpath + "lightcurve_fit.png"
            fig.savefig(filepath)

            ### bg_sub plot
            filepath_sub = dirpath + "lightcurve_bg_sub.png"
            fig_sub.savefig(filepath_sub)


    def check_event(self, event_time, event_type:str, 
                    dtvalue_left:float=1, dtvalue_right:float=1, tunit:str='min', 
                    llim:int=5, rlim:int=10, 
                    fit_function:str='linear'or'polynom',
                    gain=4.31,
                    figsize=(9,13),
                    save_path=None):
        '''
        Returns a background subtracted lightcurve around specified trigger and statistics for each energy band.

        Parameters
        ----------
        event_time: str
            time of the event/trigger in UTC
        event_type: str
            type of the event/triggered mission
        dtvalue_left: float
            time to plot before the trigger in minutes (by default, can be changed by changing 'tunit' parameter, see below)
        dtvalue_right: float
            time to plot after the trigger in minutes (by default, can be changed by changing 'tunit' parameter, see below)
        tunit: str
            unit of dtvalue_left and dtvalue_right parameters
        llim: int
            index number of the event start from the beginning of the plot 
        rlim: int
            index number of the event end from the beginning of the plot 
        fit_function: str
            if 'linear', a linear function will be used to fit the background
            if 'polynom', a second order polynomial will be used to fit the background
        figsize: Tuple
            (width,height) of the output figure
        save_path: str
            path to folder where the output folder will be saved
            if save_path=None (default), the output will not be saved 

        '''

        df = self.cut_data_around_trigger(event_time,dtvalue_left,dtvalue_right,tunit)

        ### statistics for the entire energy range
        beginning, xdata, ydata = self.select_eband(df,'all')
        ydata = self.count_rate_data(ydata)

        self.get_statistics(xdata,ydata,beginning,llim,rlim,fit_function,
                            ADC_low=0,ADC_high=256,gain=gain,
                            event_type=event_type,event_time=event_time,save_path=save_path)
        
        ### statistics for the BATSE energy range
        beginning, xdata, ydata = self.select_eband(df,'batse')
        ydata = self.count_rate_data(ydata)

        self.get_statistics(xdata,ydata,beginning,llim,rlim,fit_function,
                            ADC_low=0,ADC_high=128,gain=gain,
                            event_type=event_type,event_time=event_time,save_path=save_path)        

        ### statistics for the 0-64 ADC energy range # for HR plot
        beginning, xdata, ydata = self.select_eband(df,'0-64')
        ydata = self.count_rate_data(ydata)

        self.get_statistics(xdata,ydata,beginning,llim,rlim,fit_function,
                            ADC_low=0,ADC_high=64,gain=gain,
                            event_type=event_type,event_time=event_time,save_path=save_path)        

        ### statistics for the 64-128 ADC energy range # for HR plot
        beginning, xdata, ydata = self.select_eband(df,'64-128')
        ydata = self.count_rate_data(ydata)

        self.get_statistics(xdata,ydata,beginning,llim,rlim,fit_function,
                            ADC_low=64,ADC_high=128,gain=gain,
                            event_type=event_type,event_time=event_time,save_path=save_path)        

        ### statistics for individual energy bands 
        # number of energy bands
        ncols = int(2**8/2**self.bin_mode)
        # do not plot energy bands below cutoff
        empty_bins = int(self.cutoff/(2**self.bin_mode))

        for band in reversed(range(empty_bins,ncols)):
            ADC_low = band*256/ncols
            ADC_high = (band+1)*256/ncols

            beginning, xdata, ydata = self.select_eband(df,band)
            ydata = self.count_rate_data(ydata)

            self.get_statistics(xdata,ydata,beginning,llim,rlim,fit_function,
                                ADC_low=ADC_low,ADC_high=ADC_high,gain=gain,
                                event_type=event_type,event_time=event_time,save_path=save_path)
        
        ### create lightcurves
        self.bg_sub_lightcurve(event_time, event_type, llim, rlim, fit_function, dtvalue_left, dtvalue_right, tunit, figsize, gain, save_path)


    def skymap(self, event_time, event_type, event_ra, event_dec,
                    save_path=None):
        '''
        Returns a skymap with marked event, Sun and the Earth's shadow.

        Parameters
        ----------
        event_time: str
            time of the event in UTC
        event_type: str
            type of the event to be used in plot title and filename, e.g. 'GRB'
        event_ra: float
            right ascension of the event in degrees
        event_dec: float
            declination of the event in degrees
        save_path: str
            path to folder where the skymap will be saved
            if None (default), the skymap will not be saved 
        '''
        time_index = self.longitude.index[self.longitude.index.get_loc(event_time,method='nearest')]
        lon = self.longitude[time_index]
        lat = self.latitude[time_index]
        alt = self.altitude[time_index]
        
        return plot_skymap(event_time=event_time,event_type=event_type,event_ra=event_ra,event_dec=event_dec,lon=lon,lat=lat,alt=alt,save_path=save_path)


