from dataclasses import dataclass
from collections.abc import MutableSequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from astropy.coordinates import ICRS, AltAz, EarthLocation, get_sun
import astropy.units as u
from astropy.time import Time
import scipy.optimize as opt
import os 

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
    # filepath: str
    time_format: str = '%Y-%m-%dZ%H:%M:%S.%f'
    
    def __init__(self,filepath:str):
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

    def get_chunk(self,time:str):
        return Chunk(exp_time=self.exp_time,
                     bin_mode=self.bin_mode,
                     cutoff=self.cutoff,
                     time_utc=self.time_utc[time],
                     lon=self.longitude[time],
                     lat=self.latitude[time],
                     alt=self.altitude[time],
                     data=self.data[time]
                     )

    def is_GRB_in_file(self):
        trig = pd.read_csv(r'C:\Users\maria\Desktop\CubeSats\all_triggers.csv',usecols=['grb_date','mission'])
        trig_date = pd.to_datetime(trig.grb_date)
        cond_trig_in_data = np.logical_and(trig_date > self.time_utc[0], trig_date < self.time_utc[len(self.time_utc)-1])
        trig_mission = trig.mission[cond_trig_in_data].reset_index(drop=True)
        trig_date = trig_date[cond_trig_in_data].reset_index(drop=True)
        if len(trig_date)==0:
            return 'No GRB in file.'
        else:
            return [x for x in zip(trig_date,trig_mission)]

    def is_SGR_in_file(self,path=r'C:\Users\maria\Desktop\CubeSats\SGRJ1935+2154_list.csv'):
        trig = pd.read_csv(path,usecols=['time','mission'])
        trig_date = pd.to_datetime(trig.time)
        cond_trig_in_data = np.logical_and(trig_date > self.time_utc[0], trig_date < self.time_utc[len(self.time_utc)-1])
        trig_mission = trig.mission[cond_trig_in_data].reset_index(drop=True)
        trig_date = trig_date[cond_trig_in_data].reset_index(drop=True)
        if len(trig_date)==0:
            return 'No SGR in file.'
        else:
            return [x for x in zip(trig_date,trig_mission)]

    def is_SF_in_file(self):
        trig = pd.read_csv(r'C:\Users\maria\Desktop\CubeSats\KW_sf_list.csv',usecols=['DateTime','Class'])
        trig_date = pd.to_datetime(trig.DateTime)
        cond_trig_in_data = np.logical_and(trig_date > self.time_utc[0], trig_date < self.time_utc[len(self.time_utc)-1])
        sf_class = trig.Class[cond_trig_in_data].reset_index(drop=True)
        trig_date = trig_date[cond_trig_in_data].reset_index(drop=True)
        if len(trig_date)==0:
            return 'No SF in file.'
        else:
            return [x for x in zip(trig_date,sf_class)]
    
    def ADC_to_keV(self,ADC):
        keV = 4.08*ADC - 154
        keV_cutoff = 4.08*self.cutoff - 154
        if (keV < keV_cutoff):
            keV = keV_cutoff
        return int(round(keV,-1))
    
    def check_event(self, event_time, event_type:str, 
                       dtvalue:float=1, tunit:str='min', 
                       llim:int=5, rlim:int=10, vlines:bool=False, 
                       fit_function:str='linear'or'polynom' or function):
        '''
        plots +-dtvalue part of the file around the event_time
        returns: peak time in (each band +) cutoff-370 + cutoff-890
                 SNR at peak in (each band +) cutoff-370 + cutoff-890
                 count rate (cnt/s) above bgd at peak in (each band +) cutoff-370 + cutoff-890
                 T90
                 SNR in T90 in (each band +) cutoff-370 + cutoff-890
                 counts (cnt) above bgd during T90 in (each band +) cutoff-370 + cutoff-890
        add: each band stuff
             background-sub plot
             choice of band for fit
             add statistical errors for cr at peak/counts in t90
        '''

        event_time = pd.to_datetime(event_time)
        dt = pd.Timedelta(dtvalue,tunit)
        start = event_time - dt
        end = event_time + dt 
        ncols = int(2**8/2**self.bin_mode)
        nrows = int(2*dtvalue*60/self.exp_time)
        cps = np.zeros((ncols,nrows))

        time_list = []
        timestamp = []

        j = 0
        for i in self.data.index:
            t = self.time_utc[i]
            if np.logical_and(t > start,t < end):
                time_list.append(t)
                timestamp.append(j)
                for n in range(ncols):
                    cps[n][j] = self.data[n][i]
                j += 1

        index_from = int(len(time_list)/2-llim)
        index_to = int(len(time_list)/2+rlim)

        if (fit_function == 'linear'):
            def function(x,a1,a0):
                return a1*x + a0
        elif (fit_function == 'polynom'):
            def function(x,a2,a1,a0):
                return a2*x*x + a1*x + a0

        def make_fit(ADC_lower_limit,ADC_upper_limit,f):
            E_low = self.ADC_to_keV(ADC_lower_limit)
            E_high = self.ADC_to_keV(ADC_upper_limit)
            
            E_n_bins = int(2**8/2**self.bin_mode)
            first_band = ADC_lower_limit*E_n_bins/256
            last_band = ADC_upper_limit*E_n_bins/256

            c = np.zeros(nrows)
            for band in range(int(first_band),int(last_band)):
                c += cps[band]
            
            # data for background fit
            xdata = timestamp[:index_from]+timestamp[index_to:]
            ydata = np.concatenate((c[:index_from],c[index_to:]))

            popt, pcov = opt.curve_fit(f,np.array(xdata),ydata)
            
            def cps_bgd(x):
                return f(x,*popt)

            # stat at peak
            index_peak = np.where(c == np.max(c[index_from:index_to]))[0][0]
            peak_c = np.max(c[index_from:index_to])

            peak_time = time_list[index_peak]
            crb = cps_bgd(timestamp[index_peak]) # count rate background
            err = np.sqrt(peak_c*self.exp_time)/self.exp_time
            peak_raw_cr = peak_c-crb
            snr_peak = peak_raw_cr/err
            
            # T90 calculation
            t_event = timestamp[index_from:index_to]
            utc_event = time_list[index_from:index_to]
            c_event = c[index_from:index_to]
            c_raw_event = c_event - cps_bgd(c[index_from:index_to])
            c_cum = np.cumsum(c_raw_event)
            c_cum_5 = c_cum[-1] * 0.05
            c_cum_95 = c_cum[-1] * 0.95
            
            def find_nearest(a, a0):
                # find element in nd array 'a' closest to the scalar value 'a0'
                idx = np.abs(a - a0).argmin()
                return a.flat[idx]

            index_t90_start = np.where(c_cum == find_nearest(c_cum,c_cum_5))[0][0]
            index_t90_end = np.where(c_cum == find_nearest(c_cum,c_cum_95))[0][0]

            t90 = pd.to_datetime(utc_event[index_t90_end]) - pd.to_datetime(utc_event[index_t90_start])
            c_event_t90 = sum(c_event[index_t90_start:index_t90_end+1])
            cntb_t90 = 0
            for t in t_event[index_t90_start:index_t90_end+1]:
                cntb_t90 += cps_bgd(t)

            err_t90 = np.sqrt(c_event_t90)
            c_raw_event_t90 = c_event_t90-cntb_t90
            snr_t90 = c_raw_event_t90/err_t90

            # print results:
            output = (f"statistics in {E_low}-{E_high} keV for a {event_type} at {event_time}:\n"+
                      f"peak time [utc]: {peak_time}\n"+
                      f"SNR at peak: {snr_peak}\n"+
                      f"count rate [cnt/s] above background at peak: {peak_raw_cr}\n"+
                      f"T90 [s]: {t90.round('S').seconds}\n"+
                      f"SNR in T90: {snr_t90}\n"+
                      f"counts above background in T90: {c_raw_event_t90}\n")
            
            dirpath = f"C:\\Users\\maria\\Desktop\\CubeSats\\GRBs\\analysis\\{event_time.strftime(format='%Y%m%d-%H%M%S')}_{event_type}\\"
            os.makedirs(dirpath, exist_ok=True)
            filename = f"statistics_{E_low}-{E_high}keV.txt"
            with open(dirpath+filename, "w") as text_file:
                text_file.write(output)

            print(output)
            return xdata, popt, E_low, E_high
        
        make_fit(ADC_lower_limit=0,ADC_upper_limit=128,f=function)
        xdata, popt, E_low, E_high = make_fit(ADC_lower_limit=0,ADC_upper_limit=256,f=function)
        
        ### timeplot
        fig, ax = plt.subplots(figsize=(10,5),dpi=200)
        fig.suptitle(f'{event_type}: {event_time}')
        ax.xaxis.set_major_locator(mdates.SecondLocator(bysecond=[0,15,30,45]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        ax.axvline(event_time,c='r',lw=0.5)
        ax.axvline(time_list[index_from],c='k',lw=0.5,alpha=0.5)
        ax.axvline(time_list[index_to],c='k',lw=0.5,alpha=0.5)
        
        ax.plot(time_list,function(np.array(timestamp),*popt),c='k',lw=0.5)
        ax.step(time_list,cps.sum(axis=0),c='k',where='mid',lw=0.7,label=f'{E_low} - {E_high} keV')
        ax.errorbar(time_list,cps.sum(axis=0),yerr=np.sqrt(cps.sum(axis=0)),c='k',lw=0.5,fmt=' ')
    
        for band in range(ncols):
            E_low = self.ADC_to_keV(band*256/ncols)
            E_high = self.ADC_to_keV((band+1)*256/ncols)
            if (E_low != E_high):
                ax.step(time_list,cps[band],where='mid',lw=0.75,c='C'+str(band),label=f'{E_low} - {E_high} keV')
                ax.errorbar(time_list,cps[band],yerr=np.sqrt(cps[band]),lw=0.5,c='C'+str(band),fmt=' ')
            
        ax.set_xlim(min(time_list),max(time_list))
        ax.set_xlabel('time [MM:SS]')
        ax.set_ylabel('count rate [counts/s]')
        ax.legend(loc='lower left')
        fig.tight_layout()
        filepath = f"C:\\Users\\maria\\Desktop\\CubeSats\\GRBs\\analysis\\{event_time.strftime(format='%Y%m%d-%H%M%S')}_{event_type}\\timeplot.png"
        fig.savefig(filepath)
        fig.show()

        return # file with values

    def plot_skymap(self, event_time, event_type, event_ra, event_dec):
        '''
        plots skymap with event position, sun position, Earth's shadow         
        was event in FoV? Y/N
        was Sun in FoV? Y/N
        '''
        event_time = pd.Timestamp(event_time).round('ms')
        time_index = self.longitude.index[self.longitude.index.get_loc(event_time,method='nearest')]
        lon = self.longitude[time_index]
        lat = self.latitude[time_index]
        alt = self.altitude[time_index]
        location = EarthLocation(lon=lon*u.deg, lat=lat*u.deg, height=alt*u.km)
        altaz = AltAz(obstime=Time(event_time), location=location, alt=90*u.deg, az=180*u.deg)

        ra_sat = altaz.transform_to(ICRS).ra.deg
        dec_sat = altaz.transform_to(ICRS).dec.deg

        dec_nadir = -1*lat #-1*dec_sat
        if (ra_sat < 180):
            ra_nadir = ra_sat + 180
        elif (ra_sat > 180):
            ra_nadir = ra_sat - 180

        ra_sun = get_sun(Time(event_time)).ra.deg
        dec_sun = get_sun(Time(event_time)).dec.deg

        Erad = np.arcsin(6378/(6378+alt))*180/np.pi

        # skymap
        fig, ax = plt.subplots(figsize=(10,5),dpi=200)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(15))
        ax.set_xlim(360,0)
        ax.set_ylim(-90,90)
        ax.grid(True)

        # ax.scatter(ra_sat,dec_sat)
        # ax.scatter(ra_nadir,dec_nadir,c='grey')
        ax.scatter(event_ra,event_dec,marker='x',c='red')
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

        fig.suptitle(f'{event_type}: {event_time}')
        ax.set_xlabel('Ra')
        ax.set_ylabel('Dec')
        fig.tight_layout()
        filepath = f"C:\\Users\\maria\\Desktop\\CubeSats\\GRBs\\analysis\\{pd.to_datetime(event_time).strftime(format='%Y%m%d-%H%M%S')}_{event_type}\\skymap.png"
        fig.savefig(filepath)
        fig.show()

        return # skymap

