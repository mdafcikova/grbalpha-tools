from dataclasses import dataclass
from collections.abc import MutableSequence
import pandas as pd
import numpy as np

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
        self.df = pd.read_json(filepath,lines=True)
        self.mid = self.df[self.df['type']=='spectrum'].mid.reset_index(drop=True)
        self.mid = pd.DataFrame.from_records(self.mid.to_list())
    
        self.exp_time = self.df[self.df['type']=='meta']['exptime'].reset_index(drop=True)[0]
        self.bin_mode = self.df[self.df['type']=='spectrum']['bin_mode'].reset_index(drop=True)[0]
        self.cutoff = self.df[self.df['type']=='meta']['cutoff'].reset_index(drop=True)[0]
        self.time_utc = pd.to_datetime(self.mid.utc,format='%Y-%m-%dZ%H:%M:%S.%f')
        self.time_utc = self.time_utc.set_axis(self.time_utc.round('S'))
        self.longitude = self.mid.lon.set_axis(self.time_utc.round('S'))
        self.latitude = self.mid.lat.set_axis(self.time_utc.round('S'))
        self.altitude = self.mid.alt.set_axis(self.time_utc.round('S'))
        self.data = self.df[self.df['type']=='spectrum'].data.set_axis(self.time_utc.round('S'))

    # def __len__(self):
    #     return len(self.mid)

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
            return [x for x in trig_date], [x for x in trig_mission]

    def is_SGR_in_file(self,path=r'C:\Users\maria\Desktop\CubeSats\SGRJ1935+2154_list.csv'):
        trig = pd.read_csv(path,usecols=['time','mission'])
        trig_date = pd.to_datetime(trig.time)
        cond_trig_in_data = np.logical_and(trig_date > self.time_utc[0], trig_date < self.time_utc[len(self.time_utc)-1])
        trig_mission = trig.mission[cond_trig_in_data].reset_index(drop=True)
        trig_date = trig_date[cond_trig_in_data].reset_index(drop=True)
        if len(trig_date)==0:
            return 'No SGR in file.'
        else:
            return [x for x in trig_date], [x for x in trig_mission]

    def is_SF_in_file(self):
        trig = pd.read_csv(r'C:\Users\maria\Desktop\CubeSats\KW_sf_list.csv',usecols=['DateTime','Class'])
        trig_date = pd.to_datetime(trig.DateTime)
        cond_trig_in_data = np.logical_and(trig_date > self.time_utc[0], trig_date < self.time_utc[len(self.time_utc)-1])
        sf_class = trig.Class[cond_trig_in_data].reset_index(drop=True)
        trig_date = trig_date[cond_trig_in_data].reset_index(drop=True)
        if len(trig_date)==0:
            return 'No SF in file.'
        else:
            return [x for x in trig_date], [x for x in sf_class]

    def check_event(self, event_time, event_type, event_ra=None, event_dec=None):
        '''
        plots +-dt part of the file around the event_time
        plots skymap with event position, sun position, Earth's shadow
        returns: peak time in each band + cutoff-370 + cutoff-890
                 SNR at peak in each band + cutoff-370 + cutoff-890
                 count rate (cnt/s) above bgd at peak in each band + cutoff-370 + cutoff-890
                 T90
                 SNR in T90 in each band + cutoff-370 + cutoff-890
                 counts (cnt) above bgd during T90 in each band + cutoff-370 + cutoff-890
                 was event in FoV? Y/N
                 was Sun in FoV? Y/N
        '''

        return print('hi')# plot + skymap + file with values





