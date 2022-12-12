#%% import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks

#%% load file
filepath = r'C:\Users\maria\Desktop\CubeSats\GRBs\files\out_all.json'
datafile = pd.read_json(filepath,lines=True)

#%% variables definition
time_format = '%Y-%m-%dZ%H:%M:%S.%f'
exp = datafile[datafile['type']=='meta']['exptime'].reset_index(drop=True)[0]
nbins = datafile[datafile['type']=='spectrum']['bin_mode'].reset_index(drop=True)[0]
data = datafile[datafile['type']=='spectrum'].data.reset_index(drop=True)
data = pd.DataFrame.from_records(data)
mid = datafile[datafile['type']=='spectrum'].mid.reset_index(drop=True)
mid = pd.DataFrame.from_records(mid.to_list())
time = pd.to_datetime(mid.utc,format=time_format)

#%% find peaks
Eband = 3
peaks = find_peaks(data[Eband]) # 48-16 ADC

def ADC_to_keV(ADC):
    keV = 4.08 * ADC - 154
    return keV

Emin = ADC_to_keV(2**8/2**nbins*Eband)
Emax = ADC_to_keV(2**8/2**nbins*(Eband+1))

#%% plot
fig, ax = plt.subplots(figsize=(10,3),dpi=200)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
# plt.axvline(time[peaks[0][0]],c='r',lw=0.75)

plt.step(time,data[Eband],where='mid',lw=1,label=f'{round(Emin)} - {round(Emax)} keV')

plt.xlim(min(time),max(time))
plt.xlabel('date')
plt.ylabel('count rate [counts/s]')
plt.legend()
plt.show()



# %%