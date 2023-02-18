#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# import matplotlib.dates as mdates
import scipy.optimize as opt

#%% load data
filepath = r'C:\Users\maria\Desktop\CubeSats\list_of_detections_new.csv'
df = pd.read_csv(filepath)
df = df[df.T90_370.notna()&df.fluence.notna()]
grbalpha = df[df['CubeSat']=='GRBAlpha'].reset_index(drop=True)
vzlusat = df[df['CubeSat']=='VZLUSAT-2'].reset_index(drop=True)

# grbalpha['t90_color'] = 'lightblue'
# grbalpha['t90_color'][grbalpha['T90']>2] = 'darkblue'
# vzlusat['t90_color'] = 'lightgreen'
# vzlusat['t90_color'][vzlusat['T90']>2] = 'darkgreen'

#%% plot
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(10,10),dpi=200)
# cmap = mpl.colormaps['summer']

# axs[0,0].scatter(grbalpha['cr_peak_cutoff_890'],grbalpha['flux_1024'],label='GRBAlpha')
axs[0,0].errorbar(grbalpha['cr_peak_cutoff_890'],grbalpha['flux_1024'],
                  xerr=grbalpha['cr_peak_cutoff_890_error'],yerr=grbalpha['flux_1024_error'],
                  fmt='o',capsize=3,label='GRBAlpha')
axs[0,0].errorbar(vzlusat['cr_peak_cutoff_890'],vzlusat['flux_1024'],
                  xerr=vzlusat['cr_peak_cutoff_890_error'],yerr=vzlusat['flux_1024_error'],
                  fmt='o',capsize=3,label='VZLUSAT-2')
axs[0,0].set_xlabel('peak count rate for E = cutoff-890 keV [cnt/s]')
axs[0,0].set_ylabel('Fermi/GBM peak flux for E = 10-1000 keV [ph/cm2/s]')
# axs[0,0].scatter(100,10,marker='v',c='k',s=0,label='GRBAlpha')
# axs[0,0].scatter(100,10,marker='x',c='k',s=0,label='VZLUSAT-2')
axs[0,0].legend()#loc='upper left')
axs[0,0].set_xscale('log')
axs[0,0].set_yscale('log')


# axs[0,1].scatter(grbalpha['cr_peak_cutoff_370'],grbalpha['flux_batse_1024'],label='GRBAlpha')
axs[0,1].errorbar(grbalpha['cr_peak_cutoff_370'],grbalpha['flux_batse_1024'],
                  xerr=grbalpha['cr_peak_cutoff_370_error'],yerr=grbalpha['flux_batse_1024_error'],
                  fmt='o',capsize=3)
axs[0,1].errorbar(vzlusat['cr_peak_cutoff_370'],vzlusat['flux_batse_1024'],
                  xerr=vzlusat['cr_peak_cutoff_370_error'],yerr=vzlusat['flux_batse_1024_error'],
                  fmt='o',capsize=3)
axs[0,1].set_xlabel('peak count rate for E = cutoff-370 keV [cnt/s]')
axs[0,1].set_ylabel('Fermi/GBM peak flux for E = 50-300 keV [ph/cm2/s]')
axs[0,1].set_xscale('log')
axs[0,1].set_yscale('log')

# axs[1,0].scatter(grbalpha['cnt_t90_cutoff_890'],grbalpha['fluence'],label='GRBAlpha')
axs[1,0].errorbar(grbalpha['cnt_t90_cutoff_890'],grbalpha['fluence'],
                  xerr=grbalpha['cnt_t90_cutoff_890_error'],yerr=grbalpha['fluence_error'],
                  fmt='o',capsize=3)
axs[1,0].errorbar(vzlusat['cnt_t90_cutoff_890'],vzlusat['fluence'],
                  xerr=vzlusat['cnt_t90_cutoff_890_error'],yerr=vzlusat['fluence_error'],
                  fmt='o',capsize=3)
axs[1,0].set_xlabel('total counts in T90 for E = cutoff-890 keV [cnt]')
axs[1,0].set_ylabel('Fermi/GBM fluence for E = 10-1000 keV [erg/cm2]')
axs[1,0].set_xscale('log')
axs[1,0].set_yscale('log')

# axs[1,1].scatter(grbalpha['cnt_t90_cutoff_370'],grbalpha['fluence_batse'],label='GRBAlpha')
axs[1,1].errorbar(grbalpha['cnt_t90_cutoff_370'],grbalpha['fluence_batse'],
                  xerr=grbalpha['cnt_t90_cutoff_370_error'],yerr=grbalpha['fluence_batse_error'],
                  fmt='o',capsize=3)
axs[1,1].errorbar(vzlusat['cnt_t90_cutoff_370'],vzlusat['fluence_batse'],
                  xerr=vzlusat['cnt_t90_cutoff_370_error'],yerr=vzlusat['fluence_batse_error'],
                  fmt='o',capsize=3)
axs[1,1].set_xlabel('total counts in T90 for E = cutoff-370 keV [cnt]')
axs[1,1].set_ylabel('Fermi/GBM fluence for E = 50-300 keV [erg/cm2]')
axs[1,1].set_xscale('log')
axs[1,1].set_yscale('log')

# %%
print(grbalpha.sort_values(by='cnt_t90_cutoff_890').snr_t90_890)
print(grbalpha.sort_values(by='cnt_t90_cutoff_370').snr_t90_370)
print(grbalpha.sort_values(by='cr_peak_cutoff_890').snr_peak_890)
print(grbalpha.sort_values(by='cr_peak_cutoff_370').snr_peak_370)
