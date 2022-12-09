#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# import matplotlib.dates as mdates
import scipy.optimize as opt

#%% load data
filepath = r'C:\Users\maria\Desktop\CubeSats\list_of_detections.csv'
df = pd.read_csv(filepath)
grbalpha = df[df['CubeSat']=='GRBAlpha']
vzlusat = df[df['CubeSat']=='VZLUSAT2']

# grbalpha['t90_color'] = 'lightblue'
# grbalpha['t90_color'][grbalpha['T90']>2] = 'darkblue'
# vzlusat['t90_color'] = 'lightgreen'
# vzlusat['t90_color'][vzlusat['T90']>2] = 'darkgreen'

#%% plot
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(10,10),dpi=200)
cmap = mpl.colormaps['summer']

axs[0,0].scatter(grbalpha['cr_peak_cutoff_890'],grbalpha['flux_1024'],
                 marker='v',c=grbalpha['exp'],cmap=cmap)#,label='GRBAlpha')
axs[0,0].scatter(vzlusat['cr_peak_cutoff_890'],vzlusat['flux_1024'],
                 marker='x',c=vzlusat['exp'],cmap=cmap)#,label='VZLUSAT-2')
axs[0,0].set_xlabel('cr_peak_cutoff_890')
axs[0,0].set_ylabel('flux_1024')
axs[0,0].scatter(100,10,marker='v',c='k',s=0,label='GRBAlpha')
axs[0,0].scatter(100,10,marker='x',c='k',s=0,label='VZLUSAT-2')
lgnd=axs[0,0].legend()#loc='upper left')
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
# axs[0,0].set_xscale('log')

# axs[0,0].set_yscale('log')

axs[0,1].scatter(grbalpha['cr_peak_cutoff_370'],grbalpha['flux_batse_1024'],
                 marker='v',c=grbalpha['exp'],cmap=cmap,label='GRBAlpha')
axs[0,1].scatter(vzlusat['cr_peak_cutoff_370'],vzlusat['flux_batse_1024'],
                 marker='x',c=vzlusat['exp'],cmap=cmap,label='VZLUSAT-2')
axs[0,1].set_xlabel('cr_peak_cutoff_370')
axs[0,1].set_ylabel('flux_batse_1024')
# axs[0,1].set_xscale('log')
# axs[0,1].set_yscale('log')

axs[1,0].scatter(grbalpha['cnt_t90_cutoff_890'],grbalpha['fluence'],
                 marker='v',c=grbalpha['exp'],cmap=cmap,label='GRBAlpha')
axs[1,0].scatter(vzlusat['cnt_t90_cutoff_890'],vzlusat['fluence'],
                 marker='x',c=vzlusat['exp'],cmap=cmap,label='VZLUSAT-2')
axs[1,0].set_xlabel('cnt_t90_cutoff_890')
axs[1,0].set_ylabel('fluence')
axs[1,0].set_xscale('log')
axs[1,0].set_yscale('log')

axs[1,1].scatter(grbalpha['cnt_t90_cutoff_370'],grbalpha['fluence_batse'],
                 marker='v',c=grbalpha['exp'],cmap=cmap,label='GRBAlpha')
plot = axs[1,1].scatter(vzlusat['cnt_t90_cutoff_370'],vzlusat['fluence_batse'],
                 marker='x',c=vzlusat['exp'],cmap=cmap,label='VZLUSAT-2')
axs[1,1].set_xlabel('cnt_t90_cutoff_370')
axs[1,1].set_ylabel('fluence_batse')
axs[1,1].set_xscale('log')
axs[1,1].set_yscale('log')


fig.subplots_adjust(right=0.96)
cbar_ax = fig.add_axes([0.99, 0.12, 0.05, 0.75])
fig.colorbar(plot,cax=cbar_ax,label='exposure time')

# %%
