#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.optimize as opt

#%% load grbalpha data
filepath = r'C:\Users\maria\Desktop\CubeSats\GRBs\files\r22k27a_404736.json'
time_format = '%Y-%m-%dZ%H:%M:%S.%f'

datafile = pd.read_json(filepath,lines=True)
exp = datafile[datafile['type']=='meta']['exptime'].reset_index(drop=True)[0]
nbins = datafile[datafile['type']=='spectrum']['bin_mode'].reset_index(drop=True)[0]
data = datafile[datafile['type']=='spectrum'].data.reset_index(drop=True)
mid = datafile[datafile['type']=='spectrum'].mid.reset_index(drop=True)
mid = pd.DataFrame.from_records(mid.to_list())
time = pd.to_datetime(mid.utc,format=time_format)

# load triggers
trig = pd.read_csv(r'C:\Users\maria\Desktop\CubeSats\all_triggers.csv',usecols=['grb_date','mission'])
trig_date = pd.to_datetime(trig.grb_date)
cond_trig_in_data = np.logical_and(trig_date > time[0], trig_date < time[len(time)-1])
trig_mission = trig.mission[cond_trig_in_data].reset_index(drop=True)
trig_date = trig_date[cond_trig_in_data].reset_index(drop=True)

#%% function def
def check_triggers(grb_date,mission,dtvalue,use_cps,llim=3,rlim=10,vlines=False,fit=False):
    dt = pd.Timedelta(dtvalue,tunit)
    start = grb_date - dt
    end = grb_date + dt 
    ncols = int(2**8/2**nbins)
    nrows = int(2*dtvalue*60/exp)

    time_list = []
    timestamp = []
    cps = np.zeros((ncols,nrows))

    j = 0
    for i in range(len(data)):
        t = time[i]
        if np.logical_and(t > start,t < end):
            time_list.append(t)
            timestamp.append(j)
            cps[0][j] = data[i][0]
            cps[1][j] = data[i][1]
            cps[2][j] = data[i][2]
            cps[3][j] = data[i][3]
            j += 1

    cps0 = cps[0]/exp
    cps1 = cps[1]/exp
    cps2 = cps[2]/exp
    cps3 = cps[3]/exp

    if (use_cps == '0'):
        use_c = cps0
    elif (use_cps == '1'):
        use_c = cps1
    elif (use_cps == '2'):
        use_c = cps2
    elif (use_cps == '3'):
        use_c = cps3
    elif (use_cps == '01'):
          use_c = cps0 + cps1
    elif (use_cps == '012'):
          use_c = cps0 + cps1 + cps2
    elif (use_cps == '0123'):
          use_c = cps0 + cps1 + cps2 + cps3
    
    # fit
    def linear(x,a,b):
        return a*x + b

    def make_fit(c,od,do):
        xdata = timestamp[:od]+timestamp[do:]
        ydata = np.concatenate((c[:od],c[do:]))

        popt, pcov = opt.curve_fit(linear,np.array(xdata),ydata)
        
        def cps_bgd(x):
            return linear(x,*popt)

        #index = np.argmax(c)
        index = np.where(c == np.max(c[od:do]))[0][0]
        # c = c[od+1:do] # od:do
        max_c = np.max(c[od:do])

        crb = cps_bgd(timestamp[index])
        err = np.sqrt(max_c*exp)/exp
        snr_peak = (max_c-crb)/err

        # T90 calculation        
        t_grb = timestamp[od:do]
        raw_c_grb = c[od:do]
        c_grb = c[od:do] - cps_bgd(c[od:do])
        c_cum = np.cumsum(c_grb)
        c_cum_5 = c_cum[-1] * 0.05
        c_cum_95 = c_cum[-1] * 0.95
        
        def find_nearest(a, a0):
            # Element in nd array 'a' closest to the scalar value 'a0'
            idx = np.abs(a - a0).argmin()
            return a.flat[idx]

        index_T90_start = np.where(c_cum == find_nearest(c_cum,c_cum_5))[0][0]
        index_T90_end = np.where(c_cum == find_nearest(c_cum,c_cum_95))[0][0]

        T90 = t_grb[index_T90_end] - t_grb[index_T90_start]
        c_grb_T90 = sum(raw_c_grb[index_T90_start:index_T90_end])        
        crb_T90 = 0
        for t in t_grb[index_T90_start:index_T90_end]:
            crb_T90 += cps_bgd(t)

        err_T90 = np.sqrt(c_grb_T90)
        snr_T90 = (c_grb_T90-crb_T90)/err_T90

        return xdata, od, do, popt, snr_peak, T90, snr_T90

    xdata, od, do, popt, snr, T90, snr_T90 = make_fit(use_c,
                                        int(len(time_list)/2-llim),
                                        int(len(time_list)/2+rlim)) 

    print(f'trigger: {grb_date}')
    print(f'SNR (70-370 keV) = {snr}')
    print(f'T90 = {T90} s')
    print(f'SNR in T90 = {snr_T90}')

    # plot
    fig, ax = plt.subplots(figsize=(10,4),dpi=200)
    fig.suptitle(f'{mission}: {grb_date}')
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=20))#byminute=[40,50,0,10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
    plt.axvline(grb_date,c='r',lw=0.5)
    if (vlines == True):
        plt.axvline(time_list[od],c='k',alpha=0.25)
        plt.axvline(time_list[do],c='k',alpha=0.25)
    
    if (fit == True):
        plt.plot(time_list[:od]+time_list[do:],linear(np.array(xdata),*popt),'--',c='C4',lw=0.5)
        plt.step(time_list,use_c,c='C4',where='mid',lw=1,label='70 - 370 keV')
        plt.errorbar(time_list,use_c,yerr=np.sqrt(cps0),c='C4',lw=0.5,fmt=' ')
    
    plt.step(time_list,cps0,c='C0',where='mid',lw=1,label='70 - 110 keV')
    plt.errorbar(time_list,cps0,yerr=np.sqrt(cps0),c='C0',lw=0.5,fmt=' ')
    plt.step(time_list,cps1,'--',c='C1',where='mid',lw=1,label='110 - 370 keV')
    plt.errorbar(time_list,cps1,yerr=np.sqrt(cps1),c='C1',lw=0.5,fmt=' ')
    plt.step(time_list,cps2,'-.',c='C2',where='mid',lw=1,label='370 - 630 keV')
    plt.errorbar(time_list,cps2,yerr=np.sqrt(cps2),c='C2',lw=0.5,fmt=' ')
    plt.step(time_list,cps3,':',c='C3',where='mid',lw=1,label='630 - 890 keV')
    plt.errorbar(time_list,cps3,yerr=np.sqrt(cps3),c='C3',lw=0.5,fmt=' ')
        
    plt.xlim(min(time_list[:]),max(time_list))
    plt.xlabel('time [MM:SS]')
    plt.ylabel('count rate [counts/s]')
    plt.legend()#loc='lower left')
    plt.show()

#%% run function
dtvalue = 2
tunit = 'min'

for trigger, mission in zip(trig_date,trig_mission):
    check_triggers(trigger,mission,dtvalue,use_cps='01',
                   llim=5,rlim=4,vlines=True,fit=True)

#%% run function manually
# trigger_list = ['2022-10-17 11:14:43.0']
# mission_list = ['SGR']
# for trigger, mission in zip(trigger_list,mission_list):
#     check_triggers(pd.to_datetime(trigger),mission,dtvalue)

# %%
