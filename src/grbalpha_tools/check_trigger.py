#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.optimize as opt

#%% load grbalpha data
filepath = r'C:\Users\maria\Desktop\GRBs\files\r22k05b_284544.json'    
time_format = '%Y-%m-%dZ%H:%M:%S.%f'

datafile = pd.read_json(filepath,lines=True)
exp = datafile[datafile['type']=='meta']['exptime'].reset_index(drop=True)[0]
nbins = datafile[datafile['type']=='spectrum']['bin_mode'].reset_index(drop=True)[0]
data = datafile[datafile['type']=='spectrum'].data.reset_index(drop=True)
mid = datafile[datafile['type']=='spectrum'].mid.reset_index(drop=True)
mid = pd.DataFrame.from_records(mid.to_list())
time = pd.to_datetime(mid.utc,format=time_format)

# load triggers
trig = pd.read_csv(r'C:\Users\maria\Desktop\all_triggers.csv',usecols=['grb_date','mission'])
trig_date = pd.to_datetime(trig.grb_date)
cond_trig_in_data = np.logical_and(trig_date > time[0], trig_date < time[len(time)-1])
trig_mission = trig.mission[cond_trig_in_data].reset_index(drop=True)
trig_date = trig_date[cond_trig_in_data].reset_index(drop=True)

#%% function def
def check_triggers(grb_date,mission,dtvalue):
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

    # fit
    def linear(x,a,b):
        return a*x + b

    def make_fit(c,od,do,left_lim=0):
        xdata = timestamp[left_lim:od+1]+timestamp[do:]
        ydata = np.concatenate((c[left_lim:od+1],c[do:]))
        c = c[left_lim:] # od:do
        
        popt, pcov = opt.curve_fit(linear,np.array(xdata),ydata)
        
        def cps_bgd(x):
            return linear(x,*popt)
        
        index = np.argmax(c)
        # index = int(len(time_list)/2)
        crb = cps_bgd(timestamp[index])
        err = np.sqrt(crb*exp)/exp
        snr = (c[index]-crb)/err
        return xdata, od, do, popt, snr, left_lim

    xdata, od, do, popt, snr, left_lim = make_fit(cps0+cps1,
                                                int(len(time_list)/2-2),int(len(time_list)/2+2)) 

    print(f'trigger: {grb_date}')
    print(f'SNR (70-370 keV) = {snr}')
    
    # plot
    fig, ax = plt.subplots(figsize=(10,4),dpi=200)
    fig.suptitle(f'{mission}: {grb_date}')
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))#byminute=[40,50,0,10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
    plt.axvline(grb_date,c='r',lw=0.75)
    # plt.axvline(time_list[od],c='k',lw=0.5)
    # plt.axvline(time_list[do],c='k',lw=0.5)

    xlim = 0#int(len(time_list)/2-10)

    # plt.plot(time_list[left_lim:od+1]+time_list[do:],f(np.array(xdata),*popt),'b--',lw=0.9,label='fit')
    plt.step(time_list[xlim:],cps0[xlim:],where='mid',lw=1,label='70 - 110 keV')
    plt.step(time_list[xlim:],cps1[xlim:],'--',where='mid',lw=1,label='110 - 370 keV')
    plt.step(time_list[xlim:],cps2[xlim:],'-.',where='mid',lw=1,label='370 - 630 keV')
    plt.step(time_list[xlim:],cps3[xlim:],':',where='mid',lw=1,label='630 - 890 keV')
        
    plt.xlim(min(time_list[xlim:]),max(time_list))
    plt.xlabel('time [MM:SS]')
    plt.ylabel('count rate [counts/s]')
    plt.legend()
    plt.show()

#%% run function
dtvalue = 2.5
tunit = 'min'

for trigger, mission in zip(trig_date,trig_mission):
    check_triggers(trigger,mission,dtvalue)

