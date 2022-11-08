#%% import libraries
'''
checks difference between local and node 6 timestamps
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% load file 

df = pd.read_csv(r'C:\Users\maria\Desktop\GRBAlpha_operations\cmp-clock-6.txt',sep=': ',header=None)
df = df.pivot(columns=0)[1]

df['Node  6, got clock'] = df['Node  6, got clock'].shift(periods=-1)
df = df.dropna().reset_index(drop=True)
df['Local time clock'] = pd.to_datetime(df['Local time clock'])
df['Node  6, got clock'] = pd.to_datetime(df['Node  6, got clock'])
df = df.rename(columns={'Local time clock': 'local', 'Node  6, got clock': 'node6'})

df['diff'] = df.local - df.node6

# %% plot

fig, ax = plt.subplots(figsize=(6,4),dpi=200)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))#byminute=[40,50,0,10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

ax.scatter(df['local'],pd.to_numeric(df['diff'])/1e9)

plt.xlabel('local time')
plt.ylabel('local time - node 6 time [s]')
plt.show()

