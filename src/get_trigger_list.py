"""
Creates a list of GRB triggers in the past x hours.
"""

import pandas as pd
from pyorbital.orbital import Orbital

trigger_list = pd.read_csv('/home/ripa/grbalpha/all_triggers.csv')

timestamp = trigger_list.grb_date.astype('datetime64')
end_time = pd.to_datetime('now') - pd.Timedelta('58 h')

utc = []
mission = []
url = []

i = 0
while (timestamp[i] > pd.Timestamp(end_time)):
    if (trigger_list.mission[i] == 'INTEGRAL ISGRI GCN' & trigger_list.note[i] == 'Weak'):
        continue
    else:
        utc.append(trigger_list.grb_date[i])
        mission.append(trigger_list.mission[i])
        url.append(trigger_list.url[i])
    i += 1

orb = Orbital('GRBAlpha', tle_file='/home/mdafcikova/tle.txt')
lon, lat, alt = orb.get_lonlatalt(pd.Series(utc))

def day_fraction(hh,mm,ss):
    return hh/24 + mm/(24*60) + ss/(24*60*60)

dfrac = [round(day_fraction(int(t[11:13]),int(t[14:16]),int(t[17:19])),5) for t in utc ]

F = []

for frac in dfrac:
    if 2*frac > 1:
        F.append(round(2*frac - 1,3))
    else:
        F.append(round(2*frac,3))

df = pd.DataFrame(columns=['UTC','F','day_fraction','f','longitude','latitude','mission','URL'])

df.UTC = utc
df.F = F
df.day_fraction = dfrac
df.f = (128 * round((df.F*360e3 - 51200/2)/128)).astype(int)
df.longitude = lon.round(2)
df.latitude = lat.round(2)
df.mission = mission
df.URL = url

# df.to_csv('daily_trigger_list.csv',index=False)
print(df)