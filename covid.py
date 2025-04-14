# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:54:47 2025

@author: me
"""

# %%
import os
import json
import section
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.dates import DateFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
import math
import urllib
import requests
import seaborn as sns
import pandas as pd
from pandas import json_normalize
import geopandas as gpd
from geopandas import GeoSeries
from geopandas import GeoDataFrame
import numpy as np
import earthpy as et
import rioxarray
import descartes as ds
import shapely
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import box
import pyproj
import mapclassify
import folium
import fiona

# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Download COVID-19 data
IND_ts = pd.read_csv("https://api.covid19india.org/csv/latest/case_time_series.csv", parse_dates=["Date"], index_col=["Date"])
IND_state_ts = pd.read_csv("https://api.covid19india.org/csv/latest/state_wise_daily.csv")
IND_Dist_ct = pd.read_csv("https://api.covid19india.org/csv/latest/district_wise.csv")

# Writing the raw data in excel files
IND_ts.to_csv("D:\\COVID19\\Data\\IND_ts_20218901.csv", index=True, encoding='utf-8-sig')
IND_state_ts.to_csv("D:\\COVID19\\Data\\state_wise_rms_20218901.csv", index=True, encoding='utf-8-sig')
IND_Dist_ct.to_csv("D:\\COVID19\\Data\\district_wise_rms_20218901.csv", index=True, encoding='utf-8-sig')

# Cleaning the district-wise data
# Modify the India time series data
# Modify State-wise data

# Read the processed data
IND_absolute = pd.read_csv("D:\\COVID19\\Data\\Districtwise_Time_Series\\dist_CARD_dbs_218122.csv", index_col=['Dist_key'])
CARD_percentage = pd.read_csv("D:\\COVID19\\Data\\Districtwise_Time_Series\\dist_CARD_per_218122.csv", index_col=['Dist_key'])
CARD_grth = pd.read_csv("D:\\COVID19\\Data\\Districtwise_Time_Series\\dist_CARD_grth_218122.csv", index_col=['Dist_key'])

# Select few districts in the country
select_dbs = IND_absolute.loc[IND_absolute["District"].isin(['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bengaluru Urban', 'Ahmedabad', 'Indore', 'Pune', 'Surat'])]
select_dbs = select_dbs.set_index('District')

# Creating graph of Deceased cases
D0 = select_dbs['ad0088']
D1 = select_dbs['ad0089'] - select_dbs['ad0080']
D2 = select_dbs['ad0731'] - select_dbs['ad0086']
D3 = select_dbs['ad0083'] - select_dbs['ad0037']
D4 = select_dbs['ad0099'] - select_dbs['ad0083']
D5 = select_dbs['ad0103'] - select_dbs['ad0093']
D6 = select_dbs['ad21012'] - select_dbs['ad0103']

# Creating Columns of Configured Cases
d1 = pd.DataFrame(data=D0.index, columns=['District'])
d2 = pd.DataFrame(data=D0.values, columns=['D0'])
D0 = pd.merge(d1, d2, left_index=True, right_index=True)

d3 = pd.DataFrame(data=D1.index, columns=['District'])
d4 = pd.DataFrame(data=D1.values, columns=['D1'])
D1 = pd.merge(d3, d4, left_index=True, right_index=True)

d5 = pd.DataFrame(data=D2.index, columns=['District'])
d6 = pd.DataFrame(data=D2.values, columns=['D2'])
D2 = pd.merge(d5, d6, left_index=True, right_index=True)

d7 = pd.DataFrame(data=D3.index, columns=['District'])
d8 = pd.DataFrame(data=D3.values, columns=['D3'])
D3 = pd.merge(d7, d8, left_index=True, right_index=True)

d9 = pd.DataFrame(data=D4.index, columns=['District'])
d10 = pd.DataFrame(data=D4.values, columns=['D4'])
D4 = pd.merge(d9, d10, left_index=True, right_index=True)

d11 = pd.DataFrame(data=D5.index, columns=['District'])
d12 = pd.DataFrame(data=D5.values, columns=['D5'])
D5 = pd.merge(d11, d12, left_index=True, right_index=True)

d13 = pd.DataFrame(data=D6.index, columns=['District'])
d14 = pd.DataFrame(data=D6.values, columns=['D6'])
D6 = pd.merge(d13, d14, left_index=True, right_index=True)

# Merging the data frames
sel_dbs = pd.merge(D0, D1, on='District')
sel_dbs = pd.merge(sel_dbs, D2, on='District')
sel_dbs = pd.merge(sel_dbs, D3, on='District')
sel_dbs = pd.merge(sel_dbs, D4, on='District')
sel_dbs = pd.merge(sel_dbs, D5, on='District')
sel_dbs = pd.merge(sel_dbs, D6, on='District')
sel_dbs.sort_values('D0', ascending=True, inplace=True)

# Converting Pandas Dataframe into Numpy Array
abs_arr = sel_dbs.to_numpy()

# Plotting the Graph
def millions_formatter(x, pos):
    return f'{x / 1e6:.0f}M'

data = abs_arr[:, 1:]
ind = sel_dbs['District']
data_shape = np.shape(data)

def get_cumulated_array(data, **kwargs):
    cum = data.cumsum(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d

cumulated_data = get_cumulated_array(data, axis=0)
cumulated_data_neg = get_cumulated_array(data, axis=0)
row_mask = (data < 0)
cumulated_data[row_mask] = cumulated_data_neg[row_mask]
data_stack = cumulated_data

cols = ["blue", "green", "red", "cyan", "magenta", "olive", "indigo"]
labels = ["Pre-unlock", "Unlock 1.0", "Unlock 2.0", "Unlock 3.0", "Unlock 4.0", "Unlock 5.0", "Beyond Unlock 5.0, Jan 22"]

# Plot lockdown-wise the graph of deceased cases in selected districts
fig = plt.figure(figsize=(18, 12))
ax = plt.subplot(111)
for i in np.arange(0, data_shape[0]):
    ax.bar(ind, data[i], left=data_stack[i], color=cols[i], label=labels[i])

plt.xlabel("Deceased Cases (x 10^4)", fontweight="bold", fontsize=40, color='black')
plt.ylabel("District Name", fontweight="bold", fontsize=40, color='black')
plt.title('Covid-19 Cases in Selected Districts', fontweight="bold", fontsize=45)
plt.xlim(-500, 13080)
plt.setp(ax.get_xticklabels(), color='black', size=40, rotation=0, horizontalalignment='center')
plt.setp(ax.get_yticklabels(), color='black', size=40, rotation=0)
plt.grid(axis="x", color='black', alpha=0.6, linewidth=1, linestyle='--')
plt.grid(axis="y", color='black', alpha=0.6, linewidth=1, linestyle='--')
plt.legend(loc='lower right', frameon=False, fontsize=20, ncol=1, framealpha=1, shadow=True, borderpad=1, labelspacing=1)
ax.spines['left'].set_color('k')
ax.spines['right'].set_color('k')
ax.spines['top'].set_color('k')
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.title.set_position([0.5, 1.05])
ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
plt.setp(ax.get_xticklabels(), rotation=0)
plt.savefig('Output.png', dpi=300, orientation='portrait', papertype='letter', format='png', bbox_inches='tight')
plt.show()