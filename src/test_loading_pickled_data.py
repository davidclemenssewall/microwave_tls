#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_loading_pickled_data.py

Created on Wed Nov 17 12:10:01 2021

@author: thayer
"""

import copy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
import os
import numpy as np
import re

# load data
df = pd.read_pickle(os.path.join('..', 'data', 'polar_gridded_data',
                                       'polar_height_maps.pkl'))

r_edges = np.load(os.path.join('..', 'data', 'polar_gridded_data',
                               'r_edges.npy'))

t_edges = np.load(os.path.join('..', 'data', 'polar_gridded_data',
                               't_edges.npy'))

# %% Function for converting filename to date

def mosaic_date_parser(project_name):
    """
    Parses the project name into a date.

    Parameters
    ----------
    project_name : str
        The project name as a string.

    Returns
    -------
    The date and time (for April 8 b)

    """
    
    # The date is always a sequence of 6 numbers in the filename.
    seq = re.compile('[0-9]{6}')
    seq_match = seq.search(project_name)
    
    if seq_match:
        year = '20' + seq_match[0][-2:]
        if year == '2020':
            date = year + '-' + seq_match[0][2:4] + '-' + seq_match[0][:2]
        elif year == '2019':
            date = year + '-' + seq_match[0][:2] + '-' + seq_match[0][2:4]
    else:
        return None
    
    # Looks like Dec 6 was an exception to the format...
    if date=='2019-06-12':
        date = '2019-12-06'
    
    # Handle April 8 b scan case.
    if project_name[seq_match.end()]=='b':
        date = date + ' 12:00:00'
        
    return date

# %% Display figures

cmap_div = copy.copy(cm.get_cmap('RdBu_r'))
cmap_div.set_bad(color='black')
cmap_seq = copy.copy(cm.get_cmap('rainbow'))
cmap_seq.set_bad(color='black')

warnings.warn('Hackish way to cycle through dataframe, check if modifying')
for i in np.arange(8)*2:
    f, axs = plt.subplots(1, 3, figsize=(15,5), 
                          subplot_kw=dict(projection='polar'))
    
    apr17 = df.at[i,'pol_mean']
    apr22 = df.at[i+1, 'pol_mean']
    hmin = min(np.nanmin(apr17), np.nanmin(apr22))
    hmax = max(np.nanmax(apr17), np.nanmax(apr22))
    
    h = axs[0].pcolormesh(t_edges, r_edges*180/np.pi, 
              apr17,
              cmap=cmap_seq, vmin=hmin, vmax=hmax)

    axs[0].set_ylim([0, df.at[i, 'beam_width']/2])
    axs[0].set_title(mosaic_date_parser(df.at[i,'project_name']))
    f.colorbar(h, ax=axs[0], shrink=0.8, label='Height (m)',
               format='%.3f')
    axs[0].yaxis.set_ticks(r_edges[r_edges<=df.at[i, 'beam_width']*np.pi/180/2]*180/np.pi)
    axs[0].tick_params(axis='y', colors='lime')
    
    
    h = axs[1].pcolormesh(t_edges, r_edges*180/np.pi, 
              apr22,
              cmap=cmap_seq, vmin=hmin, vmax=hmax)

    axs[1].set_ylim([0, df.at[i+1, 'beam_width']/2])
    axs[1].set_title(mosaic_date_parser(df.at[i+1,'project_name']))
    f.colorbar(h, ax=axs[1], shrink=0.8, label='Height (m)',
               format='%.3f')
    axs[1].yaxis.set_ticks(r_edges[r_edges<=df.at[i, 'beam_width']*np.pi/180/2]*180/np.pi)
    axs[1].tick_params(axis='y', colors='lime')
    
    diff_max = np.nanmax(np.abs(apr22-apr17))
    h = axs[2].pcolormesh(t_edges, r_edges*180/np.pi, 
              apr22-apr17,
              cmap=cmap_div, vmin=-diff_max, vmax=diff_max)

    axs[2].set_ylim([0, df.at[i+1, 'beam_width']/2])
    axs[2].set_title('Difference')
    f.colorbar(h, ax=axs[2], shrink=0.8, label='Accumulation (m)',
               format='%.3f')
    axs[2].yaxis.set_ticks(r_edges[r_edges<=df.at[i, 'beam_width']*np.pi/180/2]*180/np.pi)
    axs[2].tick_params(axis='y', colors='lime')
    
    f.suptitle(df.at[i, 'frequency'] + ' pol: ' + df.at[i, 'polarization'])
