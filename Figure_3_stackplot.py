#!/usr/bin/env python
"""
Figure_3_stackplot.py
Nathaniel A. Frissell
February 2024

This script is used to generate Figure 3 of the Frissell et al. (2024)
GRL manuscript on multi-instrument measurements of AGWs, MSTIDs, and LSTIDs.
"""

import os
import shutil
import glob
import string
letters = string.ascii_lowercase

import datetime
import calendar

import tqdm

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import xarray as xr

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
mpl.use('Agg')

# https://colorcet.holoviz.org/user_guide/Continuous.html
import colorcet

import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

import merra2CipsAirsTimeSeries
import gnss_dtec_gw
import lstid_ham
import HIAMCM
import sme_plot

pd.set_option('display.max_rows', None)

mpl.rcParams['font.size']           = 12
mpl.rcParams['font.weight']         = 'bold'
mpl.rcParams['axes.grid']           = True
mpl.rcParams['grid.linestyle']      = ':'
mpl.rcParams['figure.figsize']      = np.array([15, 8])
mpl.rcParams['axes.xmargin']        = 0
mpl.rcParams['ytick.labelsize']     = 22
mpl.rcParams['xtick.major.size']    = 6
#mpl.rcParams['xtick.minor.size']    = 10

#cbar_title_fontdict     = {'weight':'bold','size':30}
#cbar_ytick_fontdict     = {'size':30}
#xtick_fontdict          = {'weight': 'bold', 'size':30}
#ytick_major_fontdict    = {'weight': 'bold', 'size':28}
#ytick_minor_fontdict    = {'weight': 'bold', 'size':24}
#corr_legend_fontdict    = {'weight': 'bold', 'size':24}
#keo_legend_fontdict     = {'weight': 'normal', 'size':30}
#driver_xlabel_fontdict  = ytick_major_fontdict
#driver_ylabel_fontdict  = ytick_major_fontdict
#title_fontdict          = {'weight': 'bold', 'size':36}

cbar_title_fontdict     = {'weight':'bold','size':42}
cbar_ytick_fontdict     = {'weight':'bold','size':36}
xtick_fontdict          = {'weight': 'bold', 'size':mpl.rcParams['ytick.labelsize']}
ytick_major_fontdict    = {'weight': 'bold', 'size':24}
ytick_minor_fontdict    = {'weight': 'bold', 'size':24}
title_fontdict          = {'weight': 'bold', 'size':36}
ylabel_fontdict         = {'weight': 'bold', 'size':24}
reduced_legend_fontdict = {'weight': 'bold', 'size':20}

prm_dct = {}
prmd = prm_dct['meanSubIntSpect_by_rtiCnt'] = {}
prmd['scale_0']         = -0.025
prmd['scale_1']         =  0.025
prmd['cmap']            = mpl.cm.jet
prmd['cbar_label']      = 'MSTID Index'
prmd['cbar_tick_fmt']   = '%0.3f'
prmd['title']           = 'North American SuperDARN MSTID Index (~40\N{DEGREE SIGN}-60\N{DEGREE SIGN} Latititude)'
prmd['hist_bins']       = np.arange(-0.050,0.051,0.001)

prmd = prm_dct['meanSubIntSpect'] = {}
prmd['hist_bins']       = np.arange(-1500,1500,50)

prmd = prm_dct['intSpect_by_rtiCnt'] = {}
prmd['hist_bins']       = np.arange(0.,0.126,0.001)

prmd = prm_dct['intSpect'] = {}
prmd['hist_bins']       = np.arange(0.,2025,25)

prmd = prm_dct['sig_001_azm_deg'] = {}
prmd['scale_0']         = -180
prmd['scale_1']         =  180
#prmd['cmap']            = mpl.cm.hsv
# https://colorcet.holoviz.org/user_guide/Continuous.html
prmd['cmap']            = mpl.cm.get_cmap('cet_CET_C6') #colorcet.cyclic_rygcbmr_50_90_c64
prmd['cbar_label']      = 'MSTID Azimuth [deg]'
prmd['cbar_tick_fmt']   = '%.0f'
prmd['title']           = 'SuperDARN MSTID Propagation Azimuth'
prmd['hist_bins']       = np.arange(-180,185,10)
prmd['hist_polar']      = True

prmd = prm_dct['sig_001_vel_mps'] = {}
prmd['scale_0']         = 0
prmd['scale_1']         = 300
prmd['hist_bins']       = np.arange(0,500,10)
prmd['cbar_label']      = 'MSTID Speed [m/s]'
prmd['cbar_tick_fmt']   = '%.0f'
prmd['title']           = 'SuperDARN MSTID Speed'

prmd = prm_dct['sig_001_period_min'] = {}
prmd['scale_0']         = 15
prmd['scale_1']         = 60
prmd['cbar_label']      = 'MSTID Period [min]'
prmd['cbar_tick_fmt']   = '%.0f'
prmd['title']           = 'SuperDARN MSTID Period'
prmd['hist_bins']       = np.arange(15,65,2.5)

prmd = prm_dct['sig_001_lambda_km'] = {}
prmd['scale_0']         = 0
prmd['scale_1']         = 500
prmd['cbar_label']      = 'Horizontal Wavelength [km]'
prmd['cbar_tick_fmt']   = '%.0f'
prmd['title']           = 'SuperDARN Horizontal Wavelength'

prmd = prm_dct['meanSubIntSpect_by_rtiCnt_reducedIndex'] = {}
prmd['title']           = 'Reduced SuperDARN MSTID Index'
prmd['ylabel']          = 'Reduced SuperDARN\nMSTID Index'
prmd['ylim']            = (-5,5)

prmd = prm_dct['U_10HPA'] = {}
prmd['scale_0']         = -100.
prmd['scale_1']         =  100.
prmd['cmap']            = mpl.cm.bwr
prmd['cbar_label']      = 'U 10 hPa [m/s]'
prmd['title']           = 'MERRA2 Zonal Winds 10 hPa [m/s]'
prmd['data_dir']        = os.path.join('data','merra2','preprocessed')

prmd = prm_dct['U_1HPA'] = {}
prmd['scale_0']         = -100.
prmd['scale_1']         =  100.
prmd['cmap']            = mpl.cm.bwr
prmd['cbar_label']      = 'U 1 hPa [m/s]'
prmd['title']           = 'MERRA2 Zonal Winds 1 hPa [m/s]'
prmd['data_dir']        = os.path.join('data','merra2','preprocessed')

# ['DAILY_SUNSPOT_NO_', 'DAILY_F10.7_', '1-H_DST_nT', '1-H_AE_nT']
# DAILY_SUNSPOT_NO_  DAILY_F10.7_    1-H_DST_nT     1-H_AE_nT
# count       40488.000000  40488.000000  40488.000000  40488.000000
# mean           58.125963    103.032365    -10.984427    162.772167
# std            46.528777     29.990254     16.764279    175.810863
# min             0.000000     64.600000   -229.500000      3.500000
# 25%            17.000000     76.400000    -18.000000     46.000000
# 50%            50.000000     97.600000     -8.000000     92.000000
# 75%            90.000000    122.900000     -0.500000    215.000000
# max           220.000000    255.000000     64.000000   1637.000000

prmd = prm_dct['DAILY_SUNSPOT_NO_'] = {}
prmd['scale_0']         = 0
prmd['scale_1']         = 175.
prmd['cmap']            = mpl.cm.cividis
prmd['cbar_label']      = 'Daily SN'
prmd['title']           = 'Daily Sunspot Number'
prmd['data_dir']        = os.path.join('data','cdaweb_omni','preprocessed')

prmd = prm_dct['DAILY_F10.7_'] = {}
prmd['scale_0']         = 50.
prmd['scale_1']         = 200.
prmd['cmap']            = mpl.cm.cividis
prmd['cbar_label']      = 'Daily F10.7'
prmd['title']           = 'Daily F10.7 Solar Flux'
prmd['data_dir']        = os.path.join('data','cdaweb_omni','preprocessed')

prmd = prm_dct['1-H_DST_nT'] = {}
prmd['scale_0']         =  -75
prmd['scale_1']         =   25
prmd['cmap']            = mpl.cm.inferno_r
prmd['cbar_label']      = 'Dst [nT]'
prmd['title']           = 'Disturbance Storm Time Dst Index [nT]'
prmd['data_dir']        = os.path.join('data','cdaweb_omni','preprocessed')

prmd = prm_dct['1-H_AE_nT'] = {}
prmd['scale_0']         = 0
prmd['scale_1']         = 400
prmd['cmap']            = mpl.cm.viridis
prmd['cbar_label']      = 'AE [nT]'
prmd['title']           = 'Auroral Electrojet AE Index [nT]'
prmd['data_dir']        = os.path.join('data','cdaweb_omni','preprocessed')

prmd = prm_dct['OMNI_R_Sunspot_Number'] = {}
prmd['scale_0']         = 0
prmd['scale_1']         = 175.
prmd['cmap']            = mpl.cm.cividis
prmd['cbar_label']      = 'Daily SN'
prmd['title']           = 'Daily Sunspot Number'
prmd['data_dir']        = os.path.join('data','omni','preprocessed')

prmd = prm_dct['OMNI_F10.7'] = {}
prmd['scale_0']         = 50.
prmd['scale_1']         = 200.
prmd['cmap']            = mpl.cm.cividis
prmd['cbar_label']      = 'Daily F10.7'
prmd['title']           = 'Daily F10.7 Solar Flux'
prmd['data_dir']        = os.path.join('data','omni','preprocessed')

prmd = prm_dct['OMNI_Dst'] = {}
prmd['scale_0']         =  -75
prmd['scale_1']         =   25
prmd['cmap']            = mpl.cm.inferno_r
prmd['cbar_label']      = 'Dst [nT]'
prmd['title']           = 'Disturbance Storm Time Dst Index [nT]'
prmd['data_dir']        = os.path.join('data','omni','preprocessed')

prmd = prm_dct['OMNI_AE'] = {}
prmd['scale_0']         = 0
prmd['scale_1']         = 400
prmd['cmap']            = mpl.cm.viridis
prmd['cbar_label']      = 'AE [nT]'
prmd['title']           = 'Auroral Electrojet AE Index [nT]'
prmd['data_dir']        = os.path.join('data','omni','preprocessed')

prmd = prm_dct['merra2CipsAirsTimeSeries'] = {}
prmd['scale_0']         = -20
prmd['scale_1']         = 100
prmd['levels']          = 11
prmd['cmap']            = 'jet'
prmd['cbar_label']      = 'MERRA-2 Zonal Wind\n[m/s]'
prmd['title']           = 'MERRA-2 50\N{DEGREE SIGN} N Lat Zonal Winds + CIPS & AIRS GW Variance'

prmd = prm_dct['gnss_dtec_gw'] = {}
prmd['cmap']            = 'jet'
prmd['cbar_label']      = 'aTEC Amplitude (TECu)'
prmd['title']           = 'GNSS aTEC Amplitude at 115\N{DEGREE SIGN} W'

prmd = prm_dct['lstid_ham'] = {}
prmd['title']           = 'Amateur Radio 14 MHz LSTID Observations'

prmd = prm_dct['sme'] = {}
prmd['title']           = 'SuperMAG Electrojet Index (SME)'

prmd = prm_dct['reject_code'] = {}
prmd['title']           = 'MSTID Index Data Quality Flag'

# Reject code colors.
reject_codes = {}
# 0: Good Period (Not Rejected)
reject_codes[0] = {'color': mpl.colors.to_rgba('green'),  'label': 'Good Period'}
# 1: High Terminator Fraction (Dawn/Dusk in Observational Window)
reject_codes[1] = {'color': mpl.colors.to_rgba('blue'),   'label': 'Dawn / Dusk'}
# 2: No Data
reject_codes[2] = {'color': mpl.colors.to_rgba('red'),    'label': 'No Data'}
# 3: Poor Data Quality (including "Low RTI Fraction" and "Failed Quality Check")
reject_codes[3] = {'color': mpl.colors.to_rgba('gold'),   'label': 'Poor Data Quality'}
# 4: Other (including "No RTI Fraction" and "No Terminator Fraction")
reject_codes[4] = {'color': mpl.colors.to_rgba('purple'), 'label': 'Other'}
# 5: Not Requested (Outside of requested daylight times)
reject_codes[5] = {'color': mpl.colors.to_rgba('0.9'),   'label': 'Not Requested'}

def season_to_datetime(season):
    str_0, str_1 = season.split('_')
    sDate   = datetime.datetime.strptime(str_0,'%Y%m%d')
    eDate   = datetime.datetime.strptime(str_1,'%Y%m%d')
    return (sDate,eDate)

def plot_cbar(ax_info):
    cbar_pcoll = ax_info.get('cbar_pcoll')

    cbar_label      = ax_info.get('cbar_label')
    cbar_ticks      = ax_info.get('cbar_ticks')
    cbar_tick_fmt   = ax_info.get('cbar_tick_fmt','%0.3f')
    cbar_tb_vis     = ax_info.get('cbar_tb_vis',False)
    ax              = ax_info.get('ax')

    fig = ax.get_figure()

    box         = ax.get_position()

    x0  = 1.01
    wdt = 0.015
    y0  = 0.250
    hgt = (1-2.*y0)
    axColor = fig.add_axes([x0, y0, wdt, hgt])

    axColor.grid(False)
    cbar        = fig.colorbar(cbar_pcoll,orientation='vertical',cax=axColor,format=cbar_tick_fmt)

    cbar.set_label(cbar_label,fontdict=cbar_title_fontdict)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)

    axColor.set_ylim( *(cbar_pcoll.get_clim()) )

    labels = cbar.ax.get_yticklabels()
    fontweight  = cbar_ytick_fontdict.get('weight')
    fontsize    = cbar_ytick_fontdict.get('size')
    for label in labels:
        if fontweight:
            label.set_fontweight(fontweight)
        if fontsize:
            label.set_fontsize(fontsize)

#    if not cbar_tb_vis:
#        for inx in [0,-1]:
#            labels[inx].set_visible(False)

def reject_legend(fig):
    x0  = 1.01
    wdt = 0.015
    y0  = 0.250
    hgt = (1-2.*y0)

    axl= fig.add_axes([x0, y0, wdt, hgt])
    axl.axis('off')

    legend_elements = []
    for rej_code, rej_dct in reject_codes.items():
        color = rej_dct['color']
        label = rej_dct['label']
        # legend_elements.append(mpl.lines.Line2D([0], [0], ls='',marker='s', color=color, label=label,markersize=15))
        legend_elements.append(mpl.patches.Patch(facecolor=color,edgecolor=color,label=label))

    axl.legend(handles=legend_elements, loc='center left', fontsize = 42)

def my_xticks(sDate,eDate,ax,radar_ax=False,labels=True,short_labels=False,
                fmt='%d %b',fontdict=None,plot_axvline=True):
    if fontdict is None:
        fontdict = xtick_fontdict
    xticks      = []
    xticklabels = []
    curr_date   = sDate
    while curr_date < eDate:
        if radar_ax:
            xpos    = get_x_coords(curr_date,sDate,eDate)
        else:
            xpos    = curr_date
        xticks.append(xpos)
        xticklabels.append('')
        curr_date += datetime.timedelta(days=1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Define xtick label positions here.
    # Days of month to produce a xtick label.
    doms    = [1,15]

    curr_date   = sDate
    ytransaxes = mpl.transforms.blended_transform_factory(ax.transData,ax.transAxes)
    while curr_date < eDate:
        if curr_date.day in doms:
            if radar_ax:
                xpos    = get_x_coords(curr_date,sDate,eDate)
            else:
                xpos    = curr_date

            if plot_axvline:
                axvline = ax.axvline(xpos,-0.015,color='k')
                axvline.set_clip_on(False)

            if labels:
                ypos    = -0.025
                txt     = curr_date.strftime(fmt)
                ax.text(xpos,ypos,txt,transform=ytransaxes,
                        ha='left', va='top',rotation=0,
                        fontdict=fontdict)
            if short_labels:    
                if curr_date.day == 1:
                    ypos    = -0.030
                    txt     = curr_date.strftime('%b %Y')
                    ax.text(xpos,ypos,txt,transform=ytransaxes,
                            ha='left', va='top',rotation=0,
                            fontdict=fontdict)
                    ax.axvline(xpos,lw=2,zorder=5000,color='0.6',ls='--')
        curr_date += datetime.timedelta(days=1)

    xmax    = (eDate - sDate).total_seconds() / (86400.)
    if radar_ax:
        ax.set_xlim(0,xmax)
    else:
        ax.set_xlim(sDate,sDate+datetime.timedelta(days=xmax))

#    ax.grid(zorder=500)

def get_x_coords(win_sDate,sDate,eDate,full=False):
    x1  = (win_sDate - sDate).total_seconds()/86400.
    if not full:
        x1  = np.floor(x1)
    return x1

def get_y_coords(ut_time,st_uts,radar,radars):
    # Find start time index.
    st_uts      = np.array(st_uts)
    st_ut_inx   = np.digitize([ut_time],st_uts)-1
    
    # Find radar index.
    radar_inx   = np.where(radar == np.array(radars))[0]
    y1          = st_ut_inx*len(radars) + radar_inx
    return y1


def get_coords(radar,win_sDate,radars,sDate,eDate,st_uts,verts=True):
    # Y-coordinate.
    x1  = float(get_x_coords(win_sDate,sDate,eDate))
    y1  = float(get_y_coords(win_sDate.hour,st_uts,radar,radars)[0])

    if verts:
#        x1,y1   = x1+0,y1+0
        x2,y2   = x1+1,y1+0
        x3,y3   = x1+1,y1+1
        x4,y4   = x1+0,y1+1
        return ((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1))
    else:
        x0      = x1 + 0.5
        y0      = y1 + 0.5
        return (x0, y0)

def plot_mstid_values(data_df,ax,sDate=None,eDate=None,
        st_uts=[14, 16, 18, 20],
        xlabels=True, group_name=None,classification_colors=False,
        rasterized=False,radars=None,param=None,
        radar_ylabels=True,
        radarBox_fontsize='xx-large',**kwargs):

    prmd        = prm_dct.get(param,{})

#    scale_0     = prmd.get('scale_0',-0.025)
#    scale_1     = prmd.get('scale_1', 0.025)

    scale_0     = prmd.get('scale_0')
    scale_1     = prmd.get('scale_1')
    if scale_0 is None:
        scale_0 = np.nanmin(data_df.values)
    if scale_1 is None:
        scale_1 = np.nanmax(data_df.values)
    scale       = (scale_0, scale_1)

    cmap        = prmd.get('cmap',mpl.cm.jet)
    cbar_label  = prmd.get('cbar_label',param)

    if sDate is None:
        sDate = data_df.index.min()
        sDate = datetime.datetime(sDate.year,sDate.month,sDate.day)

        eDate = data_df.index.max()
        eDate = datetime.datetime(eDate.year,eDate.month,eDate.day) + datetime.timedelta(days=1)

    if radars is None:
        radars  = list(data_df.keys())

    # Reverse radars list order so that the supplied list is plotted top-down.
    radars  = radars[::-1]

    ymax    = len(st_uts) * len(radars)

    cbar_info   = {}
    bounds      = np.linspace(scale[0],scale[1],256)
    cbar_info['cbar_ticks'] = np.linspace(scale[0],scale[1],11)
    cbar_info['cbar_label'] = cbar_label

    norm    = mpl.colors.BoundaryNorm(bounds,cmap.N)

    if classification_colors:
        # Use colorscheme that matches MSTID Index in classification plots.
        from mstid.classify import MyColors
        scale_0             = -0.025
        scale_1             =  0.025
        my_cmap             = 'seismic'
        truncate_cmap       = (0.1, 0.9)
        my_colors           = MyColors((scale_0, scale_1),my_cmap=my_cmap,truncate_cmap=truncate_cmap)
        cmap                = my_colors.cmap
        norm                = my_colors.norm
                

    ################################################################################    
    current_date = sDate
    verts       = []
    vals        = []
    while current_date < eDate:
        for st_ut in st_uts:
            for radar in radars:
                win_sDate   = current_date + datetime.timedelta(hours=st_ut)

                val = data_df[radar].loc[win_sDate]

                if not np.isfinite(val):
                    continue

                if param == 'reject_code':
                    val = reject_codes.get(val,reject_codes[4])['color']

                vals.append(val)
                verts.append(get_coords(radar,win_sDate,radars,sDate,eDate,st_uts))

        current_date += datetime.timedelta(days=1)

    if param == 'reject_code':
        pcoll = PolyCollection(np.array(verts),edgecolors='0.75',linewidths=0.25,
                cmap=cmap,norm=norm,zorder=99,rasterized=rasterized)
        pcoll.set_facecolors(np.array(vals))
        ax.add_collection(pcoll,autolim=False)
    else:
        pcoll = PolyCollection(np.array(verts),edgecolors='face',linewidths=0,closed=False,
                cmap=cmap,norm=norm,zorder=99,rasterized=rasterized)
        pcoll.set_array(np.array(vals))
        ax.add_collection(pcoll,autolim=False)

    # Make gray missing data.
    ax.set_facecolor('0.90')

    # Add small radar labels on the left-hand side y-axis.
    if radar_ylabels is True:
        trans = mpl.transforms.blended_transform_factory(ax.transAxes,ax.transData)
        for rdr_inx,radar in enumerate(radars):
            for st_inx,st_ut in enumerate(st_uts):
                ypos = len(radars)*st_inx + rdr_inx + 0.5
                ax.text(-0.002,ypos,radar,transform=trans,ha='right',va='center')

    # Add large radar labels on the right hand side. ############################### 
    brkt_top    = len(radars)*len(st_uts)
    brkt_bot    = len(radars)*len(st_uts) - len(radars)
    brkt_x0     = 1.002
    brkt_x1     = 1.010
    brkt_color  = 'k'
    brkt_lw     = 2
   
    trans   = mpl.transforms.blended_transform_factory(ax.transAxes,ax.transData)
    rdr_lbls    = '\n'.join([rdr.upper() for rdr in radars[::-1]])
    bbox        = {'facecolor':'none','edgecolor':brkt_color,'pad':10.0,'lw':brkt_lw}
    rdr_box_x0  = 1.030
    arrowprops  = {'arrowstyle':'->','lw':brkt_lw}
    for st_inx,st_ut in enumerate(st_uts):
        brkt_top    = len(radars)*(st_inx+1)
        brkt_bot    = len(radars)*(st_inx+1) - len(radars)

        ax.plot([brkt_x1,brkt_x1],[brkt_bot,brkt_top],color=brkt_color,lw=brkt_lw,transform=trans,clip_on=False)
        ax.plot([brkt_x0,brkt_x1],[brkt_bot,brkt_bot],color=brkt_color,lw=brkt_lw,transform=trans,clip_on=False)
        ax.plot([brkt_x0,brkt_x1],[brkt_top,brkt_top],color=brkt_color,lw=brkt_lw,transform=trans,clip_on=False)

        brkt_half   = brkt_bot + (brkt_top-brkt_bot)/2.
        ax.annotate(rdr_lbls,(brkt_x1,brkt_half),xycoords=trans,xytext=(rdr_box_x0,(len(radars)*len(st_uts))/2.),textcoords=trans,bbox=bbox,
                arrowprops=arrowprops,va='center',ha='left',fontsize=radarBox_fontsize)

    # Add UT Time Labels
    for st_inx,st_ut in enumerate(st_uts):
        ypos    = st_inx*len(radars)
        xpos    = -0.035
        line    = ax.hlines(ypos,xpos,1.0,transform=trans,lw=3,zorder=100)
        line.set_clip_on(False)
        
        txt     = '{:02d}-{:02d}\nUT'.format(int(st_ut),int(st_ut+2))
        ypos   += len(radars)/2.
        xpos    = -0.025
#        ax.text(xpos,ypos,txt,transform=trans,
#                ha='center',va='center',rotation=90,fontdict=ytick_major_fontdict)
#        xpos    = -0.015
#        xpos    = -0.05
        xpos    = -0.025
        ax.text(xpos,ypos,txt,transform=trans,
                ha='right',va='center',rotation=0,fontdict=ytick_major_fontdict)

    xpos    = -0.035
    line    = ax.hlines(1.,xpos,1.0,transform=ax.transAxes,lw=3,zorder=100)
    line.set_clip_on(False)

    ax.set_ylim(0,ymax)

    # Set xticks and yticks to every unit to make a nice grid.
    # However, do not use this for actual labeling.
    yticks = list(range(len(radars)*len(st_uts)))
    ax.set_yticks(yticks)
    ytls = ax.get_yticklabels()
    for ytl in ytls:
        ytl.set_visible(False)

    my_xticks(sDate,eDate,ax,radar_ax=True,labels=xlabels)
    
    txt = ' '.join([x.upper() for x in radars[::-1]])
    if group_name is not None:
        txt = '{} ({})'.format(group_name,txt)
    ax.set_title(txt,fontdict=title_fontdict)

    ax_info         = {}
    ax_info['ax']   = ax
    ax_info['cbar_pcoll']   = pcoll
    ax_info['cbar_tick_fmt']= prmd.get('cbar_tick_fmt')
    ax_info.update(cbar_info)
    
    return ax_info

def list_seasons(yr_0=2010,yr_1=2022):
    """
    Give a list of the string codes for the default seasons to be analyzed.

    Season codes are in the form of '20101101_20110501'
    """
    yr = yr_0
    seasons = []
    while yr < yr_1:
        dt_0 = datetime.datetime(yr,11,1)
        dt_1 = datetime.datetime(yr+1,5,1)

        dt_0_str    = dt_0.strftime('%Y%m%d')
        dt_1_str    = dt_1.strftime('%Y%m%d')
        season      = '{!s}_{!s}'.format(dt_0_str,dt_1_str)
        seasons.append(season)
        yr += 1

    return seasons

class ParameterObject(object):
    def __init__(self,param,radars,seasons=None,
            output_dir='output',default_data_dir=os.path.join('data','mstid_index'),
            write_csvs=True,calculate_reduced=False):
        """
        Create a single, unified object for loading in SuperDARN MSTID Index data
        generated by the DARNTIDs library and output to CSV by mongo_to_csv.py.
        """

        # Create a Global Attributes dictionary that will be be applicable to all seasons
        # and printed out in the CSV file header.
        self.attrs_global = {}

        # Create parameter dictionary.
        prmd        = prm_dct.get(param,{})
        prmd['param'] = param
        if prmd.get('data_dir') is None:
            prmd['data_dir'] = default_data_dir
        self.prmd   = prmd

        # Store radar list.
        self.radars = radars
        self.attrs_global['radars']    = radars

        # Get list of seasons.
        if seasons is None:
            seasons = list_seasons()
        self.attrs_global['seasons']    = seasons

        # Load data into dictionary of dataframes.
        self.data = {season:{} for season in seasons}
        print('Loading data...')
        self._load_data()

        # Create a attrs_season dict for each season to hold statistics and other info that applies
        # to all data in a season, not just individual radars.
        for season in seasons:
            self.data[season]['attrs_season']   = {}

        try:
            # Track and report minimum orig_rti_fraction.
            # This is the percentage of range-beam cells reporting scatter in an
            # observational window. It is a critical parameter to correctly calculate
            # the reduced MSTID index and other statistcal measures.
            orig_rti_fraction       = self._load_data('orig_rti_fraction',selfUpdate=False)

            # Only keep the RTI Fractions from ``good'' periods.
            reject_dataDct          = self._load_data('reject_code',selfUpdate=False)
            for season in seasons:
                bad = reject_dataDct[season]['df'] != 0
                orig_rti_fraction[season]['df'][bad] = np.nan
            self.orig_rti_fraction  = orig_rti_fraction

            # Calculate min_orig_rti_fraction for all loaded seasons.
            orf                     = self.flatten(orig_rti_fraction)
            min_orig_rti_fraction   = np.nanmin(orf)
            self.attrs_global['min_orig_rti_fraction']  = min_orig_rti_fraction

            # Calculate min_orig_rti_fraction for each individual season.
            for season in seasons:
                orf     = orig_rti_fraction[season]['df']
                min_orf = np.nanmin(orf.values)
                self.data[season]['attrs_season']['min_orig_rti_fraction'] = min_orf

            # Calculate min_orig_rti_fraction for each individual radar.
            for season in seasons:
                orf     = orig_rti_fraction[season]['df']
                for radar in radars:
                    min_orf = np.nanmin(orf[radar].values)
                    self.data[season]['attrs_radars'][radar]['min_orig_rti_fraction'] = min_orf
        except:
            print('   ERROR calulating min_orig_rti_fraction while creating ParameterObject for {!s}'.format(param))

        self.output_dir = output_dir
        if write_csvs:
            print('Generating Season CSV Files...')
            for season in seasons:
                self.write_csv(season,output_dir=self.output_dir)

            csv_fpath   = os.path.join(self.output_dir,'radars.csv')
            self.lat_lons.to_csv(csv_fpath,index=False)

        if calculate_reduced:
            for season in seasons:
                self.calculate_reduced_index(season,write_csvs=write_csvs)

    def flatten(self,dataDct=None):
        """
        Return a single, flattened numpy array of all values from all seasons
        of a data dictionary. This is useful for calculating overall statistics.

        dataDct: Data dictionary to flatten. If None, use self.data.
        """
        if dataDct is None:
            dataDct = self.data

        data = []
        for season in dataDct.keys():
            tmp = dataDct[season]['df'].values.flatten()
            data.append(tmp)

        data = np.concatenate(data)
        return data

    def calculate_reduced_index(self,season,
            reduction_type='mean',daily_vals=True,zscore=True,
            smoothing_window   = '4D', smoothing_type='mean', write_csvs=True):
        """
        Reduce the MSTID index from all radars into a single number as a function of time.

        This function will work on any paramter, not just the MSTID index.
        """
        print("Calulating reduced MSTID index.")

        mstid_inx_dict  = {} # Create a place to store the data.

        df = self.data[season]['df']

        # Put everything into a dataframe.
        
        if daily_vals:
            date_list   = np.unique([datetime.datetime(x.year,x.month,x.day) for x in df.index])

            tmp_list        = []
            n_good_radars   = []    # Set up a system for parameterizing data quality.
            for tmp_sd in date_list:
                tmp_ed      = tmp_sd + datetime.timedelta(days=1)

                tf          = np.logical_and(df.index >= tmp_sd, df.index < tmp_ed)
                tmp_df      = df[tf]
                if reduction_type == 'median':
                    tmp_vals    = tmp_df.median().to_dict()
                elif reduction_type == 'mean':
                    tmp_vals    = tmp_df.mean().to_dict()

                tmp_list.append(tmp_vals)

                n_good  = np.count_nonzero(np.isfinite(tmp_df))
                n_good_radars.append(n_good)

            df = pd.DataFrame(tmp_list,index=date_list)
            n_good_df   = pd.Series(n_good_radars,df.index)
        else:
            n_good_df   = np.sum(np.isfinite(df),axis=1)

        data_arr    = np.array(df)
        if reduction_type == 'median':
            red_vals    = sp.nanmedian(data_arr,axis=1)
        elif reduction_type == 'mean':
            red_vals    = np.nanmean(data_arr,axis=1)

        ts  = pd.Series(red_vals,df.index)
        if zscore:
            ts  = (ts - ts.mean())/ts.std()

        reducedIndex = pd.DataFrame({'reduced_index':ts,'n_good_df':n_good_df},index=df.index)
#        reducedIndex['smoothed']    = reducedIndex['reduced_index'].rolling(smoothing_window,center=True).mean()
        reducedIndex['smoothed']    = getattr( reducedIndex['reduced_index'].rolling(smoothing_window,center=True), smoothing_type )()

        self.data[season]['reducedIndex']    = reducedIndex

        reducedIndex_attrs       = {}
        reducedIndex_attrs['reduction_type']     = reduction_type
        reducedIndex_attrs['zscore']             = zscore
        reducedIndex_attrs['daily_vals']         = daily_vals
        reducedIndex_attrs['smoothing_window']   = smoothing_window
        reducedIndex_attrs['smoothing_type']     = smoothing_type
        self.data[season]['reducedIndex_attrs']  = reducedIndex_attrs

        param           = '{!s}_reducedIndex'.format(self.prmd.get('param'))
        attrs_radars    = self.data[season]['attrs_radars']
        attrs_season    = self.data[season]['attrs_season']

        if write_csvs:
            output_dir = self.output_dir

            csv_fname       = '{!s}_{!s}.csv'.format(season,param)
            csv_fpath       = os.path.join(output_dir,csv_fname)
            with open(csv_fpath,'w') as fl:
                hdr = []
                hdr.append('# SuperDARN MSTID Index Datafile - Reduced Index')
                hdr.append('# Generated by Nathaniel Frissell, nathaniel.frissell@scranton.edu')
                hdr.append('# Generated on: {!s} UTC'.format(datetime.datetime.utcnow()))
                hdr.append('#')
                hdr.append('# Parameter: {!s}'.format(param))
                hdr.append('#')
                hdr.append('# {!s} Season Attributes:'.format(season))
                hdr.append('# {!s}'.format(attrs_season))
                hdr.append('#')
                hdr.append('# Individual Radar Attributes:')
                for radar,attr in attrs_radars.items():
                    hdr.append('# {!s}: {!s}'.format(radar,attr))
                hdr.append('#')
                hdr.append('# Reduction Attributes:')
                for attr in reducedIndex_attrs.items():
                    hdr.append('# {!s}'.format(attr))
                hdr.append('#')

                fl.write('\n'.join(hdr))
                fl.write('\n')
                
            reducedIndex.to_csv(csv_fpath,mode='a')

    def _load_data(self,param=None,selfUpdate=True):
        """
        Load data into data frames and store in self.data dictionary.
        
        param: Parameter to load. Use self.prmd.get('param') if None.
        selfUpdate:
            If True, update self.data and self.lat_lon with results and
                also return dictionary.
            If False, only return data dictionary.
        """

        data_dir    = self.prmd.get('data_dir')
        if param is None:
            param       = self.prmd.get('param')

        if selfUpdate is True:
            dataDct = self.data
        else:
            dataDct = {}
            for season in self.data.keys():
                dataDct[season] = {}

        lat_lons    = []
        for season in tqdm.tqdm(dataDct.keys(),desc='Seasons',dynamic_ncols=True,position=0):
            # Load all data from a season into a single xarray dataset.
            ds              = []
            attrs_radars    = {}

            data_vars = [] # Keep track of each column name in each data file.
            for radar in self.radars:
    #            fl  = os.path.join(data_dir,'sdMSTIDindex_{!s}_{!s}.nc'.format(season,radar))
                patt    = os.path.join(data_dir,'*{!s}_{!s}.nc'.format(season,radar))
                print('Loading: {!s}'.format(patt))
                fl      = glob.glob(patt)[0]
                tqdm.tqdm.write('--> {!s}: {!s}'.format(param,fl))
                dsr = xr.open_dataset(fl)
                ds.append(dsr)
                attrs_radars[radar] = dsr.attrs

                # Store radar lat / lons to creat a radar location file.
                lat_lons.append({'radar':radar,'lat':dsr.attrs['lat'],'lon':dsr.attrs['lon']})

                data_vars += list(dsr.data_vars) # Add columns names from the current data file.

            # Loop through each data set and ensure it has all of the columns.
            # If not, add that column with NaNs.
            # This makes the later concatenation into an XArray much easier.
            data_vars = list(set(data_vars)) # Get the unique set of column names.
            for dsr in ds:
                dvs = list(dsr.data_vars) # Get list of variable names in the current dataset.
                for dv in data_vars: # Loop through all of the required variable names.
                    if dv not in dvs: # If not present in the current data set, add it
                        dsr[dv] = dsr['lat'].copy() * np.nan

            dss   = xr.concat(ds,dim='index')

            dss = dss.stack(new_index=[...],create_index=False)
            dss = dss.swap_dims({'new_index':'index'})
            dss = dss.set_index({'index':'date'})

            # Convert parameter of interest to a datafame.
            df      = dss[param].to_dataframe()
            dfrs = {}
            for radar in tqdm.tqdm(self.radars,desc='Radars',dynamic_ncols=True,position=1,leave=False):
                tf      = df['radar'] == radar
                dft     = df[tf]
                dates   = dft.index
                vals    = dft[param]

                for date,val in zip(dates,vals):
                    if date not in dfrs:
                        dfrs[date] = {}
                    dfrs[date][radar] = val

            df  = pd.DataFrame(dfrs.values(),dfrs.keys())
            df  = df.sort_index()
            df.index.name                   = 'datetime'
            dataDct[season]['df']           = df
            dataDct[season]['attrs_radars'] = attrs_radars

        # Clean up lat_lon data table
        if selfUpdate is True:
            self.lat_lons    = pd.DataFrame(lat_lons).drop_duplicates()
        return dataDct

    def write_csv(self,season,output_dir=None):
        """
        Save data to CSV files.
        """

        param           = self.prmd.get('param')
        df              = self.data[season]['df']
        attrs_radars    = self.data[season]['attrs_radars']
        attrs_season    = self.data[season]['attrs_season']
        attrs_global    = self.attrs_global

        if output_dir is None:
            output_dir = self.output_dir

        csv_fname       = '{!s}_{!s}.csv'.format(season,param)
        csv_fpath       = os.path.join(output_dir,csv_fname)
        with open(csv_fpath,'w') as fl:
            hdr = []
            hdr.append('# SuperDARN MSTID Index Datafile')
            hdr.append('# Generated by Nathaniel Frissell, nathaniel.frissell@scranton.edu')
            hdr.append('# Generated on: {!s} UTC'.format(datetime.datetime.utcnow()))
            hdr.append('#')
            hdr.append('# Parameter: {!s}'.format(param))
            hdr.append('#')
            hdr.append('# Global Attributes (Applies to All Seasons Loaded in ParameterObject):')
            hdr.append('# {!s}'.format(attrs_global))

            hdr.append('#')
            hdr.append('# {!s} Season Attributes:'.format(season))
            hdr.append('# {!s}'.format(attrs_season))

            hdr.append('#')
            hdr.append('# Radar Attributes:')
            for radar,attr in attrs_radars.items():
                hdr.append('# {!s}: {!s}'.format(radar,attr))
            hdr.append('#')

            fl.write('\n'.join(hdr))
            fl.write('\n')
            
        df.to_csv(csv_fpath,mode='a')

    def plot_climatology(self,output_dir=None):

        if output_dir is None:
            output_dir = self.output_dir

        seasons = self.data.keys()
        radars  = self.radars
        param   = self.prmd['param']

        nrows   = 6
        ncols   = 2
        fig = plt.figure(figsize=(50,30))

        ax_list = []
        for inx,season in enumerate(seasons):
            print(' -->',season)
            ax      = fig.add_subplot(nrows,ncols,inx+1)

            data_df = self.data[season]['df']
            
            sDate, eDate = season_to_datetime(season)
            ax_info = plot_mstid_values(data_df,ax,radars=radars,param=param,sDate=sDate,eDate=eDate)
            min_orf   = po.data[season]['attrs_season'].get('min_orig_rti_fraction')
            ax_info['ax'].set_title('RTI Fraction > {:0.2f}'.format(min_orf),loc='right')
            ax_list.append(ax_info)

            season_yr0 = season[:4]
            season_yr1 = season[9:13]
            txt = '{!s} - {!s} Northern Hemisphere Winter'.format(season_yr0,season_yr1)
            ax.set_title(txt,fontdict=title_fontdict)

        fig.tight_layout(w_pad=2.25)

        if param == 'reject_code':
            reject_legend(fig)
        else:
            if len(ax_list) == 1:
                cbar_ax_inx = 0
            else:
                cbar_ax_inx = 1
            plot_cbar(ax_list[cbar_ax_inx])

        fpath = os.path.join(output_dir,'{!s}.png'.format(param))
        print('SAVING: ',fpath)
    #    fig.savefig(fpath)
        fig.savefig(fpath,bbox_inches='tight')

    def plot_histograms(self,output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir

        seasons = self.data.keys()
        radars  = self.radars
        param   = self.prmd['param']

        bins    = self.prmd.get('hist_bins',30)
        xlim    = self.prmd.get('hist_xlim',None)
        ylim    = self.prmd.get('hist_ylim',None)
        xlabel  = self.prmd.get('cbar_label',param)
        polar   = self.prmd.get('hist_polar',False)


        hist_dct    = {}
        ymax        = 1
        months      = []
        years       = []
        dates       = []
        for radar in radars:
            vals    = np.array([],dtype=float)
            for season,data_dct in self.data.items():
                df      = data_dct['df'][radar]
                tf      = np.isfinite(df.values)
                vals    = np.append(vals,df.values[tf])
                dates   = dates + df.index.tolist()

                if polar is True:
                    mean = sp.stats.circmean(vals,180.,-180.)
                    std  = sp.stats.circstd(vals,180.,-180.)
                else:
                    mean = np.mean(vals)
                    std  = np.std(vals)

                stats   = []
                stats.append('N: {:d}'.format(vals.size))
                stats.append('Mean: {:.1f}'.format(mean))
                stats.append('Std: {:.1f}'.format(std))

                dates   = list(set(dates))
                months  = list(set(months + [x.month for x in dates]))
                years   = list(set(years + [x.year for x in dates]))

            hist, bin_edges = np.histogram(vals,bins=bins)
            ymax    = np.max([ymax,np.max(hist)])

            tmp = {}
            tmp['hist']         = hist
            tmp['bin_edges']    = bin_edges
            tmp['stats']        = stats
            hist_dct[radar]     = tmp

        if xlim is None:
            xlim    = (np.min(bin_edges),np.max(bin_edges))

        if ylim is None:
            ylim    = (0, 1.025*ymax)

        rads_layout = {}
        rads_layout['pgr'] = (0,1)
        rads_layout['sas'] = (0,2)
        rads_layout['kap'] = (0,3)
        rads_layout['gbr'] = (0,4)

        rads_layout['cvw'] = (1,0)
        rads_layout['cve'] = (1,1)
        rads_layout['fhw'] = (1,2)
        rads_layout['fhe'] = (1,3)
        rads_layout['bks'] = (1,4)
        rads_layout['wal'] = (1,5)

        del_axs = []
        del_axs.append( (0,0) )
        del_axs.append( (0,5) )

        nrows           = 2
        ncols           = 6
        figsize         = (30,10)
        title_fontdict  = {'weight':'bold','size':18}
        label_fontdict  = {'weight':'bold','size':14}

        if polar is True:
            subplot_kw = {'projection':'polar'}
        else:
            subplot_kw = {}

        fig, axs = plt.subplots(nrows,ncols,figsize=figsize,subplot_kw=subplot_kw)
        for radar in radars:
            pos = rads_layout.get(radar)
            ax  = axs[pos]

            hist        = hist_dct[radar]['hist']
            bin_edges   = hist_dct[radar]['bin_edges']

            if polar is True:
                bin_edges   = np.radians(bin_edges)
                width       = np.diff(bin_edges)
                ax.bar(bin_edges[:-1],hist,width=width)
                ax.set_theta_zero_location("N")  # theta=0 at the top
                ax.set_theta_direction(-1)  # theta increasing clockwise
                stats_x = 0.800
                stats_y = 1.000
            else:
                width       = bin_edges[1]-bin_edges[0]
                ax.bar(bin_edges[:-1],hist,width=width)

                ax.set_xlabel(xlabel,fontdict=label_fontdict)
                ax.set_ylabel('Counts',fontdict=label_fontdict)

                ax.set_xlim(xlim)
                stats_x = 0.675
                stats_y = 0.975
            ax.set_ylim(ylim)

            stats       = hist_dct[radar]['stats']
            bbox        = {'boxstyle':'round','facecolor':'white','alpha':0.8}
            ax.text(stats_x,stats_y,'\n'.join(stats),va='top',transform=ax.transAxes,bbox=bbox)
            ax.set_title(radar,fontdict=title_fontdict)

        for del_ax in del_axs:
            ax  = axs[del_ax]
            ax.remove()

        title   = []
        title.append('Daytime '+self.prmd.get('title',param))
        year_str    = '{!s} - {!s}'.format(min(years),max(years))
        month_str   = ', '.join([calendar.month_abbr[x] for x in months])
        title.append('Years: {!s}; Months: {!s}'.format(year_str,month_str))
        fig.text(0.5,1.0,'\n'.join(title),ha='center',fontdict={'weight':'bold','size':28})

        fig.tight_layout()
        fpath = os.path.join(output_dir,'{!s}_histograms.png'.format(param))
        print('SAVING: ',fpath)
        fig.savefig(fpath,bbox_inches='tight')

def stackplot(po_dct,params,season,radars=None,sDate=None,eDate=None,fpath='stackplot.png'):

    _sDate, _eDate = season_to_datetime(season)
    if sDate is None:
        sDate = _sDate
    if eDate is None:
        eDate = _eDate

    print(' Plotting Stackplot: {!s}'.format(fpath))
    nrows   = len(params)
    ncols   = 1
    fig = plt.figure(figsize=(25,nrows*5))

    ax_list = []
    for inx,param in enumerate(params):
        ax      = fig.add_subplot(nrows,ncols,inx+1)

        if param.endswith('_reducedIndex'):
            base_param      = param.rstrip('_reducedIndex')
            plotType        = 'reducedIndex'
        elif    (param == 'merra2CipsAirsTimeSeries' 
              or param == 'gnss_dtec_gw' 
              or param == 'lstid_ham'
              or param == 'sme'
              or param == 'HIAMCM'):
            base_param      = param
            plotType        = param
        else:
            base_param      = param
            plotType        = 'climo'

        # Get Parameter Object
        po      = po_dct.get(base_param)
        if plotType == 'reducedIndex':
            data_df = po.data[season]['reducedIndex']
            prmd    = prm_dct.get(param,{})
        elif  (plotType == 'merra2CipsAirsTimeSeries' 
            or plotType == 'gnss_dtec_gw' 
            or plotType == 'lstid_ham'
            or plotType == 'sme'
            or plotType == 'HIAMCM'):
            data_df = None
            prmd    = prm_dct.get(param,{})
        else:
            data_df = po.data[season]['df']
            prmd    = po.prmd

        if inx == nrows-1:
            xlabels = True
        else:
            xlabels = False

        if plotType == 'reducedIndex':
            handles = []

            xx      = data_df.index
            yy      = data_df['reduced_index']
            label   = 'Raw'
            hndl    = ax.plot(xx,yy,label=label)
            handles.append(hndl[0])

            xx      = data_df.index
            yy      = data_df['smoothed']
            ri_attrs    = po.data[season]['reducedIndex_attrs']
            label   = '{!s} Rolling {!s}'.format(ri_attrs['smoothing_window'],ri_attrs['smoothing_type'].capitalize())
            hndl    = ax.plot(xx,yy,lw=3,label=label)
            handles.append(hndl[0])

            ax1     = ax.twinx()
            xx      = data_df.index
            yy      = data_df['n_good_df']
            label   = 'n Data Points'
            hndl    = ax1.plot(xx,yy,color='0.8',ls='--',label=label)
            ax1.set_ylabel('n Data Points\n(Dashed Line)',fontdict=ylabel_fontdict)
            handles.append(hndl[0])

            ax.legend(handles=handles,loc='lower left',ncols=3,prop=reduced_legend_fontdict)

            ax.set_xlim(sDate,eDate)

            min_orf   = po.data[season]['attrs_season'].get('min_orig_rti_fraction')
            ax.set_title('RTI Fraction > {:0.2f}'.format(min_orf),loc='right')

            ax_info = {}
            ax_info['ax']           = ax
        elif plotType == 'merra2CipsAirsTimeSeries':
            mca     = merra2CipsAirsTimeSeries.Merra2CipsAirsTS()
            if 'scale_0' in prmd:
                prmd['vmin'] = prmd['scale_0']
            if 'scale_1' in prmd:
                prmd['vmax'] = prmd['scale_1']
            result  = mca.plot_ax(ax,plot_cbar=False,ylabel_fontdict=ylabel_fontdict,**prmd)

            ax.set_xlim(sDate,eDate)

            if xlabels is False:
                ax.set_xlabel('')

            ax_info = {}
            ax_info['ax']           = ax
            ax_info['cbar_pcoll']   = result['cbar_pcoll']
            ax_info['cbar_label']   = prmd.get('cbar_label')
        elif plotType == 'gnss_dtec_gw':
            dTEC = gnss_dtec_gw.GNSS_dTEC_GW()
            result  = dTEC.plot_ax(ax,plot_cbar=False,ylabel_fontdict=ylabel_fontdict,**prmd)

            ax.set_xlim(sDate,eDate)
            ax.set_ylim(40,50)

            if xlabels is False:
                ax.set_xlabel('')

            ax_info = {}
            ax_info['ax']           = ax
            ax_info['cbar_pcoll']   = result['cbar_pcoll']
            ax_info['cbar_label']   = prmd.get('cbar_label')
        elif plotType == 'lstid_ham':
            lstid = lstid_ham.LSTID_HAM()
            result  = lstid.plot_ax(ax,legend_fontsize='x-large',ylabel_fontdict=ylabel_fontdict,
                    legend_ncols=1,**prmd)

            ax.set_xlim(sDate,eDate)

            if xlabels is False:
                ax.set_xlabel('')

            if xlabels is False:
                ax.set_xlabel('')

            ax_info = {}
            ax_info['ax']           = ax
        elif plotType == 'sme':
            sme     = sme_plot.SME_PLOT()
            result  = sme.plot_ax(ax,legend_fontsize='x-large',ylabel_fontdict=ylabel_fontdict,
                    xlim=(sDate,eDate),**prmd)

            ax.set_xlim(sDate,eDate)

            if xlabels is False:
                ax.set_xlabel('')

            if xlabels is False:
                ax.set_xlabel('')

            ax_info = {}
            ax_info['ax']           = ax
        elif plotType == 'HIAMCM':
            hiamcm  = HIAMCM.HIAMCM()
            result  = hiamcm.plot_ax(ax,prm='ww',lats=(40.,60.),
                                     plot_cbar=False,ylabel_fontdict=ylabel_fontdict,**prmd)

            ax.set_xlim(sDate,eDate)

            if xlabels is False:
                ax.set_xlabel('')

            ax_info = {}
            ax_info['ax']           = ax
            ax_info['cbar_pcoll']   = result['cbar_pcoll']
            ax_info['cbar_label']   = result.get('cbar_label')

            prmd['title'] = result.get('title')
        else: 
            if radars is None:
                _radars = po.radars
            else:
                _radars = radars

            ax_info = plot_mstid_values(data_df,ax,radars=_radars,param=param,xlabels=xlabels,
                    sDate=sDate,eDate=eDate)

            min_orf   = po.data[season]['attrs_season'].get('min_orig_rti_fraction')
            ax_info['ax'].set_title('RTI Fraction > {:0.2f}'.format(min_orf),loc='right')
            ax_info['radar_ax']     = True
        ax_list.append(ax_info)

        ylim    = prmd.get('ylim')
        if ylim is not None:
            ax.set_ylim(ylim)

        ylabel  = prmd.get('ylabel')
        if ylabel is not None:
            ax.set_ylabel(ylabel,fontdict=ylabel_fontdict)

        txt = '({!s}) '.format(letters[inx])+prmd.get('title',param)
        left_title_fontdict  = {'weight': 'bold', 'size': 24}
        ax.set_title('')
        ax.set_title(txt,fontdict=left_title_fontdict,loc='left')

        season_yr0 = season[:4]
        season_yr1 = season[9:13]
        txt = '{!s} - {!s} Northern Hemisphere Winter'.format(season_yr0,season_yr1)
        fig.text(0.5,1.01,txt,ha='center',fontdict=title_fontdict)

    # Set X-Labels and X-Tick Labels
    for inx,(param,ax_info) in enumerate(zip(params,ax_list)):
        ax          = ax_info.get('ax')
        ax.set_xlabel('')
        radar_ax    = ax_info.get('radar_ax',False)

        if inx != nrows-1:
            fontdict = ylabel_fontdict.copy()
            fontdict['weight']  = 'normal'
            fontdict['size']    = 18
        else:
            fontdict = ylabel_fontdict.copy()

        my_xticks(sDate,eDate,ax,radar_ax=radar_ax,
                  labels=False,short_labels=True,plot_axvline=False)

    fig.tight_layout()

    for param,ax_info in zip(params,ax_list):
        # Plot Colorbar ################################################################
        ax  = ax_info.get('ax')
        if param == 'reject_code':
            ax_pos  = ax.get_position()
            x0  = 1.005
            wdt = 0.015
            y0  = ax_pos.extents[1]
            hgt = ax_pos.height

            axl= fig.add_axes([x0, y0, wdt, hgt])
            axl.axis('off')

            legend_elements = []
            for rej_code, rej_dct in reject_codes.items():
                color = rej_dct['color']
                label = rej_dct['label']
                # legend_elements.append(mpl.lines.Line2D([0], [0], ls='',marker='s', color=color, label=label,markersize=15))
                legend_elements.append(mpl.patches.Patch(facecolor=color,edgecolor=color,label=label))

            axl.legend(handles=legend_elements, loc='center left', fontsize = 18)
        elif ax_info.get('cbar_pcoll') is not None:
            ax_pos  = ax.get_position()
            x0  = 1.01
            wdt = 0.015
            y0  = ax_pos.extents[1]
            hgt = ax_pos.height
            axColor = fig.add_axes([x0, y0, wdt, hgt])
            axColor.grid(False)

            cbar_pcoll      = ax_info.get('cbar_pcoll')
            cbar_label      = ax_info.get('cbar_label')
            cbar_ticks      = ax_info.get('cbar_ticks')
            cbar_tick_fmt   = prmd.get('cbar_tick_fmt')
            cbar_tb_vis     = ax_info.get('cbar_tb_vis',False)

			# fraction : float, default: 0.15
			#     Fraction of original axes to use for colorbar.
			# 
			# shrink : float, default: 1.0
			#     Fraction by which to multiply the size of the colorbar.
			# 
			# aspect : float, default: 20
			#     Ratio of long to short dimensions.
			# 
			# pad : float, default: 0.05 if vertical, 0.15 if horizontal
			#     Fraction of original axes between colorbar and new image axes.
            cbar  = fig.colorbar(cbar_pcoll,orientation='vertical',
                    cax=axColor,format=cbar_tick_fmt)

            cbar_label_fontdict = {'weight': 'bold', 'size': 24}
            cbar.set_label(cbar_label,fontdict=cbar_label_fontdict)
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)

            cbar.ax.set_ylim( *(cbar_pcoll.get_clim()) )

            labels = cbar.ax.get_yticklabels()
            fontweight  = cbar_ytick_fontdict.get('weight')
            fontsize    = 18
            for label in labels:
                if fontweight:
                    label.set_fontweight(fontweight)
                if fontsize:
                    label.set_fontsize(fontsize)

    fig.savefig(fpath,bbox_inches='tight')

def prep_dir(path,clear=False):
    if clear:
        if os.path.exists(path):
            shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':

    output_base_dir     = 'output'
#    mstid_data_dir      = os.path.join('data','mongo_out','mstid_MUSIC','guc')
#    mstid_data_dir      = os.path.join('data','mongo_out','mstid_GSMR_fitexfilter_using_mstid_2016_dates','guc')
#    mstid_data_dir      = os.path.join('data','mongo_out','mstid_2016','guc')
    mstid_data_dir      = os.path.join('data','mongo_out','mstid_GSMR_fitexfilter_rtiThresh-0.25','guc')
    plot_climatologies  = False
    plot_histograms     = False
    plot_stackplots     = True

    radars          = []
    # 'High Latitude Radars'
    radars.append('pgr')
    radars.append('sas')
    radars.append('kap')
    radars.append('gbr')
    # 'Mid Latitude Radars'
    radars.append('cvw')
    radars.append('cve')
    radars.append('fhw')
    radars.append('fhe')
    radars.append('bks')
    radars.append('wal')

#    # Ordered by Longitude
#    radars          = []
#    radars.append('cvw')
#    radars.append('pgr')
#    radars.append('cve')
#    radars.append('sas')
#    radars.append('fhw')
#    radars.append('fhe')
#    radars.append('kap')
#    radars.append('bks')
#    radars.append('wal')
#    radars.append('gbr')

    params = []
    params.append('meanSubIntSpect_by_rtiCnt') # This is the MSTID index.
#    params.append('meanSubIntSpect')
#    params.append('intSpect_by_rtiCnt')
#    params.append('intSpect')

#    params.append('sig_001_azm_deg')
#    params.append('sig_001_lambda_km')
#    params.append('sig_001_period_min')
#    params.append('sig_001_vel_mps')

#    params.append('reject_code')

#    params.append('U_10HPA')
#    params.append('U_1HPA')

#    params.append('OMNI_R_Sunspot_Number')
#    params.append('OMNI_Dst')
#    params.append('OMNI_F10.7')
#    params.append('OMNI_AE')

#    params.append('1-H_AE_nT')
#    params.append('1-H_DST_nT')
#    params.append('DAILY_F10.7_')
#    params.append('DAILY_SUNSPOT_NO_')

    seasons = list_seasons()
#    seasons = []
#    seasons.append('20121101_20130501')
##    seasons.append('20171101_20180501')
#    seasons.append('20181101_20190501')

################################################################################
# LOAD RADAR DATA ##############################################################

    po_dct  = {}
    for param in params:
        # Generate Output Directory
        output_dir  = os.path.join(output_base_dir,param)
        prep_dir(output_dir,clear=True)

        if param == 'meanSubIntSpect_by_rtiCnt':
            calculate_reduced=True
        else:
            calculate_reduced=False

        po = ParameterObject(param,radars=radars,seasons=seasons,
                output_dir=output_dir,default_data_dir=mstid_data_dir,
                calculate_reduced=calculate_reduced)

        po_dct[param]   = po

################################################################################
# CLIMATOLOGIES ################################################################

    if plot_climatologies:
        for param,po in po_dct.items():
            print('Plotting Climatology: {!s}'.format(param))
            po.plot_climatology()

################################################################################
# HISTOGRAMS ###################################################################
    if plot_histograms:
        for param,po in po_dct.items():
            print('Plotting Climatology: {!s}'.format(param))
            po.plot_histograms()

################################################################################
# STACKPLOTS ###################################################################

    stack_sets  = {}
##    ss = stack_sets['cdaweb_omni'] = []
##    ss.append('meanSubIntSpect_by_rtiCnt')
##    ss.append('1-H_AE_nT')
##    ss.append('1-H_DST_nT')
##    ss.append('DAILY_F10.7_')
###    ss.append('DAILY_SUNSPOT_NO_')

#    ss = stack_sets['omni'] = []
#    ss.append('meanSubIntSpect_by_rtiCnt')
#    ss.append('OMNI_AE')
#    ss.append('OMNI_Dst')
##    ss.append('OMNI_F10.7')
##    ss.append('OMNI_R_Sunspot_Number')
#
#    ss = stack_sets['mstid_merra2'] = []
#    ss.append('meanSubIntSpect_by_rtiCnt')
#    ss.append('U_1HPA')
#    ss.append('U_10HPA')
#
#    ss = stack_sets['data_quality'] = []
#    ss.append('meanSubIntSpect_by_rtiCnt')
#    ss.append('reject_code')

#    ss = stack_sets['mstid_index_reduced'] = []
#    ss.append('meanSubIntSpect_by_rtiCnt')
#    ss.append('meanSubIntSpect_by_rtiCnt_reducedIndex')
#
##    ss = stack_sets['mstid_index'] = []
##    ss.append('meanSubIntSpect_by_rtiCnt')

    ss = stack_sets['figure_3'] = []
    ss.append('merra2CipsAirsTimeSeries')
    ss.append('HIAMCM')
    ss.append('gnss_dtec_gw')
    ss.append('meanSubIntSpect_by_rtiCnt')
    ss.append('lstid_ham')
    ss.append('sme')
#    ss.append('meanSubIntSpect_by_rtiCnt_reducedIndex')

    if plot_stackplots:
        for stack_code,stack_params in stack_sets.items():
            stack_dir  = os.path.join(output_base_dir,'stackplots',stack_code)
            prep_dir(stack_dir,clear=True)
            for season in seasons:
                if stack_code == 'figure_3':
                    if season != '20181101_20190501':
                        continue
                png_name    = '{!s}_stack_{!s}.png'.format(season,stack_code)
                png_path    = os.path.join(stack_dir,png_name) 

                stackplot(po_dct,stack_params,season,fpath=png_path)
