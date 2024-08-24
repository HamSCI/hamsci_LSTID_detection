#!/usr/bin/env python
"""
This class will generate a time series plot of Mary Lou West's LSTID Amateur Radio Statistics.
"""
import os
import shutil
import datetime

import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

def load_supermag():
#    # Load Raw SuperMAG data, remove out of range and bad data, and 
#    # compute SME.
#    data_dir    = os.path.join('data','supermag_sme')
#    fpath       = os.path.join(data_dir,'20230808-02-16-supermag.csv.bz2')
#
#    df_0        = pd.read_csv(fpath,parse_dates=[0])
#    df_0        = df_0.set_index('Date_UTC')
#
#    sDate       = datetime.datetime(2010,1,1)
#    eDate       = datetime.datetime(2023,1,1)
#    tf          = np.logical_and(df_0.index >= sDate, df_0.index < eDate)
#    df_0        = df_0[tf]
#    df_0        = df_0.replace(999999,np.nan)
#
#    df_0['SME'] = df_0['SMU'] - df_0['SML']
#
#    sDate_str   = sDate.strftime('%Y%m%d')
#    eDate_str   = eDate.strftime('%Y%m%d')
#    out_fname   = '{!s}_{!s}_SME.csv.bz2'.format(sDate_str,eDate_str)
#    out_path    = os.path.join(data_dir,out_fname)
#
#    df_0.to_csv(out_path)

    data_dir    = os.path.join('data','supermag_sme')
    fpath       = os.path.join(data_dir,'20100101_20230101_SME.csv.bz2')

    df_0        = pd.read_csv(fpath,parse_dates=[0])
    df_0        = df_0.set_index('Date_UTC')

    return df_0

class SME_PLOT(object):
    def __init__(self):
        self.df = load_supermag()

    def plot_figure(self,png_fpath='output.png',figsize=(16,5),**kwargs):
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1,1,1)

        result  = self.plot_ax(ax,**kwargs)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax,xlim=None,ylabel_fontdict={},legend_fontsize='large',**kwargs):
        fig     = ax.get_figure()
        
        df      = self.df

        hndls   = []

        if xlim is None:
            xlim = (min(df.index), max(df.index))
        tf      = np.logical_and(df.index >= xlim[0], df.index < xlim[1])
        df      = df[tf].copy()

        xx      = df.index
        yy      = df['SME']
        ylabel  = 'SME Index [nT]'
        ax.set_ylabel(ylabel,fontdict=ylabel_fontdict)
        ax.set_xlabel('UTC Date')

        ax.plot(xx,yy,color='k')

        title   = 'SuperMAG SME Index'
        ax.set_title(title)

        ax.set_xlim(xlim)

        result  = {}
        result['title'] = title
        return result

if __name__ == '__main__':
    output_dir = os.path.join('output','sme_plot')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    png_fname   = 'sme_plot.png'
    png_fpath   = os.path.join(output_dir,png_fname)

    sme_plot    = SME_PLOT()
    sme_plot.plot_figure(png_fpath=png_fpath)
    print(png_fpath)
