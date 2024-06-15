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

def load_omni():
    data_dir    = os.path.join('data','cdaweb_omni')
    fpath       = os.path.join(data_dir,'20230530_OMNI2_H0_MRG1HR_99633.csv')

    df_0        = pd.read_csv(fpath,parse_dates=[0],comment='#')
    dt_key      = 'TIME_AT_CENTER_OF_HOUR_yyyy-mm-ddThh:mm:ss.sssZ'
    df_0.index  = df_0[dt_key].apply(lambda x: x.replace(tzinfo=None)).values
    del df_0[dt_key]

    params  = ['DAILY_SUNSPOT_NO_', 'DAILY_F10.7_', '1-H_DST_nT', '1-H_AE_nT'] 

    # Set bad values to NaN.
    bad = {}
    bad['DAILY_F10.7_'] = 999.9
    bad['1-H_AE_nT']    = 9999
    for col,val in bad.items():
        tf = df_0[col] == val
        df_0.loc[tf,col] = np.nan

    return df_0

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



class LSTID_HAM(object):
    def __init__(self,data_in='data/lstid_ham/WinterHam_2018_19.csv'):
        self.load_data(data_in)
    
    def load_data(self,data_in):
        df  = pd.read_csv(data_in,comment='#',parse_dates=[0])

        # Convert data columns to floats.
        cols = ['start_time', 'end_time', 'low_range_km', 'high_range_km', 'tid_hours', 'range_km', 'cycles', 'period_hr']
        for col in cols:
            df.loc[:,col] = pd.to_numeric(df[col],errors='coerce')

        self.data_in    = data_in
        self.df         = df

    def plot_figure(self,png_fpath='output.png',figsize=(16,5),**kwargs):

        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1,1,1)

        result  = self.plot_ax(ax,**kwargs)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax,xlim=None,ylabel_fontdict={},legend_fontsize='large',
            legend_ncols=2,plot_ae=False,plot_dst=False,plot_sme=False,**kwargs):
        fig     = ax.get_figure()
        
        df      = self.df

        hndls   = []

        xx      = df['date']

        if xlim is None:
            xlim = (min(xx), max(xx))

        yy      = df['tid_hours']
        label  = 'TID Occurrence [hr]'
        hndl    = ax.bar(xx,yy,width=1,label=label,color='green',align='edge')
        hndls.append(hndl)
        ylabel  = 'LSTID Occurrence [hr]\nLSTID Period [hr]'
        ax.set_ylabel(ylabel,fontdict=ylabel_fontdict)
        ax.set_xlabel('UTC Date')

        yy      = df['period_hr']
        ylabel  = 'LSTID Period [hr]'
        hndl    = ax.bar(xx,yy,label=ylabel,color='orange',align='edge')
        hndls.append(hndl)
        ax.legend(handles=hndls,loc='upper right',fontsize=legend_fontsize,ncols=legend_ncols)

        if plot_ae:
            omni = load_omni()
            tf = np.logical_and(omni.index >= xlim[0], omni.index < xlim[1])
            omni = omni[tf].copy()

            ax2 = ax.twinx()
            ax2_xx = omni.index
            ax2_yy = omni['1-H_AE_nT']
            ax2.plot(ax2_xx,ax2_yy,color='k')
            ax2.set_ylabel('AE [nT]')

        if plot_dst:
            omni = load_omni()
            tf = np.logical_and(omni.index >= xlim[0], omni.index < xlim[1])
            omni = omni[tf].copy()

            ax2 = ax.twinx()
            ax2_xx = omni.index
            ax2_yy = omni['1-H_DST_nT']
            ax2.plot(ax2_xx,ax2_yy,color='k')
            ax2.axhline(0,color='k')
            ax2.set_ylabel('DST [nT]')

        if plot_sme:
            supermag = load_supermag()
            tf = np.logical_and(supermag.index >= xlim[0], supermag.index < xlim[1])
            supermag = supermag[tf].copy()

            ax2 = ax.twinx()
            ax2_xx = supermag.index
            ax2_yy = supermag['SME']
            ax2.plot(ax2_xx,ax2_yy,color='k')
            ax2.set_ylabel('SME [nT]')

        title   = 'Amateur Radio 14 MHz LSTID Observations'
        ax.set_title(title)

        ax.set_xlim(xlim)

        result  = {}
        result['title'] = title
        return result

if __name__ == '__main__':
    output_dir = os.path.join('output','lstid_ham')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for plot_sme in [True,False]:
        lstid = LSTID_HAM()
        if plot_sme:
            png_fname   = 'lstid_ham_sme.png'
        else:
            png_fname   = 'lstid_ham.png'

        png_fpath   = os.path.join(output_dir,png_fname)

        lstid.plot_figure(png_fpath=png_fpath,plot_sme=plot_sme)
        print(png_fpath)
