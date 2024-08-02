#!/usr/bin/env python
"""
This class will generate a time series plot of LSTID Amateur Radio
Statistics generated using the automated sinFit method.
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

class LSTID_HAM(object):
    def __init__(self,data_in='data/lstid_ham/WinterHam_2018_19.csv'):
        self.load_data(data_in)
    
    def load_data(self,data_in):
        df  = pd.read_csv(data_in,comment='#',parse_dates=[0])

        # Convert data columns to floats.
        cols = ['start_time', 'end_time', 'low_range_km', 'high_range_km', 'tid_hours', 'range_range', 'cycles', 'period_hr']
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
            legend_ncols=2,**kwargs):
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

    lstid = LSTID_HAM()
    png_fname   = 'lstid_ham.png'
    png_fpath   = os.path.join(output_dir,png_fname)

    lstid.plot_figure(png_fpath=png_fpath)
    print(png_fpath)
