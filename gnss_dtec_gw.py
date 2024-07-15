#!/usr/bin/env python
"""
This class will generate a time series plot of Global Navigation Satellite System 
absolute Total Electron Content (GNSS aTEC).
"""
import os
import shutil
import datetime

import numpy as np
import pandas as pd

import h5py

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

class GNSS_dTEC_GW(object):
    def __init__(self,data_in='data/gnss_dtec/dTEC115W180days_abs_2019.h5'):
        self.load_data(data_in)
    
    def load_data(self,data_in):
        fin     = h5py.File(data_in,'r')
        dtec    = fin['115W/dtec'][:]

#        xx180   = fin['115W/xx'][:] #The days array
        sDate   = datetime.datetime(2018,11,1)
        dates   = [sDate]
        for inx in range(180):
            dates.append(dates[-1] + datetime.timedelta(days=1))
        lats    = fin['115W/yy'][:,0] #The latitude array
        fin.close()

        self.data_in    = data_in
        self.dtec       = dtec
        self.dates      = dates
        self.lats       = lats

    def plot_figure(self,png_fpath='output.png',figsize=(16,8),**kwargs):

        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1,1,1)

        result  = self.plot_ax(ax,**kwargs)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax,vmin=-20,vmax=100,levels=11,cmap='jet',ylim=(32,50),plot_cbar=True,ylabel_fontdict={},**kwargs):
        fig     = ax.get_figure()

        dtec    = self.dtec
        dates   = self.dates
        lats    = self.lats

        levels  =[0,0.02,0.03,0.04,0.06,0.08,0.1,0.12]
        cbar_pcoll = ax.contourf(dates[:-1],lats,dtec.T,levels=levels,cmap=cmap)
        ax.set_ylabel('Latitude',fontdict=ylabel_fontdict)
        ax.set_xlabel('UTC Date')
        ax.set_ylim(ylim)
        ax.grid(False)

        if plot_cbar:
            fig.colorbar(cbar_pcoll,label=r'aTEC amplitude (TECu)')

        result  = {}
        result['cbar_pcoll'] = cbar_pcoll
        return result

if __name__ == '__main__':
    output_dir = os.path.join('output','gnss_dtec')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    png_fname   = 'gnss_dtec.png'
    png_fpath   = os.path.join(output_dir,png_fname)

    dTEC = GNSS_dTEC_GW()
    dTEC.plot_figure(png_fpath=png_fpath)
