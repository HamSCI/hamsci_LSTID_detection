#!/usr/bin/env python
"""
This class will generate a time series plot with MERRA2 winds in the background a AIRS and CIPS
gravity wave variance on top.
"""
import os
import shutil
import datetime

import numpy as np
import pandas as pd

import xarray as xr

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

class Merra2CipsAirsTS(object):
    def __init__(self,data_in='data/merra2_cips_airs_timeSeries/zt_cips_3a+airs_merra2_loc_20181101-20190430_50N_zm.nc'):
        self.load_data(data_in)
    
    def load_data(self,data_in):
        ds              = xr.load_dataset(data_in)
        self.data_in    = data_in
        self.ds         = ds
        return ds

    def plot_figure(self,png_fpath='output.png',figsize=(16,8),**kwargs):

        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1,1,1)

        result  = self.plot_ax(ax,**kwargs)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax,vmin=-20,vmax=100,levels=11,cmap='jet',
                plot_cbar=True,cbar_pad=0.05,cbar_aspect=20,
                ylabel_fontdict={},**kwargs):
        fig     = ax.get_figure()

        ds      = self.ds
        dates   = [datetime.datetime.strptime(str(int(x)),'%Y%m%d') for x in ds['DATE']]
        sDate   = min(dates)
        eDate   = max(dates)


        # Plot MERRA-2 Zonal Winds
        zz  = np.array(ds['ZONAL_WIND'])

        xx  = dates
        yy  = np.nanmean(np.array(ds['GEOPOTENTIAL_HEIGHT']),axis=1)

        # Keep only finite values of height.
        tf  = np.isfinite(yy)
        yy  = yy[tf]
        zz  = zz[tf,:]

        cbar_pcoll  = ax.contourf(xx,yy,zz,levels=levels,vmin=vmin,vmax=vmax,cmap=cmap)
        cntr        = ax.contour(xx,yy,zz,levels=levels,colors='0.3')
        ax.set_xlabel('UTC Date')
        ax.set_ylabel('Geopot. Height [km]',fontdict=ylabel_fontdict)
        ax.grid(False)

        if plot_cbar:
            lbl     = 'MERRA-2 Zonal Wind (m/s) (50\N{DEGREE SIGN} N)'
            cbar    = fig.colorbar(cbar_pcoll,label=lbl,pad=cbar_pad,aspect=cbar_aspect)

        # Plot CIPS GW Variance
        ax1     = ax.twinx()

        airs_cips_lw = 4
        xx      = dates
        yy      = np.array(ds['AIRS_GW_VARIANCE'])
        lbl     = 'AIRS (30 km)'
        ax1.plot(xx,yy,color='black',lw=airs_cips_lw,zorder=100,label=lbl)

        xx      = dates
        yy      = np.array(ds['CIPS_GW_VARIANCE'])
        lbl     = 'CIPS (50 km)'
        ax1.plot(xx,yy,color='fuchsia',lw=airs_cips_lw,zorder=100,label=lbl)
        
        lbl     = 'CIPS (%$^{2}$) and AIRS (K$^{2}$)\nGW Variance'
        ax1.set_ylabel(lbl,fontdict=ylabel_fontdict)
        ax1.grid(False)
        ax1.set_ylim(0,0.25)

        ax1.legend(loc='upper right',ncols=2,fontsize='large')

        result  = {}
        result['cbar_pcoll']    = cbar_pcoll
        return result

if __name__ == '__main__':
    output_dir = os.path.join('output','merra2CipsAirsTimeSeries')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    png_fname   = 'merra2CipsAirsTimeSeries.png'
    png_fpath   = os.path.join(output_dir,png_fname)

    mca = Merra2CipsAirsTS()
    mca.plot_figure(png_fpath=png_fpath)
