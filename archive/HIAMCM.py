#!/usr/bin/env python
"""
This class will plot HIAMCM outputs saved to a netCDF file.
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

class HIAMCM(object):
    def __init__(self,data_in='data/HIAMCM/07DEC2018-31MAR2019.mzgw.grads.nc'):
        self.load_data(data_in)
    
    def load_data(self,data_in):
        ds  = xr.open_dataset(data_in) 
        self.data_in    = data_in
        self.ds         = ds

    def plot_figure(self,png_fpath='output.png',figsize=(16,8),**kwargs):

        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1,1,1)

        result  = self.plot_ax(ax,**kwargs)

        ax.set_title(result['title'])

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax,prm='ww',lats=(40.,60.),
                cmap='jet',plot_cbar=True,ylabel_fontdict={},**kwargs):

        ds      = self.ds

        fig     = ax.get_figure()
        
#        lat_inx = np.where(ds['lats'] == lat)[0][0]
        xx      = ds[prm]['dates']
        yy      = ds[prm]['alts']
#        zz      = ds[prm][:,lat_inx,:]
        tf      = np.logical_and(ds['lats'] >= min(lats), 
                                 ds['lats'] < max(lats))
        zz      = np.mean(ds[prm][:,tf,:],axis=1)
        cbar_pcoll  = ax.contourf(xx,yy,zz.T,cmap=cmap,**kwargs)

        ax.set_xlabel('Date')
        ax.set_ylabel('Altitude [km]',fontdict=ylabel_fontdict)

        ax.grid(False)

        cbar_label = ds[prm].attrs.get('desc',prm)
        if plot_cbar:
            fig.colorbar(cbar_pcoll,label=cbar_label)

        prm_title = ds[prm].attrs.get('title',prm)
        title = 'HIAMCM {:0.0f}\N{DEGREE SIGN} N - {:0.0f}\N{DEGREE SIGN} N Lat Average {!s}'.format(lats[0],lats[1],prm_title)

        result  = {}
        result['cbar_pcoll'] = cbar_pcoll
        result['cbar_label'] = cbar_label
        result['title']      = title
        result['zz']         = zz
        return result

if __name__ == '__main__':
    output_dir = os.path.join('output','HIAMCM')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    png_fname   = 'HIAMCM.png'
    png_fpath   = os.path.join(output_dir,png_fname)

    hiamcm  = HIAMCM()
    hiamcm.plot_figure(png_fpath=png_fpath)
    print(png_fpath)
