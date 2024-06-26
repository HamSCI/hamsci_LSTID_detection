import os
import shutil
import warnings
import pickle
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import xarray as xr
import joblib
import math
import datetime
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from data_loading import create_xarr, mad, create_label_df
from utils import DateIter

import yaml

import bokeh
from bokeh.layouts import layout, column
from bokeh.models import Div, RangeSlider, Spinner, ColumnDataSource, Slider
from bokeh.plotting import figure, show, curdoc
from bokeh.transform import linear_cmap
from bokeh.io import show, output_notebook
from bokeh.themes import Theme
from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature

plt.rcParams['font.size']           = 18
plt.rcParams['font.weight']         = 'bold'
plt.rcParams['axes.titleweight']    = 'bold'
plt.rcParams['axes.labelweight']    = 'bold'
plt.rcParams['axes.xmargin']        = 0

class JobLibLoader(object):
    def __init__(self):
        self.date_iter = None

    def _load_data(self):
        ################################################################################
        # Load in CSV Histograms #######################################################
        parent_dir     = 'data_files'
        data_out_path  = 'processed_data/full_data.joblib'
        if not os.path.exists(data_out_path):
            full_xarr = create_xarr(
                parent_dir=parent_dir,
                expected_shape=(720, 300),
                dtype=(np.uint16, np.float32),
                apply_fn=mad,
                plot=False,
            )
            joblib.dump(full_xarr, data_out_path)
        date_iter       = DateIter(data_out_path) #, label_df=label_out_path)
        self.date_iter  = date_iter

jll = JobLibLoader()

def fmt_xaxis(ax,xlim=None,label=True):
    ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))
    ax.set_xlabel('Time [UTC]')
    ax.set_xlim(xlim)

def plot_heatmap(date,times,ranges_km,arr,xlim=None,cb_pad=0.04):
    # Plot Heatmap #########################
    fig     = plt.figure(figsize=(18,5))
    ax      = fig.add_subplot(1,1,1)
    ax.set_title(f'| {date} |')
    mpbl = ax.pcolormesh(times,ranges_km,arr,cmap='plasma')
    plt.colorbar(mpbl,label='Radio Spots',aspect=10,pad=cb_pad)
    fmt_xaxis(ax,xlim)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(500,2000)
    plt.show()
    plt.close()

def load_spots(date, x_trim=.08333, y_trim=.08,sigma=4.2):
    if jll.date_iter is None:
        jll._load_data()

    arr = jll.date_iter.get_date(date,raise_missing=False)
    if arr is None:
        warnings.warn(f'Date {date} has no input')
        return
        
    xl_trim, xrt_trim   = x_trim if isinstance(x_trim, (tuple, list)) else (x_trim, x_trim)
    yl_trim, yr_trim    = x_trim if isinstance(y_trim, (tuple, list)) else (y_trim, y_trim)
    xrt, xl             = math.floor(xl_trim * arr.shape[0]), math.floor(xrt_trim * arr.shape[0])
    yr, yl              = math.floor(yl_trim * arr.shape[1]), math.floor(yr_trim * arr.shape[1])
    arr                 = arr[xrt:-xl, yr:-yl]

    ranges_km   = arr.coords['height'].values
    times       = [date + x for x in pd.to_timedelta(arr.coords['time'])]
    arr         = np.nan_to_num(arr, nan=0)
    arr         = gaussian_filter(arr.T, sigma=(sigma, sigma))

    # Plotting Code ################################################################ 
    # X-Limits for plotting
    x_0     = date + datetime.timedelta(hours=12)
    x_1     = date + datetime.timedelta(hours=24)
    xlim    = (x_0, x_1)

    result = {}
    result['date']      = date
    result['arr']       = arr
    result['times']     = times
    result['ranges_km'] = ranges_km
    result['xlim']      = xlim    
    return result

def my_sin2(T_hr=3, amplitude=200, phase=0, offset=1400.):
    # create the function we want to fit
    freq   = 1./(datetime.timedelta(hours=T_hr).total_seconds())
    result = amplitude * np.sin( (2*np.pi*tt_sec*freq )+ phase ) + offset
    data   = pd.DataFrame({'curve':result},index=times)
    data.index.name = 'time'
    return data

class SinFit(object):
    def __init__(self,times,T_hr=3,amplitude=200,phase=0,offset=1400.):
        t0          = min(times)
        tt_sec      = np.array([(x-t0).total_seconds() for x in times])

        self.times      = times
        self.tt_sec     = tt_sec

        p0  = {}
        p0['T_hr']      = T_hr
        p0['amplitude'] = amplitude
        p0['phase']     = phase
        p0['offset']    = offset
        self.params     = p0

    def sin(self,**kwArgs):
        times       = self.times
        tt_sec      = self.tt_sec

        self.params.update(kwArgs)
        p0          = self.params
        T_hr        = p0['T_hr']
        amplitude   = p0['amplitude']
        phase       = p0['phase']
        offset      = p0['offset']

        freq        = 1./(datetime.timedelta(hours=T_hr).total_seconds())
        result      = amplitude * np.sin( (2*np.pi*tt_sec*freq )+ phase ) + offset
        data        = pd.DataFrame({'curve':result},index=times)
        data.index.name = 'time'
        return data

class BkApp(object):
    def __init__(self,result):
        self.result = result

    def bkapp(self,doc):
        result      = self.result
        date        = result['date']
        times       = result['times']
        ranges_km   = result['ranges_km']
        image       = result['arr']
        
        x_range   = result['xlim']
        y_range   = (min(ranges_km), max(ranges_km))
        title     = date.strftime('%Y %b %d')
        
        plot      = figure(x_range=x_range, y_range=y_range, title=title)
        
        yy        = min(ranges_km)
        dh        = np.ptp(ranges_km)
        xx        = min(times)
        dw        = max(times) - min(times)
        color_mapper = bokeh.models.LinearColorMapper(palette="Viridis256", low=0, high=10)
        r         = plot.image(image=[image], color_mapper=color_mapper,dh=dh, dw=dw, x=xx, y=yy)

        color_bar = r.construct_color_bar(padding=1)
        plot.add_layout(color_bar, "right")

        sin_fit     = SinFit(times)
        data        = sin_fit.sin()
        source      = ColumnDataSource(data=data)
        plot.line('time','curve',source=source,line_color='white',line_width=2,line_dash='dashed')
        
        def callback(attr, old, new):
            source_new = sin_fit.sin(T_hr=new)
            source.data = ColumnDataSource.from_df(source_new)

        slider = Slider(start=0.1, end=10, value=sin_fit.params['T_hr'], step=0.1, title="Period [hr]:")
        slider.on_change('value', callback)

        doc.add_root(column(slider, plot))
        
        doc.theme = Theme(json=yaml.load("""
            attrs:
                figure:
                    # background_fill_color: "#DDDDDD"
                    outline_line_color: white
                    toolbar_location: above
                    height: 500
                    width: 1000
                Grid:
                    grid_line_dash: [6, 4]
                    # grid_line_color: white
        """, Loader=yaml.FullLoader))

