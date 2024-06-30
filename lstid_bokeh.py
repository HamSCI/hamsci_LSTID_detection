import os
import shutil
from functools import partial
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
from bokeh.models import DatetimeRangeSlider, Div, RangeSlider, Spinner, ColumnDataSource, Slider
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
    def __init__(self,cache_dir='bokeh_cache',clear_cache=False):
        if clear_cache and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.date_iter  = None
        self.cache_dir  = cache_dir

        self.sDate      = datetime.datetime(2018,11,1)
        self.eDate      = datetime.datetime(2019,4,30)

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

    def load_spots(self, date, x_trim=.08333, y_trim=.08,sigma=4.2):
        date_str  = date.strftime('%Y%m%d')
        pkl_fname = '{!s}_spotArray.pkl'.format(date_str)
        pkl_fpath = os.path.join(self.cache_dir,pkl_fname)

        if not os.path.exists(pkl_fpath):
            if self.date_iter is None:
                self._load_data()

            arr = self.date_iter.get_date(date,raise_missing=False)
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

            print('SAVING: {!s}'.format(pkl_fpath))
            with open(pkl_fpath,'wb') as fl:
                pickle.dump(result,fl)
        else:
            print('LOADING: {!s}'.format(pkl_fpath))
            with open(pkl_fpath,'rb') as fl:
                result = pickle.load(fl)

        return result

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

def my_sin2(T_hr=3, amplitude=200, phase=0, offset=1400.):
    # create the function we want to fit
    freq   = 1./(datetime.timedelta(hours=T_hr).total_seconds())
    result = amplitude * np.sin( (2*np.pi*tt_sec*freq )+ phase ) + offset
    data   = pd.DataFrame({'curve':result},index=times)
    data.index.name = 'time'
    return data

class SinFit(object):
    def __init__(self,times,fig,
                 T_hr=3,amplitude_km=200,phase_hr=0,offset_km=1400.,
                 slope_kmph=0,pivot_hr=0,
                 sTime=None,eTime=None):

        # Define time arrays and parameters.
        t0          = min(times)
        tt_sec      = np.array([(x-t0).total_seconds() for x in times])

        if sTime is None:
            sTime = min(times)

        if eTime is None:
            eTime = max(times)

        self.times      = np.array(times)
        self.tt_sec     = tt_sec
        self.t0         = t0

        # Define initial sinusoid parameters.
        p0  = {}
        p0['T_hr']          = T_hr
        p0['amplitude_km']  = amplitude_km
        p0['phase_hr']      = phase_hr
        p0['offset_km']     = offset_km
        p0['pivot_hr']      = pivot_hr
        p0['slope_kmph']    = slope_kmph
        p0['sTime']         = sTime
        p0['eTime']         = eTime
        self.params         = p0

        # Calculate initial sinusoid.
        data                = self.sin()
        source              = ColumnDataSource(data=data)
        self.source         = source

        # Plot sinusoid on figure.
        line                = fig.line('time','curve',source=source,line_color='white',line_width=2,line_dash='dashed')
        self.fig            = fig
        self.line           = line

        # Define sliders to adjust sinusoid parameters.
        slider_amplitude_km = Slider(start=0, end=3000, value=self.params['amplitude_km'],
                                     step=10, title="Amplitude [km]",sizing_mode='stretch_both')
        slider_amplitude_km.on_change('value', partial(self.cb_slider,param='amplitude_km'))

        slider_period = Slider(start=0.1, end=10, value=self.params['T_hr'],
                               step=0.1, title="Period [hr]",sizing_mode='stretch_both')
        slider_period.on_change('value', partial(self.cb_slider,param='T_hr'))

        slider_phase_hr = Slider(start=-10, end=10, value=self.params['phase_hr'],
                                 step=0.1, title="Phase [hr]",sizing_mode='stretch_both')
        slider_phase_hr.on_change('value', partial(self.cb_slider,param='phase_hr'))

        slider_offset_km = Slider(start=0, end=3000, value=self.params['offset_km'],
                                  step=10, title="Offset [km]",sizing_mode='stretch_both')
        slider_offset_km.on_change('value', partial(self.cb_slider,param='offset_km'))

        slider_slope_kmph = Slider(start=-1000, end=1000, value=self.params['slope_kmph'],
                                   step=10, title="Slope [km/hr]",sizing_mode='stretch_both')
        slider_slope_kmph.on_change('value', partial(self.cb_slider,param='slope_kmph'))

        slider_pivot_hr = Slider(start=-10, end=10, value=self.params['pivot_hr'],
                                 step=0.1, title="Pivot [hr]",sizing_mode='stretch_both')
        slider_pivot_hr.on_change('value', partial(self.cb_slider,param='pivot_hr'))

        slider_dtRange = DatetimeRangeSlider(start=min(self.times), end=max(self.times),
                            format = '%H:%M',
                            value=(self.params['sTime'],self.params['eTime']),
                             step=(60*1000), title="Datetime Range",sizing_mode='stretch_both')
        slider_dtRange.on_change('value', self.cb_dtRange)

        # Put all sliders into a Bokeh column layout.
        col_objs    = []
        col_objs.append(slider_amplitude_km)
        col_objs.append(slider_period)
        col_objs.append(slider_phase_hr)
        col_objs.append(slider_offset_km)
        col_objs.append(slider_slope_kmph)
        col_objs.append(slider_pivot_hr)
        col_objs.append(slider_dtRange)
        self.widgets = bokeh.layouts.column(*col_objs, sizing_mode="fixed", height=400, width=250)

    def cb_slider(self,attr,old,new,param):
        """
        Callback function for sliders.
        """
        source_new          = self.sin(**{param:new})
        self.source.data    = ColumnDataSource.from_df(source_new)

    def cb_dtRange(self,attr,old,new):
        """
        Callback function for datetime range slider.
        """
        sTime               = datetime.datetime.utcfromtimestamp(new[0]/1000.)
        eTime               = datetime.datetime.utcfromtimestamp(new[1]/1000.)
        source_new          = self.sin(sTime=sTime,eTime=eTime)
        self.source.data    = ColumnDataSource.from_df(source_new)

    def sin(self,**kwArgs):
        """
        Calculate a sinusoid using the parameters stored in self.params.

        Any self.params item can be updated by passing a keyword argument.
        """
        times       = self.times
        tt_sec      = self.tt_sec

        self.params.update(kwArgs)
        p0              = self.params
        T_hr            = p0['T_hr']
        amplitude_km    = p0['amplitude_km']
        phase_hr        = p0['phase_hr']
        offset_km       = p0['offset_km']
        pivot_hr        = p0['pivot_hr']
        slope_kmph      = p0['slope_kmph']
        sTime           = p0['sTime']
        eTime           = p0['eTime']

        phase_rad   = (2.*np.pi) * (phase_hr / T_hr) 
        freq        = 1./(datetime.timedelta(hours=T_hr).total_seconds())
        result      = amplitude_km * np.sin( (2*np.pi*tt_sec*freq ) + phase_rad ) +  offset_km

        if slope_kmph != 0:
            result      += (slope_kmph/3600.)*(tt_sec + pivot_hr*3600.)

        tf = np.logical_and(times >= sTime, times <= eTime)
        if np.count_nonzero(~tf) > 0:
            result[~tf] = np.nan

        data        = pd.DataFrame({'curve':result},index=times)
        data.index.name = 'time'
        return data

class SpotHeatMap(object):
    def __init__(self,date=None,jll=None):
        if jll is None:
            jll = JobLibLoader()
        self.jll    = jll

        if date is None:
            date    = jll.sDate

        fig         = figure()
        self.fig    = fig

        self.update_heatmap(date)

    def update_heatmap(self,date):
        fig         = self.fig
        if hasattr(self,'img'):
            fig.renderers.remove(self.img)

        data        = self.jll.load_spots(date)
        self.data   = data

        times       = data['times']
        ranges_km   = data['ranges_km']
        image       = data['arr']

        xx          = min(times)
        dw          = max(times) - min(times)
        yy          = min(ranges_km)
        dh          = np.ptp(ranges_km)
        if not hasattr(self,'cmapper'):
            self.cmapper     = bokeh.models.LinearColorMapper(palette="Viridis256", low=0, high=10)
        self.img    = fig.image(image=[image], color_mapper=self.cmapper,dh=dh, dw=dw, x=xx, y=yy)

        if not hasattr(self,'color_bar'):
            self.color_bar   = self.img.construct_color_bar(padding=1)
            fig.add_layout(self.color_bar, "right")

        fig.x_range.start   = data['xlim'][0]
        fig.x_range.end     = data['xlim'][1]
        fig.y_range.start   = min(ranges_km)
        fig.y_range.end     = max(ranges_km)
        fig.title.text      = date.strftime('%Y %b %d')
        fig.xaxis.formatter = bokeh.models.DatetimeTickFormatter()

    def date_picker(self):
        date_picker = bokeh.models.DatePicker(value=self.jll.sDate.date(), min_date=self.jll.sDate.date(), max_date=self.jll.eDate.date())
        date_picker.on_change('value', self.cb_date_picker)
        return date_picker

    def cb_date_picker(self,attr,old,new):
        """
        Callback function for the date picker.
        """
        date    = datetime.datetime.fromisoformat(new)
        self.update_heatmap(date)

class BkApp(object):
    def __init__(self,jll=None):
        self.jll = jll

    def bkapp(self,doc):
        shp     = SpotHeatMap(jll=self.jll)
        sin_fit = SinFit(shp.data['times'],shp.fig)

        header = []
        header.append(bokeh.models.Button(label="Back", button_type="success"))
        date_picker = shp.date_picker()
        header.append(date_picker)
        header.append(bokeh.models.Button(label="Forward", button_type="success"))
        header  = bokeh.layouts.row(*header)

        row     = bokeh.layouts.row(sin_fit.widgets,shp.fig,height=1000)
        lyt     = bokeh.layouts.column(header,row,sizing_mode='stretch_both')
        doc.add_root(lyt)
        
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

