#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import string
letters = string.ascii_lowercase

def mpl_style():
    plt.rcParams['font.size']           = 18
    plt.rcParams['font.weight']         = 'bold'
    plt.rcParams['axes.titleweight']    = 'bold'
    plt.rcParams['axes.labelweight']    = 'bold'
    plt.rcParams['axes.xmargin']        = 0
    plt.rcParams['axes.titlesize']      = 'x-large'
mpl_style()

def my_xticks(sDate,eDate,ax,radar_ax=False,labels=True,short_labels=False,
                fmt='%d %b',fontdict=None,plot_axvline=True):
    if fontdict is None:
        fontdict = {'weight': 'bold', 'size':mpl.rcParams['ytick.labelsize']}
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

def fmt_xaxis(ax,xlim=None,label=True):
    ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))
    ax.set_xlabel('Time [UTC]')
    ax.set_xlim(xlim)

def adjust_axes(ax_0,ax_1):
    """
    Force geospace environment axes to line up with histogram
    axes even though it doesn't have a color bar.
    """
    ax_0_pos    = list(ax_0.get_position().bounds)
    ax_1_pos    = list(ax_1.get_position().bounds)
    ax_0_pos[2] = ax_1_pos[2]
    ax_0.set_position(ax_0_pos)

def curve_combo_plot(result_dct,cb_pad=0.125,
                     output_dir=os.path.join('output','daily_plots'),
                     auto_crit=None):
    """
    Make a curve combo stackplot that includes:
        1. Heatmap of Ham Radio Spots
        2. Raw Detected Edge
        3. Filtered, Windowed Edge
        4. Spectra of Edges

    Input:
        result_dct: Dictionary of results produced by run_edge_detect().
    """
    md              = result_dct.get('metaData')
    date            = md.get('date')
    xlim            = md.get('xlim')
    winlim          = md.get('winlim')
    fitWinLim       = md.get('fitWinLim')
    lstid_criteria  = md.get('lstid_criteria')

    arr             = result_dct.get('spotArr')
    edge_0          = result_dct.get('000_detectedEdge')
    sin_fit         = result_dct.get('sin_fit')
    poly_fit        = result_dct.get('poly_fit')
    p0_sin_fit      = result_dct.get('p0_sin_fit')
    p0_poly_fit     = result_dct.get('p0_poly_fit')
    stability       = result_dct.get('stability')
    data_detrend    = result_dct.get('data_detrend')

    ranges_km   = arr.coords['ranges_km']
    arr_times   = [pd.Timestamp(x) for x in arr.coords['datetimes'].values]

    nCols   = 1
    nRows   = 4

    axInx   = 0
    figsize = (18,nRows*6)

    fig     = plt.figure(figsize=figsize)
    axs     = []

    # Plot Heatmap #########################
    for plot_fit in [False, True]:
        axInx   = axInx + 1
        ax      = fig.add_subplot(nRows,nCols,axInx)
        axs.append(ax)

        mpbl = ax.pcolormesh(arr_times,ranges_km,arr,cmap='plasma')
        plt.colorbar(mpbl,aspect=10,pad=cb_pad,label='Scaled Amateur Radio Data')
        if not plot_fit:
            ax.set_title(f'| {date} |')
        else:
            ax.plot(arr_times,edge_0,lw=2,label='Detected Edge')

            if p0_sin_fit != {}:
                ax.plot(sin_fit.index,sin_fit+poly_fit,label='Sin Fit',color='white',lw=3,ls='--')

            ax2 = ax.twinx()
            ax2.plot(stability.index,stability,lw=2,color='0.5')
            ax2.grid(False)
            ax2.set_ylabel('Edge Coef. of Variation\n(Grey Line)')

            for wl in winlim:
                ax.axvline(wl,color='0.8',ls='--',lw=2)

            for wl in fitWinLim:
                ax.axvline(wl,color='lime',ls='--',lw=2)

            ax.legend(loc='upper center',fontsize='x-small',ncols=4)

        fmt_xaxis(ax,xlim)
        ax.set_ylabel('Range [km]')
        ax.set_ylim(1000,2000)

    # Plot Detrended and fit data. #########
    axInx   = axInx + 1
    ax      = fig.add_subplot(nRows,nCols,axInx)
    axs.append(ax)

    ax.plot(data_detrend.index,data_detrend,label='Detrended Edge')
    ax.plot(sin_fit.index,sin_fit,label='Sin Fit',color='red',lw=3,ls='--')

    for wl in fitWinLim:
        ax.axvline(wl,color='lime',ls='--',lw=2)
    

    ax.set_ylabel('Range [km]')
    fmt_xaxis(ax,xlim)
    ax.legend(loc='lower right',fontsize='x-small',ncols=4)

    # Print TID Info
    axInx   = axInx + 1
    ax      = fig.add_subplot(nRows,nCols,axInx)
    ax.grid(False)
    for xtl in ax.get_xticklabels():
        xtl.set_visible(False)
    for ytl in ax.get_yticklabels():
        ytl.set_visible(False)
    axs.append(ax)

    fontdict = {'weight':'normal','family':'monospace'}
    
    txt = []
    txt.append('2nd Deg Poly Fit')
    txt.append('(Used for Detrending)')
    for key, val in p0_poly_fit.items():
        if key == 'r2':
            txt.append('{!s}: {:0.2f}'.format(key,val))
        else:
            txt.append('{!s}: {:0.1f}'.format(key,val))
    ax.text(0.01,0.95,'\n'.join(txt),fontdict=fontdict,va='top')

    txt = []
    txt.append('Sinusoid Fit')
    for key, val in p0_sin_fit.items():
        if key == 'r2':
            txt.append('{!s}: {:0.2f}'.format(key,val))
        elif key == 'is_lstid':
            txt.append('{!s}: {!s}'.format(key,val))
        else:
            txt.append('{!s}: {:0.1f}'.format(key,val))
    ax.text(0.30,0.95,'\n'.join(txt),fontdict=fontdict,va='top')

    results = {}
    if auto_crit == True:
        txt = []
        txt.append('Automatic LSTID Classification\nCriteria from Sinusoid Fit')
        for key, val in lstid_criteria.items():
            txt.append('{!s} <= {!s} < {!s}'.format(val[0],key,val[1]))
        ax.text(0.01,0.3,'\n'.join(txt),fontdict=fontdict,va='top',bbox={'facecolor':'none','edgecolor':'black','pad':5})    
            
        results['sin_is_lstid']  = {'msg':'Auto', 'classification': p0_sin_fit.get('is_lstid')}

    fig.tight_layout()

    # Add panel labels.
    for ax_inx, ax in enumerate(axs):
        lbl = '({!s})'.format(letters[ax_inx])
        ax.set_title(lbl,loc='left')

    # Add meta data about data sources.
    meta = result_dct['metaData']
    meta_title = []
    freq_str = meta.get('freq_str')
    if freq_str is not None:
        meta_title.append(freq_str)

    region = meta.get('region')
    if region is not None:
        if region == 'NA':
            region = 'North America'
        meta_title.append(region)
    
    datasets = meta.get('datasets')
    if datasets is not None:
        meta_title.append('{!s}'.format(datasets))

    meta_title = '\n'.join(meta_title)
    axs[0].set_title(meta_title,loc='right',fontdict={'size':'x-small'})

    # Account for colorbars and line up all axes.
    for ax_inx, ax in enumerate(axs):
        if ax_inx == 0:
            continue
        adjust_axes(ax,axs[0])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    date_str    = date.strftime('%Y%m%d')
    png_fname   = f'{date_str}_curveCombo.png'
    png_fpath   = os.path.join(output_dir,png_fname)
    print('   Saving: {!s}'.format(png_fpath))
    fig.savefig(png_fpath,bbox_inches='tight')
    plt.close()

    result_dct['p0_sin_fit'] = p0_sin_fit
    return result_dct

def sin_fit_key_params_to_csv(all_results,output_dir='output'):
    """
    Generate a CSV with sin fit parameters for an entire season.
    """

    sDate   = min(all_results.keys())
    eDate   = max(all_results.keys())

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = eDate.strftime('%Y%m%d')

    # Create parameter dataframe.
    params = []
    params.append('T_hr')
    params.append('amplitude_km')
    params.append('is_lstid')
    params.append('agree')
    
    df_lst = []
    df_inx = []
    for date,results in all_results.items():
        if results is None:
            continue

        p0_sin_fit = results.get('p0_sin_fit')
        tmp = {}
        for param in params:
            tmp[param] = p0_sin_fit.get(param,np.nan)

        df_lst.append(tmp)
        df_inx.append(date)

    df          = pd.DataFrame(df_lst,index=df_inx)
    # Force amplitudes to be positive.
    df.loc[:,'amplitude_km']    = np.abs(df['amplitude_km'])

    # Set non-LSTID parameters to NaN
    csv_fname   = '{!s}-{!s}_sinFit.csv'.format(sDate_str,eDate_str)
    csv_fpath   = os.path.join(output_dir,csv_fname)
    df.to_csv(csv_fpath)

def plot_sin_fit_analysis(all_results,
                          T_hr_vmin=0,T_hr_vmax=5,T_hr_cmap='rainbow',
                          output_dir='output'):
    """
    Plot an analysis of the sin fits for the entire season.
    """

    sDate   = min(all_results.keys())
    eDate   = max(all_results.keys())

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = eDate.strftime('%Y%m%d')
    png_fname   = '{!s}-{!s}_sinFitAnalysis.png'.format(sDate_str,eDate_str)
    png_fpath   = os.path.join(output_dir,png_fname)

    # Create parameter dataframe.
    params = []
    params.append('T_hr')
    params.append('amplitude_km')
    params.append('phase_hr')
    params.append('offset_km')
    params.append('slope_kmph')
    params.append('r2')
    params.append('T_hr_guess')
    params.append('selected')

    df_lst = []
    df_inx = []
    for date,results in all_results.items():
        if results is None:
            continue

        all_sin_fits = results.get('all_sin_fits')
        for p0_sin_fit in all_sin_fits:
            tmp = {}
            for param in params:
                if param in ['selected']:
                    tmp[param] = p0_sin_fit.get(param,False)
                else:
                    tmp[param] = p0_sin_fit.get(param,np.nan)
            
            # Get the start and end times of the good fit period.
            fitWinLim   =  results['metaData']['fitWinLim']
            tmp['fitStart'] = fitWinLim[0]
            tmp['fitEnd']   = fitWinLim[1]

            df_lst.append(tmp)
            df_inx.append(date)

    df                = pd.DataFrame(df_lst,index = df_inx)
    # Calculate the duration in hours of the good fit period.
    df['duration_hr'] = (df['fitEnd'] - df['fitStart']).apply(lambda x: x.total_seconds()/3600.)
    df_sel            = df[df.selected].copy() # Data frame with fits that have been selected as good.

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = eDate.strftime('%Y%m%d')
    csv_fname   = '{!s}-{!s}_allSinFits.csv'.format(sDate_str,eDate_str)
    csv_fpath   = os.path.join(output_dir,csv_fname)
    df.to_csv(csv_fpath)

    csv_fname   = '{!s}-{!s}_selectedSinFits.csv'.format(sDate_str,eDate_str)
    csv_fpath   = os.path.join(output_dir,csv_fname)
    df_sel.to_csv(csv_fpath)

    # Plotting #############################
    nrows   = 4
    ncols   = 1
    ax_inx  = 0
    axs     = []

    cbar_info = {} # Keep track of colorbar info in a dictionary to plot at the end after fig.tight_layout() because of issues with cbar placement.

    figsize = (30,nrows*6.5)
    fig     = plt.figure(figsize=figsize)

    # ax with LSTID Amplitude Analysis #############################################
    prmds   = {}
    prmds['amplitude_km'] = prmd = {}
    prmd['title']   = 'Ham Radio TID Amplitude'
    prmd['label']   = 'Amplitude [km]'
    prmd['vmin']    = 10
    prmd['vmax']    = 60

    prmds['T_hr'] = prmd = {}
    prmd['title']   = 'Ham Radio TID Period'
    prmd['label']   = 'Period [hr]'
    prmd['vmin']    = 0
    prmd['vmax']    = 5

    prmds['r2'] = prmd = {}
    prmd['title']   = 'Ham Radio Fit $r^2$'
    prmd['label']   = '$r^2$'
    prmd['vmin']    = 0
    prmd['vmax']    = 1

    for param in ['amplitude_km','T_hr','r2']:
        prmd            = prmds.get(param)
        title           = prmd.get('title',param)
        label           = prmd.get('label',param)

        ax_inx  += 1
        ax              = fig.add_subplot(nrows,ncols,ax_inx)
        axs.append(ax)

        xx              = df_sel.index
        yy_raw          = df_sel[param]
        rolling_days    = 5
        title           = '{!s} ({!s} Day Rolling Mean)'.format(title,rolling_days)
        yy              = df_sel[param].rolling(rolling_days,center=True).mean()

        vmin            = prmd.get('vmin',np.nanmin(yy))
        vmax            = prmd.get('vmax',np.nanmax(yy))

        cmap            = mpl.colormaps.get_cmap(T_hr_cmap)
        norm            = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        mpbl            = mpl.cm.ScalarMappable(norm,cmap)
        color           = mpbl.to_rgba(yy)
        ax.plot(xx,yy_raw,color='0.5',label='Raw Data')
        ax.plot(xx,yy,color='blue',lw=3,label='{!s} Day Rolling Mean'.format(rolling_days))
        ax.scatter(xx,yy,marker='o',c=color)
        ax.legend(loc='upper right',ncols=2)

        trans           = mpl.transforms.blended_transform_factory( ax.transData, ax.transAxes)
        ax.bar(xx,1,width=1,color=color,align='edge',zorder=-1,transform=trans,alpha=0.5)

        cbar_info[ax_inx] = cbd = {}
        cbd['ax']       = ax
        cbd['label']    = label
        cbd['mpbl']     = mpbl

        ylabel_fontdict         = {'weight': 'bold', 'size':24}
        ax.set_ylabel(label,fontdict=ylabel_fontdict)
        my_xticks(sDate,eDate,ax,fmt='%d %b')
        ltr = '({!s}) '.format(letters[ax_inx-1])
        ax.set_title(ltr+title, loc='left')

    # ax with LSTID T_hr Fitting Analysis ##########################################    
    ax_inx  += 1
    ax      = fig.add_subplot(nrows,ncols,ax_inx)
    axs.append(ax)
    ax_0    = ax


    xx      = df.index
    yy      = df.T_hr
    color   = df.T_hr_guess
    r2      = df.r2.values
    r2[r2 < 0]  = 0
    alpha   = r2
    mpbl    = ax.scatter(xx,yy,c=color,alpha=alpha,marker='o',
                         vmin=T_hr_vmin,vmax=T_hr_vmax,cmap=T_hr_cmap)

    ax.scatter(df_sel.index,df_sel.T_hr,c=df_sel.T_hr_guess,ec='black',
                         marker='o',label='Selected Fit',
                         vmin=T_hr_vmin,vmax=T_hr_vmax,cmap=T_hr_cmap)
    cbar_info[ax_inx] = cbd = {}
    cbd['ax']       = ax
    cbd['label']    = 'T_hr Guess'
    cbd['mpbl']     = mpbl

    ax.legend(loc='upper right')
    ax.set_ylim(0,10)
    ax.set_ylabel('T_hr Fit')
    my_xticks(sDate,eDate,ax,labels=(ax_inx==nrows))

    fig.tight_layout()

#    # Account for colorbars and line up all axes.
    for ax_inx, cbd in cbar_info.items():
        ax_pos      = cbd['ax'].get_position()
                    # [left, bottom,       width, height]
        cbar_pos    = [1.025,  ax_pos.p0[1], 0.02,   ax_pos.height] 
        cax         = fig.add_axes(cbar_pos)
        fig.colorbar(cbd['mpbl'],label=cbd['label'],cax=cax)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print('   Saving: {!s}'.format(png_fpath))
    fig.savefig(png_fpath,bbox_inches='tight')
