# Curve edge autoplot
# Authors:   W. Engelke, N. Callahan, University of Alabama
# Enhancements from N. Frissell, University of Scranton
# Dec. 2023
# Reads in a csv file containing spot counts for a given band/day/continent
# Outputs heat map of data, calculates/plots lower edge (Minimum Useful Range)
# using gradient method plus Butterworth filter or Lowess smoothing
# Will do a single csv file or process en entire directory of csv files

# Version changes
# V8 - expand  focus area to include >= 512 minutes

import datetime
from   datetime import datetime,timedelta, timezone
import pytz, math
import numpy as np
import pandas as pd

from   scipy.interpolate import interp1d
from   scipy.signal import filtfilt, butter, lfilter
from   scipy import signal as ss

from   PIL import Image, ImageColor, ImageFont, ImageDraw

import matplotlib as mpl
from   mpl_toolkits.axes_grid1 import make_axes_locatable
from   matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import glob
import traceback
import os, sys
import csv
import bottleneck as bn
import seaborn as sns
import statsmodels.api as sm

# set some defaults

showPlots = False # set this to False to suppress showing of plots
showfinal = True
FirstLine = True # to cause a header line to be included in the summary.csv file
first_pass = True
use_Butterworth = True
Windows_platform = False

conti = "NA"   # North America
theBand = '20' # band in meters

last_month = ''
last_year = ''

inputDirectory = ''  # Set this later, based on platform

# to save the generated plots, set this to True and set save directory
savePlots = True
saveDirectory = ''
FirstLine = True
saveSummary = True # if True, save summary file at end of processing
typeList = 'T0'
summaryResults = ([])


# Matplotlib settings to make the plots look a little nicer.
plt.rcParams['font.size']      = 10
#plt.rcParams['font.weight']    = 'bold'
plt.rcParams['axes.grid']      = True
plt.rcParams['axes.xmargin']   = 0
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['figure.figsize'] = (10,20)
#plt.rcParams['figure.constrained_layout.use'] = True   # makes plot overlap colorbar
plt.rcParams['axes.grid'] = False

def gradient_edge(img):
    new_img = np.gradient(img, axis=1)
    move_img = bn.move_mean(new_img, 35, axis=1)
    move_img = np.nan_to_num(move_img, nan=0.0)
    edge = np.argmax(move_img, axis=1)
    return edge

def lowess_smooth(edge, window_size=40, x=None):
    if x is None:
        x = np.linspace(0, len(edge), len(edge))
    smooth_edge = sm.nonparametric.lowess(edge, x, frac=window_size/len(arr))[:,1]    
    return smooth_edge

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# calculation helpers
# 
def butter_smooth(input_signal):
    # from Bill's notes:
    # Produce smoothed curve using Butterworth filter
    # f_break is # half-cycles in a minute; 
    # a 60-min. period wave has (1/60) cycles/min, or
    # (1/120) half-cycles per min. Sampling period is 1 min. 
    # See doc.: scipy.signal.butter
    f_break= 2*(1/120) # 2 times no. of half-cycles per sample period (1 min.)
    f_order = 5
    numer_poly, denom_poly = ss.butter(f_order, f_break, 
                                       analog=False, btype='lowpass', fs=1.)
    smoothed = ss.filtfilt(numer_poly, denom_poly, input_signal)     
    return smoothed

def lowess_smooth(input_signal, win=40):
    x = np.arange(len(input_signal))
    lowess = sm.nonparametric.lowess
    smooth_edge = lowess(input_signal, x, frac=win/len(arr))[:,1]    
    return smooth_edge

def rolling_mean(signal, win=35):
    # using pandas for quick-n-clean, hacks available
    # main goal with this impl. was to get the mean associated to center
    tmp_df = pd.DataFrame(signal)
    print("pd.DataFrame in rolling mean=",tmp_df)
    means = tmp_df.rolling(win, center=True).mean().to_numpy()
    print("means:",means)
    hw = win//2
    means[:hw] = 0.0
    means[-hw:] = 0.0
    return means


def plot_raw(arr, title='', xlabel='Time (min)', ylabel='Height (km)', cblabel='',ax=None):
    """ Simple way to display the image in matplotlib """
    arr = arr.transpose()[::-1,:]
    if ax is None:
        
        plt.xlim((720,1440))
        plt.title(title)
        sns.heatmap(arr, cmap='jet', robust=True, cbar=False) # cmaps was magma
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim((720,1440))

       
        y_locs, y_texts = plt.yticks()
        yticks = np.asarray([float(str(a.get_text())) for a in y_texts[::2]])
        plt.yticks(300 - y_locs[::2], yticks / 100)

        x_locs, x_texts = plt.xticks()
        plt.xticks(x_locs[::2], x_texts[::2])
        plt.xlim((720,1440))
        if showPlots:
            plt.show()
    else:
        
        ax.set_title(title,size=10)
        ax.set_xlim((720,1440))
        sns.heatmap(arr, cmap='jet', robust=True, cbar=True, ax=ax, cbar_kws={'label': cblabel})
        ax.set_yticks(np.arange(0,400,100), labels=['300','200','100','0'])
        ax.set_ylabel("Range (10 km)")
        ax.set_xticks(np.arange(0,1441, 120), labels=['00:00','02:00','04:00','06:00','08:00','10:00','12:00' , \
                                                '14:00','16:00','18:00','20:00','22:00','24:00' ], rotation=0)
        ax.set_xlabel('Time (UTC)')
        ax.vlines(x=[840,1352],color='magenta',ymin=0,ymax=300,linestyles='dashed',zorder=10)
    return

def output_sums(fd):
    global FirstLine, last_month, last_year, Sxx_1_max_month_total, Sxx_1_int_month_total
    fd.write("," + last_year + "-" + last_month + "," + '{0:1.1f}'.format(Sxx_1_max_month_total)+','+'{0:1.1f}'.format(Sxx_1_int_month_total))

def plot_images_on_date(img,theDate):
    global FirstLine, last_month, last_year, Sxx_1_max_month_total, Sxx_1_int_month_total
    
 # 
    fig = plt.figure()
    #                 (nrows, ncols),(loc (row, col))
    ax1 = plt.subplot2grid((4, 36), (0, 0), colspan=36)
    ax2 = plt.subplot2grid((4, 36), (1, 0), colspan=36)
    ax3 = plt.subplot2grid((4, 36), (2, 5), colspan=20)
    ax4 = plt.subplot2grid((4, 36), (3, 5), colspan=25)
    
    plot_raw(img, title='1. Initial Data - ' + theDate,ax=ax1,cblabel='spots in minute') # FIRST PLOT, raw data
    new_img = np.gradient(img, axis=1)
    ax1.set_xlim((720,1440))

# the following code does nothing
  #  pos1 = ax3.get_position() # try to make 3rd plot less wide
  #  print("pos1:",pos1)
  #  l, b, w, h = ax3.get_position().bounds
  #  print(l,b,w,h)
  #  ax3.set_position([l,b,0.7*w,h])  # nah.
    

    move_img = bn.move_mean(new_img, 35, axis=1)
    move_img = np.nan_to_num(move_img, nan=0.0)

    # SECOND PLOT, gradient
    plot_raw(move_img, title='2. Columnwise Gradient Moving Average (.5km) -'+ theDate,ax=ax2,cblabel='spots gradient')
    ax2.set_xlim((720,1440))
    
    assert np.isfinite(move_img.ravel()).all()

    noisy_arr = np.argmax(move_img, axis=1)
    ax3.scatter(np.arange(0, noisy_arr.shape[0], 1), noisy_arr, label='Gradient Points')
    
    # THIRD PLOT, scatter plot of derived edge & filtered line

    if use_Butterworth == False:

   # produce smoothed curve using Lowess algorithm
        smooth_arr = lowess_smooth(noisy_arr)
        ax3.plot(smooth_arr, color='red', label='Smoothed')
        ax3.set_title('Argmax Averaged, Lowess smoothing')

    else:

    # Produce smoothed curve using Butterworth filter
    # FILTERBREAK is # half-cycles in a minute; a 60-min. period wave has (1/60) cycles/min, or
    # (1/120) half-cycles per min. Sampling period is 1 min. See doc.: scipy.signal.butter
        FILTERBREAK = (1./120.)*2. # 2 times no. of half-cycles per sample period (1 min.)
        FILTERORDER = 5

    # for bandpass filter: these are frequency, expressed as 1/(2 * period) in minutes
    # so a 4-hour LSTID has period of 240 mintues; 3.5 hour LSTID has period of 210 minutes
        lowcut = (1./420.)*2.
        highcut=  (1./120.)*2.  # highcut same as above
        
     #   b, a = butter(FILTERORDER, FILTERBREAK, analog=False, btype='lowpass', fs=1.)
        b, a = butter(FILTERORDER, [lowcut, highcut], analog=False, btype='bandpass', fs=1.)
        smooth_arr = filtfilt(b, a, noisy_arr) # plotedge is the filtered signal
        print('1',smooth_arr)
        smooth_arr = smooth_arr + 80 # correct because bandpass removes the DC component
        print('2',smooth_arr)

        # frequency test - overrride with synth data

        # COMMENT THIS SECTION OUT FOR NORMAL ANALYSIS OF DATA

        period1 = 1.0 
        freq1 = 1.0 / (period1 * 60.)
     #   freq2 = 1.0 / (period2 * 60.) 
        amplitude = 100.  # in 10s of km # original was 50
        mean = 150.        # middle of plot

   # Following replaces actual data with synthetic sine wave to test FFT
   # Comment this section out to run normal job
     #   smooth_arr.fill(150)
     #   for x in range(1020, 1200): # was 720 1439
     #       smooth_arr[x] = amplitude * 0.5 * math.sin(2.0 * math.pi * freq1 * x) + mean


     
        
        ax3.plot(smooth_arr, color='red', label='Smoothed')
        print("Smoothed array:",smooth_arr)
        
        ax3.set_title('3. Argmax Averaged, Butterworth filter -'+ theDate,size=10)
        ax1.plot(300-smooth_arr, color='red', label='Smoothed') # put this on axis #1 
        ax2.plot(300-smooth_arr, color='red', label='Smoothed') # put this on axis #2
       
    ax3.set_ylabel('Range (10 km)')
    ax3.set_xlabel('Time (UTC)')
   
    ax3.set_ylim([0,300])
  #  ax3.set_xlim((900,1320))
    ax3.set_xlim((840,1353))
    ax3.set_xticks([840,900,960,1020,1080,1140,1200,1260,1320,1353])
    ax3.set_xticklabels(['14:00','15:00','16:00','17:00','18:00','19:00', \
                        '20:00','21:00','22:00','22:53'], rotation=45)
  #  ax3.set_xticks(np.arange(840,1353,57), labels=['14:00','15:00','16:00', \
   #                             '17:00','18:00','19:00','20:00','21:00','22:00' ], rotation=0)
    ax3.vlines(x=[840,1353],color='magenta',ymin=0,ymax=300,linestyles='dashed',lw=4, zorder=10)
    ax3.legend()
    outputFilename = theDate + "_" + conti + "_" + theBand + "m.png"
    outputCurve = theDate + "_" + conti + "_" + theBand + "m.csv"
  

    # FFT
    fs = 1./60.   # sampling frequency in Hz (one sample per minute)

   # smooth_arr_1 = smooth_arr[900:1319] # work only with the focus area, 900 to 1320 minutes
    smooth_arr_1 = smooth_arr[840: 1353] # V8 update, 14:00 to 22:30 UTC

 #   f, t, Sxx = ss.spectrogram(smooth_arr_1, fs, nperseg = 190,noverlap= 180, window=('tukey',0.1) )
 #   f_2, t_2, Sxx_2 = ss.spectrogram(smooth_arr_1, fs, nperseg = 512,noverlap= 1, window=('tukey',0.1) )

    f, t, Sxx = ss.spectrogram(smooth_arr_1, fs, nperseg = 128,noverlap= 64, window=('tukey',0.1) )
    
  #  f, t, Sxx = ss.spectrogram(smooth_arr_1, fs, nperseg = 512,noverlap= 500, window=('tukey',0.1) )
    f_2, t_2, Sxx_2 = ss.spectrogram(smooth_arr_1, fs, nperseg = 512,noverlap= 1, window=('tukey',0.1) )

    print("len f",len(f))
    print("len t",len(t))
    print("len Sxx",len(Sxx))
    print("len Sxx[0])",len(Sxx[0]))
    for i in range(1,10):
        print( '{:6.6f}'.format(f[i]), ' {:6.2f}'.format(1/(f[i]*3600)), str(' {:>6.2f}'.format(Sxx[i][0]).rjust(12)))

    
    print("len f_2",len(f_2))
    print("len t_2",len(t_2))
    print("len Sxx_2",len(Sxx_2))
    print("len Sxx_2[0])",len(Sxx_2[0]))
    for i in range(1,10):
        print( '{:6.6f}'.format(f_2[i]), ' {:6.2f}'.format(1/(f_2[i]*3600)), str(' {:>6.2f}'.format(Sxx_2[i][0]).rjust(12)))
    #print("halt")
    #z=input()

   # print("period in hours",1/(f * 3600))
    #print("freq.",f)
    Sxx_1 = Sxx[0:32, 0:128]
    noise_measure = np.std(Sxx_1)
    Sxx_max = np.max(Sxx_1)
    Sxx_tot = np.sum(Sxx_1)
    F1sum = np.sum(Sxx[1])
    F2sum = np.sum(Sxx[2])
    F3sum = np.sum(Sxx[3])

#  pcolormesh parameters:
#  X, Y,  C     --- X is columns & Y is rows;  C is in (rows,cols) (!)

# nearest, auto , gouraud

    # discard all the higher frequencies, no info there (was filtered out)
   # ax4.vlines(x=[5400,18000],color='magenta',ymin=0,ymax=300,linestyles='dashed',lw=4, zorder=10)
    print("t:",t)
    print("f:",f[0:8])
    print("Sxx",Sxx[0:8])
    mpbl = ax4.pcolormesh(t, f[0:8], Sxx[0:8], shading='auto') # auto
 #   mpbl = ax4.pcolormesh(t_2, f_2[0:8], Sxx_2[0:8], shading='auto') # auto

    ax4.set_xlabel("Time(UTC)")
    ax4.set_ylabel('Period (hours)')

    ax4.set_title("4. Power Spectral Density - " + theDate +"\n Max="+ '{0:1,.0f}'.format(Sxx_max) + " Tot. Power="   \
                  + '{0:1,.0f}'.format(Sxx_tot) + " Noise=" + '{0:1,.0f}'.format(noise_measure),size=8)
    
   # ax4.set_xticks(np.arange(6000,18000, 1714), labels=['15:00','16:00', \
    #                            '17:00','18:00','19:00','20:00','21:00','22:00'], rotation=0)
 #   ax4.set_xticks(np.arange(6000,24000,int((24000-6000)/11)), labels=['14:00','15:00','16:00', \
 #                               '17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00'], rotation=0)
  #  ax4.set_xlim([6000,16236])
 #   ax4.set_xticks([6000,7200,8400,9600,10800,12000,13200,14400,15600,16236])
  #  ax4.set_xticklabels(['14:00','15:00','16:00','17:00','18:00','19:00', \
   #                     '20:00','21:00','22:00','22:53'],rotation=45)

 #  ax4.set_yticks([0, 8.77192982e-05, 1.75438596e-04, 2.63157895e-04, 3.50877193e-04, 4.38596491e-04])
 #   ax4.set_yticklabels(['0.','3.16','1.58','1.05','0.79','0.63'])

  #  ax4.set_xticks([2900,3600,7200,10800,14400,18000,21600,25200,28800,31920])
  #  ax4.set_xticklabels(['14:00','15:00','16:00','17:00','18:00','19:00', \
          #              '20:00','21:00','22:00','22:53'], rotation=45)

    ax4.set_xticks([1875, 5009, 8144, 11279, 14414, 17548,20683,23818,26953, 28750])
    ax4.set_xticklabels(['14:00','15:00','16:00','17:00','18:00','19:00', \
                        '20:00','21:00','22:00','22:53'], rotation=45)
   # ax4.set_xticklabels(['14:00', '22:53'])

    cbar = plt.colorbar(mpbl,label='Power Spectral Density',ax=ax4)
    
    # do this manually, as tight_layout can't handle it
    plt.subplots_adjust(top = 0.95, bottom = 0.2, hspace=0.8)

    if savePlots:
        outfile = os.path.join(saveDirectory, outputFilename)
       # outputCurveData = os.path.join(saveDirectory, outputCurve)
        outputSpectrum = os.path.join(saveDirectory, 'spectrum.csv')
        outputLine = os.path.join(saveDirectory, theDate + '_line.csv')
        print("Save derived line to:",outputLine)
        # Save 12:00 to 24:00
        np.savetxt(outputLine, smooth_arr[840:1320],delimiter=',')
        plt.savefig(outfile)
        print("Saved plot", outputFilename)
      #  print("Saved curve data:",outputCurveData)
       # print("smooth_arr len:",len(smooth_arr))
        print("smooth arr",smooth_arr[840:1320])
        print("Sxx:",Sxx[0:9])

        with open(outputSpectrum, 'a', newline='') as mywriter:
          #  mywriter = csv.writer(csvfile, delimiter=',')
            mywriter.write(theDate)
            mywriter.write(',')

            for i in range(0,9):
                mywriter.write(str(Sxx[i][0]))
                mywriter.write(',')
            mywriter.write('\n')

    if showPlots:
        plt.show()


################## Section to write results of above analysis to summary csv file ###########    

    if saveSummary == True:
        current_month = theDate[5:7]
        current_year = theDate[0:4]
        print("current month:",current_month)
        with open(saveDirectory  + '\\summary.csv', 'a') as fd:
          if FirstLine == True:
              fd.write('Date,Raw Max,Raw Sum,Noise Sum,F1 sum, F2 sum, F3 sum\n')
              last_month = current_month
              last_year = theDate[0:4]
              FirstLine = False
              Sxx_1_max_month_total = 0
              Sxx_1_int_month_total = 0
          print("write summary line for ",theDate, "to dir:",saveDirectory)
          print("fingerprint", F1sum, F2sum, F3sum)
          fd.write(theDate + ','  + '{0:1.1f}'.format(Sxx_1.max())+','+'{0:1.1f}'.format(Sxx_1.sum()) + 
                    ',' + '{0:1.1f}'.format(noise_measure) + ',' +  '{0:1.1f}'.format(F1sum) +',' +
                          '{0:1.1f}'.format(F2sum)         + ',' +  '{0:1.1f}'.format(F3sum) )
       #   if rejected == True:
       #       fd.write(',' + '{0:1.1f}'.format(Sxx_1.sum()))
       #   if No_LSTID == True:
       #       fd.write(',,' + '{0:1.1f}'.format(Sxx_1.sum()))
       #   if LSTID == True:
       #       fd.write(',,,' + '{0:1.1f}'.format(Sxx_1.sum()))
          if current_month != last_month:
          #  print("Write Monthly Summary",(last_year + "-" + last_month))
         #   fd.write("," + last_year + "-" + last_month + "," + '{0:1.1f}'.format(Sxx_1_max_month_total)+','+'{0:1.1f}'.format(Sxx_1_int_month_total))
          #  output_sums(fd)
            monthlyResults = [(last_year + "-" + last_month), ('{0:1.1f}'.format(Sxx_1_max_month_total)), ('{0:1.1f}'.format(Sxx_1_int_month_total))]
            summaryResults.append([last_year , last_month, '{0:1.1f}'.format(Sxx_1_max_month_total), '{0:1.1f}'.format(Sxx_1_int_month_total)])

            Sxx_1_max_month_total = Sxx_1.max()
            Sxx_1_int_month_total = Sxx_1.sum()
            last_month = current_month
            last_year = current_year
       #   else:
        #    if LSTID == True:
        #        Sxx_1_max_month_total = Sxx_1_max_month_total + Sxx_1.max()
        #        Sxx_1_int_month_total = Sxx_1_int_month_total + Sxx_1.sum()
          else:
             Sxx_1_max_month_total = Sxx_1_max_month_total + Sxx_1.max()
             Sxx_1_int_month_total = Sxx_1_int_month_total + Sxx_1.sum()
              
          fd.write("\n")
          fd.close()

    plt.close('all')

    return # this is not required, but shows intentional end of function




############################# MAIN #####################################################

#set up array to hold spot counts
dist = [0] * 300         # this is a column (x) of distance values in 10s of km range
minute = [dist] * 1440   # this is an array of columns, one per minute
spotcount = np.array(minute, dtype=np.uint32) # 1440 columns of 300 ranges
arr = spotcount  # to accommodate the lowess implementation
startdate = ''
enddate = ''

if 'win' in sys.platform: # Adjust escape chars so runs on Win or Linux platform
    Windows_platform = True
    print("Detected Windows platform")
    # point this to directory containing the input csv spot files
 #   inputDirectory = "E:\\multisource_data_NA_20m_T0_24hr\\*.csv"
    inputDirectory = "C:\\Users\\bengelke\\Box\share\\multisource_plots_and_data\\multisource_data_NA_20m_T0_24hr\\*.csv"
  #  saveDirectory = 'E:\\curve_comboplots4'
    saveDirectory = 'C:\\Users\\bengelke\\Box\\share\\multisource_plots_and_data\\curve_comboplots4'
else:
    Windows_platform = False
    print("Defaulting to Linux platform")
    inputDirectory = ".//data_files//*.csv"
    saveDirectory = './/curve_comboplots'

fileList = glob.glob(inputDirectory)

print("Show plots? (y, N)")
qshowplots = input()
if qshowplots == '' or qshowplots == 'N' or qshowplots == 'n':
    showPlots = False
else:
    showPlots = True

jobtype = '0'
while jobtype != '1' and jobtype != '2':
    print("Job type: Enter 1 to process a single date, or 2 to process a range of dates")
    jobtype = input()

if jobtype == '1':
    print("Date to process (YYYY-MM-DD)?")
    startdate = input()
    startdate_obj =  datetime.strptime(startdate, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    enddate_obj = startdate_obj
else:
    print("Start date? (YYYY-MM-DD); or Enter to start at beginning and process entire directory")
    startdate = input()
    if startdate != '':
        startdate_obj =  datetime.strptime(startdate, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        print("End date? (YYYY-MM-DD); Enter to start at start date and process all later files")
        enddate = input()
        if enddate != '':
            enddate_obj =  datetime.strptime(enddate, '%Y-%m-%d').replace(tzinfo=timezone.utc)


startdate_ts = datetime.timestamp(startdate_obj)
enddate_ts   = datetime.timestamp(enddate_obj)

# Build array of columns (minutes), where each entry contains the number of th rang bin
# where the max gradient of the column appears

for datafile in fileList:

    if Windows_platform == True:
    	f_dat = datafile.rsplit('\\') # extract the date from the file path
    	theDate = f_dat[7][6:16]
    else:
        f_dat = datafile.rsplit('/')
        theDate = f_dat[3][6:16]
 #   print("f_dat=",f_dat)
 #   print("theDate:",theDate)
    
    theDate_obj = datetime.strptime(theDate, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    theDate_ts = datetime.timestamp(theDate_obj)
    if jobtype == '1': # are we processing a single date
        if startdate != theDate:
            continue
    else:
        if theDate_ts < startdate_ts or  \
           theDate_ts > enddate_ts:
            continue
    print("Processing ",theDate)


    df =  pd.read_csv(datafile, dtype=np.uint32, header=None).T
    arr = np.loadtxt(datafile, dtype=np.uint32, delimiter=',').T

    spotcount = np.transpose(df).to_numpy()
 #   print("transposed df:",spotcount)

    plot_images_on_date(spotcount,theDate)




print("end of processing")
if saveSummary:
    summaryResults.append([last_year , last_month, '{0:1.1f}'.format(Sxx_1_max_month_total), '{0:1.1f}'.format(Sxx_1_int_month_total)])
    with open(saveDirectory  + '\\summary.csv', 'a') as fd:
     # output_sums(fd)
      fd.write("\n\n")
      for month in summaryResults:
        for value in month:
            fd.write(value + ",")
        fd.write('\n')


        
    
