import pandas as pd
import numpy as np
import xarray as xr

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

import os
import datetime
import joblib

def pad_axis(arr, expected_size, dtype=np.uint8, axis=0):
    shape_mismatch = expected_size - arr.shape[axis]
    left_pad = shape_mismatch // 2
    
    if shape_mismatch > 0:
        right_pad = shape_mismatch - left_pad
        axis_pad = (left_pad, right_pad)
        full_pad = [(0, 0) if i != axis else axis_pad for i in range(arr.ndim)]
        arr = np.pad(arr, tuple(full_pad), mode='constant', constant_values=0)
    elif shape_mismatch < 0:
        left_pad = -shape_mismatch // 2
        right_pad = -shape_mismatch - left_pad
        arr = arr[:,left_pad:-right_pad]
        
    assert arr.dtype == dtype, dtype
    assert arr.shape[axis] == expected_size, f'{arr.shape[axis]} Mismatches Expected {axis} Dimension of {expected_size}'
    return arr
    
def pad_img(img, expected_shape=(1440, 300), dtype=np.uint8):
    """
    Raw input data has inconsistent size, very close but not precisely
    the intended (1440, 300) size. This pads the image to make
    it exactly (1440, 300) but does so evenly on both sides, if required.
    """
    assert len(expected_shape) == img.ndim
    for i in range(img.ndim):
        img = pad_axis(img, expected_shape[i], axis=i, dtype=dtype)
    return img

def cut_half(img, expected_size=1440): #, vempty=True):
    """ Simple preprocessing for image, could add additional adjustments here """
    if expected_size:
        assert img.shape[0] == expected_size, f'Mismatch with width, dim 0 of {img.shape} != {expected_size}'
        assert not expected_size % 2, 'Width must be even'
    img = img[expected_size // 2:,:]
    return img

def create_xarr(
    parent_dir='raw_data/', 
    filter_fn=None, 
    max_iter=None, 
    read_pandas=True, 
    expected_shape=(720, 300),
    dtype=np.uint8, 
    height_start=0, 
    time_start='12:00',
    apply_fn=None,
    plot=True,
    split_idx=1,
):
    in_dtype, out_dtype = dtype if len(dtype) == 2 else (dtype, dtype)
    img_list = list()
    stat_list = list()
    file_list = sorted(os.listdir(parent_dir))
    if filter_fn is None:
        filter_fn = lambda x : x.endswith('.csv')
        
    for i, file in enumerate(filter(filter_fn, file_list)):
        full_path = os.path.join(parent_dir, file)
        if max_iter is not None and i >= max_iter: break
        print(i, end='\r')
        split_file = file.split('_')
        # if len(split_file) != 3:
        #     raise ValueError('Split index incorrect')
        try:
            date = pd.to_datetime(split_file[split_idx].replace('.csv',''))
        except pd.errors.ParserError:
            raise ValueError(f'Split returned invalid date')

        if read_pandas:
            img = pd.read_csv(full_path)
            assert np.all(img >= 0)
            assert np.all(img <= np.iinfo(in_dtype).max)
            img = img.to_numpy(dtype=in_dtype)
        else:
            img = np.genfromtxt(full_path, delimiter=',').astype(in_dtype)

        img = pad_img(img, expected_shape=(expected_shape[0] * 2, expected_shape[1]), dtype=in_dtype) # standardize width
        img = cut_half(img, expected_size=expected_shape[0] * 2) # trim to 12 hours of daytime
        assert img.shape == expected_shape, img.shape
        if apply_fn is not None:
            img = apply_fn(img)

        if plot:
            plt.figure()
            plt.title(file)
            plt.imshow(img.T)
            plt.show()
            
        img_list.append((date.to_pydatetime(), img))
        
    dates, imgs = zip(*img_list)
    # start_time = datetime.datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
    # times = [start_time + datetime.timedelta(minutes=i) for i in range(expected_shape[0])]
    times = pd.timedelta_range(start='12:00:00', end='23:59:00', freq='1min')
    heights = np.arange(height_start, 10 * expected_shape[1], 10)

    img_arr = np.stack(imgs, axis=0, dtype=out_dtype)
    assert img_arr.shape[1] == expected_shape[0], f'{img_arr.shape} | {expected_shape}'
    assert img_arr.shape[2] == expected_shape[1], f'{img_arr.shape} | {expected_shape}'
        
    full_xarr = xr.DataArray(
        img_arr,
        coords={
            'date' : list(dates),
            'time' : times,
            'height' : heights,
        },
        dims=['date','time','height'],
    )
    return full_xarr

def str_to_float_col(x):
    return x.astype(str).str.strip().replace('',0).astype(np.float32)

def create_label_df(csv_path='official_labels.csv'):
    official_df = pd.read_csv(csv_path)
    
    index = pd.to_datetime(official_df['Date'].str.strip(), format='%m/%d/%Y')
    index.name = 'date'
    label_df = pd.DataFrame(
        {
            'binary_label' : ~(official_df['Start time'] == 0),
            'xmin' : str_to_float_col(official_df['Start time']).sub(12).multiply(60),
            'xmax' : str_to_float_col(official_df['End time']).sub(12).multiply(60),
            'ymin' : str_to_float_col(official_df['Low range']).div(10),
            'ymax' : str_to_float_col(official_df['High range']).div(10),
            'period' : str_to_float_col(official_df['Period']).multiply(60),            
        },
        index=index,
    )
    return label_df

# def quantile_normalize(t):
#     assert t.shape[0] > t.shape[1] and t.shape[1] > t.shape[2], t.shape
#     flat_t = t.reshape(t.shape[0], -1)
#     flat_t = st.rankdata(flat_t, method='dense', axis=1) #, nan_policy='raise')
    
#     quantiles = ranks / len(chunk)

#     # Apply inverse cumulative distribution function to get normalized values
#     normalized_values = np.percentile(chunk, quantiles * 100)
    
#     t = flat_t.reshape(t.shape)
#     return t

def mad(t, min_dev=.05):
    median = np.median(t, axis=(0, 1), keepdims=True)
    abs_devs = np.abs(t - median)
    mad = abs_devs / max(np.median(abs_devs, axis=(0, 1), keepdims=True), min_dev)
    assert t.shape == mad.shape, f'{t.shape} | {mad.shape}'
    return mad
