import pandas as pd
import numpy as np
import xarray as xr
import os
import joblib

class DateIter():
    def __init__(self, xarr, label_df=None, apply_fn=None):
        if isinstance(xarr, xr.DataArray):
            self.data = xarr
        elif isinstance(xarr, str):
            self.data = joblib.load(xarr)
        else:
            raise TypeError(f'Unexpected type {type(xarr)} for input array')
        
        if label_df is None or isinstance(label_df, pd.DataFrame):
            self.label_df = label_df
        elif isinstance(label_df, str):
            self.label_df = joblib.load(label_df)
        else:
            raise TypeError(f'Unexpected type {type(label_df)} for input array')
        
        if self.label_df is not None:
            label_index = self.label_df.index
            data_index = self.data.indexes['date']
            if not np.all(label_index.isin(data_index)):
                missing_dates = label_index[label_index.isin(data_index)].tolist()
                raise ValueError(
                    f'Missing the following dates represented with labels: {missing_dates}'
                )
        
        self._apply_fn = apply_fn
        return
    
    @property
    def apply_fn(self):
        return self._apply_fn
    
    @apply_fn.setter
    def apply_fn(self, x):
        self._apply_fn = x
        return
    
    def get_date(self, date, raise_missing=True):
        date = pd.to_datetime(date)
#         arr = self.data[date,:,:]
        try:
            xarr = self.data.sel(date=date)
        except KeyError as ke:
            if raise_missing:
                raise ke
            elif self.label_df is not None:
                return None, None
            else:
                return
        
        if self.apply_fn is not None:
            xarr = self.apply_fn(xarr)

        if self.label_df is not None:
            try:
                label = self.label_df.loc[date,['xmin','xmax']]
            except KeyError:
                label = None
            return xarr, label
        else:
            return xarr
    
    def iter_dates(self, dates, skip_missing=False, **get_kwargs):
        for date in dates:
            if isinstance(date, tuple):
                start_date, end_date = date
                dates = pd.date_range(start=start_date, end=end_date)
                for date, arr in self.iter_dates(dates, skip_missing=skip_missing, **get_kwargs):
                    yield date, arr
            else:
                if skip_missing:
                    try:
                        yield date, self.get_date(date, **get_kwargs)
                    except KeyError:
                        continue
                else:
                    yield date, self.get_date(date, **get_kwargs)
                
    def iter_all(self):
        return self.iter_dates(self.data.indexes['date'])
    
    def iter_labels(self):
        if self.label_df is None:
            raise AttributeError('No labels assigned, cannot iter by nonexistent `label_df`')
        dates = self.label_df.index[self.label_df.index.isin(self.data.indexes['date'])]
        return self.iter_dates(dates)

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

def cut_half(img, expected_size=1440):
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
    apply_fn=None,
    split_idx=1,
):
    in_dtype, out_dtype = dtype if len(dtype) == 2 else (dtype, dtype)
    img_list = list()
    file_list = sorted(os.listdir(parent_dir))
    if filter_fn is None:
        filter_fn = lambda x : x.endswith('.csv')
        
    for i, file in enumerate(filter(filter_fn, file_list)):
        full_path = os.path.join(parent_dir, file)
        if max_iter is not None and i >= max_iter: break
        print(i, end='\r')
        split_file = file.split('_')
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

        img_list.append((date.to_pydatetime(), img))
        
    dates, imgs = zip(*img_list)
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

def mad(t, min_dev=.05):
    median = np.median(t, axis=(0, 1), keepdims=True)
    abs_devs = np.abs(t - median)
    mad = abs_devs / max(np.median(abs_devs, axis=(0, 1), keepdims=True), min_dev)
    assert t.shape == mad.shape, f'{t.shape} | {mad.shape}'
    return mad
