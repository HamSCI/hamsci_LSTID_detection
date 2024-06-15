import pandas as pd
import xarray as xr
import numpy as np
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
#         assert isinstance(dates, list), 'Must pass list, either of datetimes/strings or tuples of datetimes/strings'
        for date in dates:
            if isinstance(date, tuple):
                start_date, end_date = date
                dates = pd.date_range(start=start_date, end=end_date)
                for date, arr in self.iter_dates(dates, skip_missing=skip_missing, **get_kwargs):
                    yield date, arr
                # for date in dates:
                    # this should be recursive
                    # if skip_missing:
                    #     try:
                    #         yield date, self.get_date(date, **get_kwargs)
                    #     except KeyError:
                    #         continue
                    # else:
                    #     yield date, self.get_date(date, **get_kwargs)
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