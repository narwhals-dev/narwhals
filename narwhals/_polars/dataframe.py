import polars as pl
from narwhals._polars.namespace import PolarsNamespace


class PolarsDataFrame:
    def __init__(self, df):
        self._native_frame = df
    def __repr__(self):
        return 'PolarsDataFrame'

    def __narwhals_dataframe__(self):
        return self
    def __narwhals_namespace__(self):
        return PolarsNamespace()
    
    def _from_native_frame(self, df):
        return self.__class__(df)
    
    def __getattr__(self, attr):
        return lambda *args, **kwargs: self._from_native_frame(getattr(self._native_frame, attr)(*args, **kwargs))

    def __getitem__(self, item):
        return self._from_native_frame(self._native_frame.__getitem__(item))
    
    @property
    def columns(self):
        return self._native_frame.columns


class PolarsLazyFrame(pl.LazyFrame):
    def __init__(self, df):
        self._native_frame = df
    def __repr__(self):
        return 'PolarsDataFrame'
    def __narwhals_lazyframe__(self):
        return self

    def _from_native_frame(self, df):
        return self.__class__(df)
    
    def __getattr__(self, attr):
        return lambda *args, **kwargs: self._from_native_frame(getattr(self._native_frame, attr)(*args, **kwargs))

    @property
    def columns(self):
        return self._native_frame.columns