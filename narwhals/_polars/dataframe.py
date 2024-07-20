import polars as pl

from narwhals._polars.namespace import PolarsNamespace
from narwhals._polars.utils import extract_args_kwargs, translate_dtype
from narwhals.dependencies import get_polars


class PolarsDataFrame:
    def __init__(self, df):
        self._native_dataframe = df

    def __repr__(self):
        return "PolarsDataFrame"

    def __narwhals_dataframe__(self):
        return self

    def __narwhals_namespace__(self):
        return PolarsNamespace()
    
    def __native_namespace__(self):
        return get_polars()

    def _from_native_frame(self, df):
        return self.__class__(df)

    def __getattr__(self, attr):
        def func(*args, **kwargs):
            args, kwargs = extract_args_kwargs(args, kwargs)
            return self._from_native_frame(
                getattr(self._native_dataframe, attr)(*args, **kwargs)
            )

        return func

    @property
    def schema(self):
        schema = self._native_dataframe.schema
        return {
            name: translate_dtype(dtype)
            for name, dtype in schema.items()
        }

    def __getitem__(self, item):
        pl = get_polars()
        result = self._native_dataframe.__getitem__(item)
        if isinstance(result, pl.Series):
            from narwhals._polars.series import PolarsSeries

            return PolarsSeries(result)
        return self._from_native_frame(result)

    @property
    def columns(self):
        return self._native_dataframe.columns


class PolarsLazyFrame(pl.LazyFrame):
    def __init__(self, df):
        self._native_dataframe = df

    def __repr__(self):
        return "PolarsDataFrame"

    def __narwhals_lazyframe__(self):
        return self

    def _from_native_frame(self, df):
        return self.__class__(df)

    def __getattr__(self, attr):
        return lambda *args, **kwargs: self._from_native_frame(
            getattr(self._native_dataframe, attr)(*args, **kwargs)
        )

    @property
    def columns(self):
        return self._native_dataframe.columns
