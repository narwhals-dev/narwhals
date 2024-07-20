import polars as pl

class PolarsDataFrame(pl.DataFrame):
    ...

    def __narwhals_dataframe__(self):
        return self

class PolarsLazyFrame(pl.LazyFrame):
    ...

    def __narwhals_lazyframe__(self):
        return self
