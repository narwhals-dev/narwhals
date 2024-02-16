import polars as pl
from polars_api_compat.spec import (
    DataFrame as DataFrameT,
    LazyFrame as LazyFrameT,
    Expr as ExprT,
)


def convert(df):
    return DataFrame(df), pl


class DataFrame(DataFrameT):
    def __init__(self, df):
        self.df = df

    def with_columns(self, *args, **kwargs):
        return DataFrame(self.df.with_columns(*args, **kwargs))

    def filter(self, cond):
        return self.df.filter(cond)

    def select(self, *args):
        return self.df.select(*args)

    def __getattr__(self, attr):
        return getattr(self.df, attr)

    @property
    def dataframe(self):
        return self.df
