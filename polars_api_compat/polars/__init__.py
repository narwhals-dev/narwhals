import polars as pl
def convert(df):
    return DataFrame(df), pl

class DataFrame:
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
