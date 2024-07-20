import polars as pl

class PolarsNamespace:
    def __getattr__(self, attr):
        return getattr(pl, attr)
