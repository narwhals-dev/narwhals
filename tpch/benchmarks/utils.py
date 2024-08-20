import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

lib_to_reader = {
    "dask": lambda path: dd.read_parquet(path, dtype_backend="pyarrow"),
    "pandas": lambda path: pd.read_parquet(path, engine="pyarrow"),
    "polars": pl.scan_parquet,
    "pyarrow": pq.read_table,
}
