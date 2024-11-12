from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

lineitem = Path("data") / "lineitem.parquet"
region = Path("data") / "region.parquet"
nation = Path("data") / "nation.parquet"
supplier = Path("data") / "supplier.parquet"
part = Path("data") / "part.parquet"
partsupp = Path("data") / "partsupp.parquet"
orders = Path("data") / "orders.parquet"
customer = Path("data") / "customer.parquet"
line_item = Path("data") / "lineitem.parquet"

IO_FUNCS = {
    "pandas": lambda x: pd.read_parquet(x, engine="pyarrow"),
    "pandas[pyarrow]": lambda x: pd.read_parquet(
        x, engine="pyarrow", dtype_backend="pyarrow"
    ),
    "polars[eager]": lambda x: pl.read_parquet(x),
    "polars[lazy]": lambda x: pl.scan_parquet(x),
    "pyarrow": lambda x: pq.read_table(x),
    "dask": lambda x: dd.read_parquet(x, engine="pyarrow", dtype_backend="pyarrow"),
}
