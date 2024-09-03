from pathlib import Path

import pandas as pd
import polars as pl
from queries import q6

lineitem = Path("data") / "lineitem.parquet"
IO_FUNCS = {
    "pandas": lambda x: pd.read_parquet(x, engine="pyarrow"),
    "pandas[pyarrow]": lambda x: pd.read_parquet(
        x, engine="pyarrow", dtype_backend="pyarrow"
    ),
    "polars[eager]": lambda x: pl.read_parquet(x),
    "polars[lazy]": lambda x: pl.scan_parquet(x),
}

tool = "pandas"
fn = IO_FUNCS[tool]
print(q6.query(fn(lineitem)))

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q6.query(fn(lineitem)))

tool = "polars[eager]"
fn = IO_FUNCS[tool]
print(q6.query(fn(lineitem)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q6.query(fn(lineitem)).collect())
