from pathlib import Path

import pandas as pd
import polars as pl
from queries import q17

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

lineitem = Path("data") / "lineitem.parquet"
part = Path("data") / "part.parquet"

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
print(q17.query(fn(lineitem), fn(part)))

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)))

tool = "polars[eager]"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)).collect())
