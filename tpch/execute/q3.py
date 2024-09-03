from pathlib import Path

import pandas as pd
import polars as pl
from queries import q3

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


customer = Path("data") / "customer.parquet"
lineitem = Path("data") / "lineitem.parquet"
orders = Path("data") / "orders.parquet"

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
print(q3.query(fn(customer), fn(lineitem), fn(orders)))

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q3.query(fn(customer), fn(lineitem), fn(orders)))

tool = "polars[eager]"
fn = IO_FUNCS[tool]
print(q3.query(fn(customer), fn(lineitem), fn(orders)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q3.query(fn(customer), fn(lineitem), fn(orders)).collect())
