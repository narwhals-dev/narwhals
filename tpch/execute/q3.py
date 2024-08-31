from pathlib import Path

import ibis
import pandas as pd
import polars as pl
from queries import q3

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


con_pd = ibis.pandas.connect()
con_pl = ibis.polars.connect()

customer = Path("data") / "customer.parquet"
lineitem = Path("data") / "lineitem.parquet"
orders = Path("data") / "orders.parquet"

IO_FUNCS = {
    "pandas": lambda x: pd.read_parquet(x, engine="pyarrow"),
    "pandas[pyarrow]": lambda x: pd.read_parquet(
        x, engine="pyarrow", dtype_backend="pyarrow"
    ),
    "pandas[pyarrow][ibis]": lambda x: con_pd.read_parquet(
        x, engine="pyarrow", dtype_backend="pyarrow"
    ),
    "polars[eager]": lambda x: pl.read_parquet(x),
    "polars[lazy]": lambda x: pl.scan_parquet(x),
    "polars[lazy][ibis]": lambda x: con_pl.read_parquet(x),
}

tool = "pandas[pyarrow][ibis]"
fn = IO_FUNCS[tool]
print(q3.query_ibis(fn(customer), fn(lineitem), fn(orders), tool="pandas"))

tool = "polars[lazy][ibis]"
fn = IO_FUNCS[tool]
print(q3.query_ibis(fn(customer), fn(lineitem), fn(orders), tool="polars"))

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q3.query_pandas_native(fn(customer), fn(lineitem), fn(orders)))

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
