from pathlib import Path

import pandas as pd
import polars as pl
from queries import q7

nation = Path("data") / "nation.parquet"
customer = Path("data") / "customer.parquet"
lineitem = Path("data") / "lineitem.parquet"
orders = Path("data") / "orders.parquet"
supplier = Path("data") / "supplier.parquet"

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
print(q7.query(fn(nation), fn(customer), fn(lineitem), fn(orders), fn(supplier)))


tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q7.query(fn(nation), fn(customer), fn(lineitem), fn(orders), fn(supplier)))

tool = "polars[eager]"
fn = IO_FUNCS[tool]
print(q7.query(fn(nation), fn(customer), fn(lineitem), fn(orders), fn(supplier)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(
    q7.query(fn(nation), fn(customer), fn(lineitem), fn(orders), fn(supplier)).collect()
)
