from pathlib import Path

import pandas as pd
import polars as pl
from queries import q10

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

nation = Path("data") / "nation.parquet"
partsupp = Path("data") / "partsupp.parquet"
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
print(q10.query(fn(nation), fn(nation), fn(partsupp), fn(supplier)))

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q10.query(fn(nation), fn(nation), fn(partsupp), fn(supplier)))

tool = "polars[eager]"
fn = IO_FUNCS[tool]
print(q10.query(fn(nation), fn(nation), fn(partsupp), fn(supplier)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q10.query(fn(nation), fn(nation), fn(partsupp), fn(supplier)).collect())
