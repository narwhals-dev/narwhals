from __future__ import annotations

import pandas as pd
import polars as pl
import pyarrow as pa
import narwhals as nw
from narwhals.typing import IntoFrame


def agnostic_get_columns(df_native: IntoFrame) -> list[str]:
    df = nw.from_native(df_native)
    column_names = df.columns
    return column_names


data = {"a": [1, 2, 3], "b": [4, 5, 6]}
df_pandas = pd.DataFrame(data)
df_polars = pl.DataFrame(data)
table_pa = pa.table(data)

print("pandas output")
print(agnostic_get_columns(df_pandas))

print("Polars output")
print(agnostic_get_columns(df_polars))

print("PyArrow output")
print(agnostic_get_columns(table_pa))