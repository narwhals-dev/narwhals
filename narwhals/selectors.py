from __future__ import annotations

from typing import Any

from narwhals import dtypes
from narwhals.dtypes import translate_dtype
from narwhals.expression import Expr
from narwhals.utils import flatten


def by_dtype(*dtypes: Any) -> Expr:
    """
    Select columns based on their dtype.

    Arguments:
        dtypes: one or data types to select

    Examples:
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> import pandas as pd
        >>> import polars as pl
        >>>
        >>> data = {'a': [1, 2], 'b': ['x', 'y'], 'c': [4.1, 2.3]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)

        Let's define a dataframe-agnostic function to select int64 and float64
        dtypes and multiplies each value by 2:

        >>> def func(df_any):
        ...     df = nw.from_native(df_any)
        ...     df = df.select(ncs.by_dtype(nw.Int64, nw.Float64)*2)
        ...     return nw.to_native(df)

        We can then pass either pandas or Polars dataframes:

        >>> func(df_pd)
           a    c
        0  2  8.2
        1  4  4.6
        >>> func(df_pl)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ c   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 2   ┆ 8.2 │
        │ 4   ┆ 4.6 │
        └─────┴─────┘
    """
    return Expr(
        lambda plx: plx.selectors.by_dtype(
            [translate_dtype(plx, dtype) for dtype in flatten(dtypes)]
        )
    )


def numeric() -> Expr:
    return by_dtype(
        dtypes.Int64,
        dtypes.Int32,
        dtypes.Int16,
        dtypes.Int8,
        dtypes.UInt64,
        dtypes.UInt32,
        dtypes.UInt16,
        dtypes.UInt8,
        dtypes.Float64,
        dtypes.Float32,
    )
