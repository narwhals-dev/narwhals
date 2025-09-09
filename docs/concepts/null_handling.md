# Null/NaN handling

## TL;DR

All dataframe tools, except for those which piggy-back off of pandas, make a clear
distinction between NaN and null values. 

!!! tip
    **We recommend only handling null values in applications and leaving NaN values as an 
    edge case resulting from users having performed undefined mathematical operations.**

## What's the difference?

Most data tools except pandas make a clear distinction between:

- Null values, representing missing data.
- NaN values, resulting from "illegal" mathematical operations like `0/0`.

In Narwhals, this is reflected in separate methods for Null/NaN values:

| Operation | Null                                          | NaN                                                                                                                                                                                                                                              |
| --------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| is        | [`Expr.is_null`][narwhals.Expr.is_null]       | [`Expr.is_nan`][narwhals.Expr.is_nan]                                                                                                                                                                                                            |
| fill      | [`Expr.fill_null`][narwhals.Expr.fill_null]   | [`Expr.fill_nan`][narwhals.Expr.fill_nan]                                                                                                                                                                                                        |
| drop      | [`Expr.drop_nulls`][narwhals.Expr.drop_nulls] | *Not yet implemented (See [discussion](https://github.com/narwhals-dev/narwhals/issues/3031#issuecomment-3219910366))*<br>[`polars.Expr.drop_nans`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.drop_nans.html) |
| count     | [`Expr.null_count`][narwhals.Expr.null_count] | *No upstream equivalent*                                                                                                                                                                                                                         |

In pandas however the concepts are muddied, as different sentinel values represent *missing* [depending on the data type](https://pandas.pydata.org/docs/user_guide/missing_data.html).

Check how different tools distinguish them (or don't) in the following example:

```python exec="1" source="above" session="null_handling"
import narwhals as nw
import numpy as np
from narwhals.typing import IntoFrameT

data = {"a": [1.0, 0.0, None]}


def check_null_behavior(df: IntoFrameT) -> IntoFrameT:
    return (
        nw.from_native(df)
        .with_columns(a=nw.col("a") / nw.col("a"))
        .with_columns(
            a_is_null=nw.col("a").is_null(),
            a_is_nan=nw.col("a").is_nan(),
        )
    ).to_native()
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="null_handling"
    import pandas as pd

    df = pd.DataFrame(data)
    print(check_null_behavior(df))
    ```

=== "pandas (pyarrow-backed)"
    ```python exec="true" source="material-block" result="python" session="null_handling"
    import pandas as pd

    df = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
    print(check_null_behavior(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="null_handling"
    import polars as pl

    df = pl.DataFrame(data)
    print(check_null_behavior(df))
    ```

=== "PyArrow"
    ```python exec="true" source="material-block" result="python" session="null_handling"
    import pyarrow as pa

    df = pa.table(data)
    print(check_null_behavior(df))
    ```

Notice how the classic pandas dtypes make no distinction between the concepts, whereas the other
libraries do. Note however that discussion on what PyArrow-backed pandas dataframe should do
[is ongoing](https://github.com/pandas-dev/pandas/issues/32265).

## NaN comparisons

According to the IEEE-754 standard, NaN should compare as not equal to itself, and cannot
be compared with other floating point numbers. Python and PyArrow follow these rules:

```python exec="1" source="above" session="nan-comparisons" result="python"
import pyarrow as pa
import pyarrow.compute as pc

print("Python result:")
print(float("nan") == float("nan"), 0.0 == 0.0)
print()
print("PyArrow result:")
arr = pa.array([float("nan"), 0.0])
print(pc.equal(arr, arr))
```

Polars and DuckDB, however, don't follow this rule, and treat NaN as equal to itself.

```python exec="1" source="above" session="nan-comparisons" result="python"
import polars as pl
import duckdb

print("Polars result")
df = pl.DataFrame({"a": [float("nan"), 0.0]})
print(df.with_columns(a_equals_a=pl.col("a") == pl.col("a")))
print()
print("DuckDB result")
print(duckdb.sql("from df select a, a == a as a_equals_a"))
```

Furthermore, Polars [excludes NaN values in `max`](https://github.com/pola-rs/polars/issues/23635)
whereas DuckDB treats them as larger than any other floating-point value.

For all these reasons it bears reiterating that our recommendation is to only handle null values in applications, and leave NaN values as an edge case resulting from users having performed undefined mathematical operations.
