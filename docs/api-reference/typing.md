# `narwhals.typing`

Narwhals comes fully statically typed. In addition to `nw.DataFrame`, `nw.Expr`,
`nw.Series`, `nw.LazyFrame`, we also provide the following type hints:

::: narwhals.typing
    handler: python
    options:
      members:
        - DataFrameT
        - Frame
        - FrameT
        - IntoDataFrame
        - IntoDataFrameT
        - IntoExpr
        - IntoFrame
        - IntoFrameT
        - IntoSeries
        - IntoSeriesT
        - SizeUnit
        - TimeUnit
      show_source: false
      show_bases: false

## `nw.narwhalify`, or `nw.from_native`?

Although some people find the former more readable, the latter is better at preserving type hints.

Here's an example:
```python
import polars as pl
import narwhals as nw
from narwhals.typing import IntoDataFrameT, DataFrameT

df = pl.DataFrame({"a": [1, 2, 3]})


def func(df_native: IntoDataFrameT) -> IntoDataFrameT:
    df = nw.from_native(df_native, eager_only=True)
    return df.select(b=nw.col("a")).to_native()


reveal_type(func(df))


@nw.narwhalify(strict=True)
def func_2(df: DataFrameT) -> DataFrameT:
    return df.select(b=nw.col("a"))


reveal_type(func_2(df))
```

Running `mypy` on it gives:
```console
$ mypy t.py 
t.py:13: note: Revealed type is "polars.dataframe.frame.DataFrame"
t.py:21: note: Revealed type is "Any"
Success: no issues found in 1 source file
```

In the first case, mypy can infer that `df` is a `polars.DataFrame`. In the second case, it can't.

If you want to make the most out of type hints and preserve them as much as possible, we recommend
`nw.from_native` and `nw.to_native`. Type hints will still be respected
inside the function body if you type the arguments.
