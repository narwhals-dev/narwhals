# `narwhals.typing`

Narwhals comes fully statically typed. In addition to `nw.DataFrame`, `nw.Expr`,
`nw.Series`, `nw.LazyFrame`, we also provide the following type hints:

## `DataFrameT`
A `TypeVar` bound to `nw.DataFrame`. Use this when you have a function which
accepts a `nw.DataFrame` and returns a `nw.DataFrame` backed by the same backend, example:

  ```python
  import narwhals as nw
  from narwhals.typing import DataFrameT

  @nw.narwhalify
  def func(df: DataFrameT) -> DataFrameT:
    return df.with_columns(c=df['a']+1)
  ```

## `Frame`

Either a `nw.DataFrame` or `nw.LazyFrame`. Use this if your function can work on
either and your function doesn't care about its backend, example:

  ```python
  import narwhals as nw
  from narwhals.typing import Frame

  @nw.narwhalify
  def func(df: Frame) -> list[str]:
    return df.columns
  ```

## `FrameT`
A `TypeVar` bound to `Frame`. Use this if your function accepts either `nw.DataFrame`
or `nw.LazyFrame` and returns an object backed by the same backend, example:

  ```python
  import narwhals as nw
  from narwhals.typing import FrameT

  @nw.narwhalify
  def func(df: FrameT) -> FrameT:
    return df.with_columns(c=nw.col('a')+1)
  ```

## `IntoDataFrame`
An object which can be converted to `nw.DataFrame` (e.g. `pd.DataFrame`, `pl.DataFrame`).
Use this if your function accepts a narwhalifiable object but it doesn't care about its backend:

  ```python
  from __future__ import annotations

  import narwhals as nw
  from narwhals.typing import IntoDataFrameT

  def func(df_native: IntoDataFrame) -> tuple[int, int]:
    df = nw.from_native(df_native, eager_only=True)
    return df.shape
  ```

## `IntoDataFrameT`
A `TypeVar` bound to `IntoDataFrame`. Use this if your function accepts
a function which can be converted to `nw.DataFrame` and returns an object of the same
class:

  ```python
  import narwhals as nw
  from narwhals.typing import IntoDataFrameT

  def func(df_native: IntoDataFrameT) -> IntoDataFrameT:
    df = nw.from_native(df_native, eager_only=True)
    return nw.to_native(df.with_columns(c=df['a']+1))
  ```

## `IntoExpr`
Use this to mean "either a Narwhals expression, or something
which can be converted into one". For example, `exprs` in `DataFrame.select` is
typed to accept `IntoExpr`, as it can either accept a `nw.Expr` (e.g. `df.select(nw.col('a'))`)
or a string which will be interpreted as a `nw.Expr`, e.g. `df.select('a')`.

## `IntoFrame`
An object which can be converted to `nw.DataFrame` or `nw.LazyFrame`
(e.g. `pd.DataFrame`, `pl.DataFrame`, `pl.LazyFrame`). Use this if your function can accept
an object which can be converted to either `nw.DataFrame` or `nw.LazyFrame` and it doesn't
care about its backend:

```python
import narwhals as nw
from narwhals.typing import IntoFrame

def func(df_native: IntoFrame) -> list[str]:
  df = nw.from_native(df_native)
  return df.columns
```

## `IntoFrameT`
A `TypeVar` bound to `IntoFrame`. Use this if your function accepts an
object which is convertible to `nw.DataFrame` or `nw.LazyFrame` and returns an object
of the same type:

  ```python
  import narwhals as nw
  from narwhals.typing import IntoFrameT

  def func(df_native: IntoFrameT) -> IntoFrameT:
    df = nw.from_native(df_native)
    return nw.to_native(df.with_columns(c=nw.col('a')+1))
  ```

## `nw.narwhalify`, or `nw.from_native`?

Although the former is more readable, the latter is better at preserving type hints.

Here's an example:
```python
import polars as pl
import narwhals as nw
from narwhals.typing import IntoDataFrameT, DataFrameT

df = pl.DataFrame({'a': [1,2,3]})

def func(df_any: IntoDataFrameT) -> IntoDataFrameT:
    df = nw.from_native(df_any, eager_only=True)
    return nw.to_native(df.select(b=nw.col('a')))

reveal_type(func(df))

@nw.narwhalify(strict=True)
def func_2(df: DataFrameT) -> DataFrameT:
    return df.select(b=nw.col('a'))

reveal_type(func_2(df))
```

Running `mypy` on it gives:
```console
$ mypy f.py 
f.py:11: note: Revealed type is "polars.dataframe.frame.DataFrame"
f.py:17: note: Revealed type is "Any"
Success: no issues found in 1 source file
```

In the first case, mypy can infer that `df` is a `polars.DataFrame`. In the second case, it can't.

If you want to make the most out of type hints and preserve them as much as possible, we recommend
`nw.from_native` and `nw.to_native` - otherwise, `nw.narwhalify`. Type hints will still be respected
inside the function body if you type the arguments.
