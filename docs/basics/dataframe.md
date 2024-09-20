# DataFrame

To write a dataframe-agnostic function, the steps you'll want to follow are:

1. Initialise a Narwhals DataFrame or LazyFrame by passing your dataframe to `nw.from_native`.
    All the calculations stay lazy if we start with a lazy dataframe - Narwhals will never automatically trigger computation without you asking it to.

    Note: if you need eager execution, make sure to pass `eager_only=True` to `nw.from_native`.

2. Express your logic using the subset of the Polars API supported by Narwhals.
3. If you need to return a dataframe to the user in its original library, call `nw.to_native`.

Steps 1 and 3 are so common that we provide a utility `@nw.narwhalify` decorator, which allows you
to only explicitly write step 2.

Let's explore this with some simple examples.

## Example 1: descriptive statistics

Just like in Polars, we can pass expressions to
`DataFrame.select` or `LazyFrame.select`.

Make a Python file with the following content:

```python exec="1" source="above" session="df_ex1"
import narwhals as nw
from narwhals.typing import FrameT


@nw.narwhalify
def func(df: FrameT) -> FrameT:
    return df.select(
        a_sum=nw.col("a").sum(),
        a_mean=nw.col("a").mean(),
        a_std=nw.col("a").std(),
    )
```

Let's try it out:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import pandas as pd

    df = pd.DataFrame({"a": [1, 1, 2]})
    print(func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import polars as pl

    df = pl.DataFrame({"a": [1, 1, 2]})
    print(func(df))
    ```

=== "Polars (lazy)"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import polars as pl

    df = pl.LazyFrame({"a": [1, 1, 2]})
    print(func(df).collect())
    ```

=== "PyArrow"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import pyarrow as pa

    table = pa.table({"a": [1, 1, 2]})
    print(func(table))
    ```

Alternatively, we could have opted for the more explicit version:

```python
import narwhals as nw
from narwhals.typing import IntoFrameT


def func(df_native: IntoFrameT) -> IntoFrameT:
    df = nw.from_native(df_native)
    df = df.select(
        a_sum=nw.col("a").sum(),
        a_mean=nw.col("a").mean(),
        a_std=nw.col("a").std(),
    )
    return nw.to_native(df)
```

Despite being more verbose, it has the advantage of preserving the type annotation of the native
object - see [typing](../api-reference/typing.md) for more details.

In general, in this tutorial, we'll use the former.

## Example 2: group-by and mean

Just like in Polars, we can pass expressions to `GroupBy.agg`.
Make a Python file with the following content:

```python exec="1" source="above" session="df_ex2"
import narwhals as nw
from narwhals.typing import FrameT


@nw.narwhalify
def func(df: FrameT) -> FrameT:
    return df.group_by("a").agg(nw.col("b").mean()).sort("a")
```

Let's try it out:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="df_ex2"
    import pandas as pd

    df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    print(func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="df_ex2"
    import polars as pl

    df = pl.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    print(func(df))
    ```

=== "Polars (lazy)"
    ```python exec="true" source="material-block" result="python" session="df_ex2"
    import polars as pl

    df = pl.LazyFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    print(func(df).collect())
    ```

=== "PyArrow"
    ```python exec="true" source="material-block" result="python" session="df_ex2"
    import pyarrow as pa

    table = pa.table({"a": [1, 1, 2], "b": [4, 5, 6]})
    print(func(table))
    ```

## Example 3: horizontal sum

Expressions can be free-standing functions which accept other expressions as inputs.
For example, we can compute a horizontal sum using `nw.sum_horizontal`.

Make a Python file with the following content:

```python exec="1" source="above" session="df_ex3"
import narwhals as nw
from narwhals.typing import FrameT


@nw.narwhalify
def func(df: FrameT) -> FrameT:
    return df.with_columns(a_plus_b=nw.sum_horizontal("a", "b"))
```

Let's try it out:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="df_ex3"
    import pandas as pd

    df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    print(func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="df_ex3"
    import polars as pl

    df = pl.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    print(func(df))
    ```

=== "Polars (lazy)"
    ```python exec="true" source="material-block" result="python" session="df_ex3"
    import polars as pl

    df = pl.LazyFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    print(func(df).collect())
    ```

=== "PyArrow"
    ```python exec="true" source="material-block" result="python" session="df_ex3"
    import pyarrow as pa

    table = pa.table({"a": [1, 1, 2], "b": [4, 5, 6]})
    print(func(table))
    ```

## Example 4: multiple inputs

`nw.narwhalify` can be used to decorate functions that take multiple inputs as well and
return a non dataframe/series-like object.

For example, let's compute how many rows are left in a dataframe after filtering it based
on a series.

Make a Python file with the following content:

```python exec="1" source="above" session="df_ex4"
from typing import Any

import narwhals as nw


@nw.narwhalify(eager_only=True)
def func(df: nw.DataFrame[Any], s: nw.Series, col_name: str) -> int:
    return df.filter(nw.col(col_name).is_in(s)).shape[0]
```

We require `eager_only=True` here because lazyframe doesn't support `.shape`.

Let's try it out:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="df_ex4"
    import pandas as pd

    df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [4, 5, 6, 7, 8]})
    s = pd.Series([1, 3])
    print(func(df, s.to_numpy(), "a"))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="df_ex4"
    import polars as pl

    df = pl.DataFrame({"a": [1, 1, 2, 2, 3], "b": [4, 5, 6, 7, 8]})
    s = pl.Series([1, 3])
    print(func(df, s.to_numpy(), "a"))
    ```

=== "PyArrow"
    ```python exec="true" source="material-block" result="python" session="df_ex4"
    import pyarrow as pa

    table = pa.table({"a": [1, 1, 2, 2, 3], "b": [4, 5, 6, 7, 8]})
    a = pa.array([1, 3])
    print(func(table, a.to_numpy(), "a"))
    ```
