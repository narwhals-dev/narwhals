# Series

In [dataframe](dataframe.md), you learned how to write a dataframe-agnostic function.

We only used DataFrame methods there - but what if we need to operate on its columns?

Note that Polars does not have lazy columns. If you need to operate on columns as part of
a dataframe operation, you should use expressions - but if you need to extract a single
column, you need to ensure that you start with an eager `DataFrame`. To do that, you need
to pass `eager_only=True` to `nw.from_native`.

## Example 1: filter based on a column's values

This can stay lazy, so we just use `nw.from_native` and expressions:

```python exec="1" source="above" session="ex1"
import narwhals as nw

@nw.narwhalify
def my_func(df):
    return df.filter(nw.col('a') > 0)
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex1"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="ex1"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars (lazy)"
    ```python exec="true" source="material-block" result="python" session="ex1"
    import polars as pl

    df = pl.LazyFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df).collect())
    ```

## Example 2: multiply a column's values by a constant

Let's write a dataframe-agnostic function which multiplies the values in column
`'a'` by 2. This can also stay lazy, and can use expressions:

```python exec="1" source="above" session="ex2"
import narwhals as nw

@nw.narwhalify
def my_func(df):
    return df.with_columns(nw.col('a')*2)
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex2"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="ex2"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars (lazy)"
    ```python exec="true" source="material-block" result="python" session="ex2"
    import polars as pl

    df = pl.LazyFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df).collect())
    ```

Note that column `'a'` was overwritten. If we had wanted to add a new column called `'c'` containing column `'a'`'s
values multiplied by 2, we could have used `Expr.alias`:

```python exec="1" source="above" session="ex2.1"
import narwhals as nw

@nw.narwhalify
def my_func(df):
    return df.with_columns((nw.col('a')*2).alias('c'))
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex2.1"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="ex2.1"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars (lazy)"
    ```python exec="true" source="material-block" result="python" session="ex2.1"
    import polars as pl

    df = pl.LazyFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df).collect())
    ```

## Example 3: finding the mean of a column as a scalar

Now, we want to find the mean of column `'a'`, and we need it as a Python scalar.
This means that computation cannot stay lazy - it must execute!
Therefore, we'll pass `eager_only=True` to `nw.from_native`, and then, instead
of using expressions, we'll extract a `Series`.

```python exec="1" source="above" session="ex2"
import narwhals as nw

def my_func(df):
    df_s = nw.from_native(df, eager_only=True)
    return df_s['a'].mean()
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex2"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="ex2"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

Note that, this time, we couldn't use `@nw.narwhalify`, as the final step in
our function wasn't `nw.to_native`, so we had to explicitly use `nw.from_native`
as the first step. In general, we recommend using the decorator where possible,
as it looks a lot cleaner, and only using `nw.from_native` / `nw.to_native` explicitly
when you need them.
