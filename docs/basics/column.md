# Column

In [dataframe.md](dataframe.md), you learned how to write a dataframe-agnostic function.

We only used DataFrame methods there - but what if we need to operate on its columns?

## Extracting a column


## Example 1: filter based on a column's values

```python exec="1" source="above" session="ex1"
import narwhals as nw

def my_func(df):
    df_s = nw.DataFrame(df)
    df_s = df_s.filter(nw.col('a') > 0)
    return nw.to_native(df_s)
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex1"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="ex1"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```


## Example 2: multiply a column's values by a constant

Let's write a dataframe-agnostic function which multiplies the values in column
`'a'` by 2.

```python exec="1" source="above" session="ex2"
import narwhals as nw

def my_func(df):
    df_s = nw.DataFrame(df)
    df_s = df_s.with_columns(nw.col('a')*2)
    return nw.to_native(df_s)
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex2"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="ex2"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

Note that column `'a'` was overwritten. If we had wanted to add a new column called `'c'` containing column `'a'`'s
values multiplied by 2, we could have used `Column.rename`:

```python exec="1" source="above" session="ex2.1"
import narwhals as nw

def my_func(df):
    df_s = nw.DataFrame(df)
    df_s = df_s.with_columns((nw.col('a')*2).alias('c'))
    return nw.to_native(df_s)
```

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="ex2.1"
    import pandas as pd

    df = pd.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="ex2.1"
    import polars as pl

    df = pl.DataFrame({'a': [-1, 1, 3], 'b': [3, 5, -3]})
    print(my_func(df))
    ```
