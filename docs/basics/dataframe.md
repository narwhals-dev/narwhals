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

@nw.narwhalify
def func(df):
    return df.select(
        a_sum=nw.col('a').sum(),
        a_mean=nw.col('a').mean(),
        a_std=nw.col('a').std(),
    )
```
Let's try it out:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import pandas as pd

    df = pd.DataFrame({'a': [1, 1, 2]})
    print(func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import polars as pl

    df = pl.DataFrame({'a': [1, 1, 2]})
    print(func(df))
    ```

=== "Polars (lazy)"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import polars as pl

    df = pl.LazyFrame({'a': [1, 1, 2]})
    print(func(df).collect())
    ```

Alternatively, we could have opted for the more explicit version:
```python
import narwhals as nw

def func(df_any):
    df = nw.from_native(df_any)
    df = df.select(
        a_sum=nw.col('a').sum(),
        a_mean=nw.col('a').mean(),
        a_std=nw.col('a').std(),
    )
    return nw.to_native(df)
```

In general, we think `@nw.narwhalify` is more legible, so we'll use that wherever possible.

## Example 2: group-by and mean

Just like in Polars, we can pass expressions to `GroupBy.agg`.
Make a Python file with the following content:
```python exec="1" source="above" session="df_ex2"
import narwhals as nw

@nw.narwhalify
def func(df):
    return df.group_by('a').agg(nw.col('b').mean()).sort('a')
```
Let's try it out:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="df_ex2"
    import pandas as pd

    df = pd.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="df_ex2"
    import polars as pl

    df = pl.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df))
    ```

=== "Polars (lazy)"
    ```python exec="true" source="material-block" result="python" session="df_ex2"
    import polars as pl

    df = pl.LazyFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df).collect())
    ```

## Example 3: horizontal sum

Expressions can be free-standing functions which accept other
expressions as inputs. For example, we can compute a horizontal
sum using `nw.sum_horizontal`.

Make a Python file with the following content:
```python exec="1" source="above" session="df_ex3"
import narwhals as nw

@nw.narwhalify
def func(df):
    return df.with_columns(a_plus_b=nw.sum_horizontal('a', 'b'))
```
Let's try it out:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="df_ex3"
    import pandas as pd

    df = pd.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="df_ex3"
    import polars as pl

    df = pl.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df))
    ```

=== "Polars (lazy)"
    ```python exec="true" source="material-block" result="python" session="df_ex3"
    import polars as pl

    df = pl.LazyFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df).collect())
    ```
