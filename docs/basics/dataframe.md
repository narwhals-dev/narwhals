# DataFrame

To write a dataframe-agnostic function, the steps you'll want to follow are:

1. Initialise a Narwhals DataFrame or LazyFrame by passing your dataframe to `nw.from_native`.
   
    Note: if you need eager execution, use `nw.DataFrame` instead.

2. Express your logic using the subset of the Polars API supported by Narwhals.
3. If you need to return a dataframe to the user in its original library, call `narwhals.to_native`.

Let's try writing a simple example.

## Example 1: group-by and mean

Make a Python file `t.py` with the following content:
```python exec="1" source="above" session="df_ex1"
import narwhals as nw

def func(df):
    # 1. Create a Narwhals dataframe
    df_s = nw.from_native(df)
    # 2. Use the subset of the Polars API supported by Narwhals
    df_s = df_s.group_by('a').agg(nw.col('b').mean()).sort('a')
    # 3. Return a library from the user's original library
    return nw.to_native(df_s)
```
Let's try it out:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import pandas as pd

    df = pd.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df))
    ```

=== "Polars (eager)"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import polars as pl

    df = pl.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df))
    ```

=== "Polars (lazy)"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import polars as pl

    df = pl.LazyFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df).collect())
    ```
