# Quick start

## Prerequisites

Please start by following the [installation instructions](installation.md).

Then, please install the following:

- [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [Polars](https://pola-rs.github.io/polars/user-guide/installation/)

## Simple example

Create a Python file `t.py` with the following content:

```python exec="1" source="above" session="quickstart" result="python"
import pandas as pd
import polars as pl
import narwhals as nw


def my_function(df_any):
    df = nw.from_native(df_any)
    column_names = df.columns
    return column_names


df_pandas = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df_polars = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

print('pandas output')
print(my_function(df_pandas))
print('Polars output')
print(my_function(df_polars))
```

If you run `python t.py` then your output should look like the above. This is the simplest possible example of a dataframe-agnostic
function - as we'll soon see, we can do much more advanced things.
Let's learn about what you just did, and what Narwhals can do for you!

Note: these examples are only using pandas and Polars. Please see the following to find the [supported libriaries](extending.md).
