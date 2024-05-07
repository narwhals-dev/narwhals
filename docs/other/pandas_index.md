# What about the pandas Index?

There are two types of pandas users:

- The ones who make full use of the Index's power.
- The `.reset_index(drop=True)` ones, who would rather not think about the Index.

Narwhals aims to accommodate both!

- If you'd rather not think about the Index, then don't
  worry: it's not part of the Narwhals public API, and you'll never have to worry about
  resetting the index or about pandas doing funky index alignment for you.
- If you want your library to cater to Index powerusers who would be very angry if you reset
  their beautiful Index on their behalf, then don't worry: Narwhals makes certain promises
  with regards to the Index.

Let's learn about what Narwhals promises.

## 1. Narwhals will preserve your index for dataframe operations

```python exec="1" source="above" session="ex1"
import narwhals as nw

def my_func(df_any):
    df = nw.from_native(df_any)
    df = df.with_columns(a_plus_one=nw.col('a')+1)
    return nw.to_native(df)
```

Let's start with a dataframe with an Index with values `[7, 8, 9]`.

```python exec="true" source="material-block" result="python" session="ex1"
import pandas as pd

df = pd.DataFrame({'a': [2, 1, 3], 'b': [3, 5, -3]}, index=[7, 8, 9])
print(my_func(df))
```

Note how the result still has the original index - Narwhals did not modify
it.

## 2. Index alignment follows the left-hand-rule

pandas automatically aligns indices for users. For example:

```python exec="1" source="above" session="ex2"
import pandas as pd

df_pd = pd.DataFrame({'a': [2, 1, 3], 'b': [4, 5, 6]})
s_pd = df_pd['a'].sort_values()
df_pd['a_sorted'] = s_pd
```
Reading the code, you might expect that `'a_sorted'` will contain the
values `[1, 2, 3]`.

**However**, here's what actually happens:
```python exec="1" source="material-block" session="ex2" result="python"
print(df_pd)
```
In other words, pandas' index alignment undid the `sort_values` operation!

Narwhals, on the other hand, preserves the index of the left-hand-side argument.
Everything else will be inserted positionally, just like Polars would do:

```python exec="1" source="material-block" session="ex2" result="python"
import narwhals as nw

df = nw.from_native(df_pd)
s = nw.from_native(s_pd, allow_series=True)
df = df.with_columns(a_sorted=s.sort())
print(nw.to_native(df))
```

If you keep these two rules in mind, then Narwhals will both help you avoid
Index-related surprises whilst letting you preserve the Index for the subset
of your users who consciously make great use of it.
