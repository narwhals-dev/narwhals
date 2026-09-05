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

## 1. Narwhals will preserve your index for common dataframe operations

```python exec="yes" source="above" session="ex1"
import narwhals as nw
from narwhals.typing import IntoFrameT


def my_func(df: IntoFrameT) -> IntoFrameT:
    df = nw.from_native(df)
    df = df.with_columns(a_plus_one=nw.col("a") + 1)
    return nw.to_native(df)
```

Let's start with a dataframe with an Index with values `[7, 8, 9]`.

```python exec="yes" source="material-block" result="python" session="ex1"
import pandas as pd

df = pd.DataFrame({"a": [2, 1, 3], "b": [3, 5, -3]}, index=[7, 8, 9])
print(my_func(df))
```

Note how the result still has the original index - Narwhals did not modify
it. Narwhals will preserve your original index for most common dataframe
operations. However, Narwhals will _not_ preserve the original index for
`DataFrame.group_by`, because there, overlapping index and column names
raise errors.

## 2. Index alignment follows the left-hand-rule

pandas automatically aligns indices for users. For example:

```python exec="yes" source="above" session="ex2"
import pandas as pd

df_pd = pd.DataFrame({"a": [2, 1, 3], "b": [4, 5, 6]})
s_pd = df_pd["a"].sort_values()
df_pd["a_sorted"] = s_pd
```

Reading the code, you might expect that `'a_sorted'` will contain the
values `[1, 2, 3]`.

**However**, here's what actually happens:

```python exec="yes" source="material-block" session="ex2" result="python"
print(df_pd)
```

In other words, pandas' index alignment undid the `sort_values` operation!

Narwhals, on the other hand, preserves the index of the left-hand-side argument.
Everything else will be inserted positionally, just like Polars would do:

```python exec="yes" source="material-block" session="ex2" result="python"
import narwhals as nw

df = nw.from_native(df_pd)
s = nw.from_native(s_pd, allow_series=True)
df = df.with_columns(a_sorted=s.sort())
print(nw.to_native(df))
```

## 3. Narwhals can copy an index from one object to another

`nw.maybe_get_index` returns the index of a pandas-like object, and
`nw.maybe_set_index` accepts whatever it returns. The two compose:

```python exec="yes" source="material-block" session="ex3" result="python"
import narwhals as nw
import pandas as pd

df = nw.from_native(pd.DataFrame({"a": [2, 1, 3]}, index=[7, 8, 9]))
result = nw.from_native(pd.DataFrame({"b": [4, 5, 6]}))
print(nw.to_native(nw.maybe_set_index(result, index=nw.maybe_get_index(df))))
```

For Polars, PyArrow and other non-pandas-like backends, `maybe_get_index` returns
`None` and `maybe_set_index` returns its input unchanged, so the same code runs
everywhere.

If you keep these rules in mind, then Narwhals will both help you avoid
Index-related surprises whilst letting you preserve the Index for the subset
of your users who consciously make great use of it.
