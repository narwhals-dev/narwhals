# Why?

You may think that pandas, Polars, and all dataframe libraries are quite similar. But are they really?

For example, do the following produce the same output?

```python
import pandas as pd
import polars as pl

print(3 in pd.Series([1, 2, 3]))
print(3 in pl.Series([1, 2, 3]))
```

Try it out and see ;) Spoiler alert: they don't. pandas checks if `3` is in the index,
Polars checks if it's in the values.

How about
```python
df_left = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df_right = pd.DataFrame({'a': [1, 2, 3], 'c': [4, 5, 6]})
df_left.merge(df_right, left_on='b', right_on='c', how='left')
```
versus

```python
df_left = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df_right = pl.DataFrame({'a': [1, 2, 3], 'c': [4, 5, 6]})
df_left.join(df_right, left_on='b', right_on='c', how='left')
```

?

There are several such subtle difference between the libraries. Writing dataframe-agnostic code is hard!
But by having a unified, simple, and predictable API, you can focus on behaviour rather than on subtle
implementation differences.

Furthermore, both pandas and Polars frequently deprecate behaviour. Narwhals handles this for you by
testing against nightly builds of both libraries and handling backwards compatibility internally 
(so you don't have to!).
