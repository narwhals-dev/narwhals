# API Reference

Anything documented in the API reference is intended to work consistently among
supported backends.

For example:
```python
import narwhals as nw

df.with_columns(
    a_mean = nw.col('a').mean(),
    a_std = nw.col('a').std(),
)
```
is supported, as `DataFrame.with_columns`, `narwhals.col`, `Expr.mean`, and `Expr.std` are
all documented in the API reference.

However,
```python
import narwhals as nw

df.with_columns(
    a_ewm_mean = nw.col('a').ewm_mean(alpha=.7),
)
```
is not - `Expr.ewm_mean` only appears in the Polars API reference, but not in the Narwhals
one.

In general, you should expect any fundamental dataframe operation to be supported - if
one that you need is not, please do open a feature request!
