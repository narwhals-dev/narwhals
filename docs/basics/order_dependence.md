# Order-dependence

Narwhals has four main public classes:

- `Expr`: what gets created when you write `nw.col('a')`.
- `DataFrame`: in-memory, eager dataframe with a well-defined row order which
  is preserved across `with_columns` and `select` operations.
- `LazyFrame`: a dataframe which makes no assumptions about row-ordering. This
  allows it to be backed by SQL engines.
- `Series`: 1-dimensional in-memory structure with a defined row order. This is
  what you get if you extract a single column from a `DataFrame`.

Row order is important to think about when performing operations which rely on it,
such as:

- `diff`, `shift`.
- `cum_sum`, `cum_min`, ...
- `rolling_sum`, `rolling_min`, ...
- `is_first_distinct`, `is_last_distinct`.

When row-order is defined, as is the case for `DataFrame`, these operations pose
no issue.

```python exec="1" result="python" session="order_dependence" source="above"
import narwhals as nw
import pandas as pd

df_pd = pd.DataFrame({"a": [1, 3, 4], "i": [0, 1, 2]})
df = nw.from_native(df_pd)
print(df.with_columns(a_cum_sum=nw.col("a").cum_sum()))
```

When row order is undefined however, then these operations do not have a defined
result. To make them well-defined, you need to follow them with `over` in which
you specify `order_by`. For example:

- `nw.col('a').cum_sum()` can only be executed by a `DataFrame`.
- `nw.col('a').cum_sum().over(order_by="i")` can only be executed by either a `DataFrame`
  or a `LazyFrame`.

```python exec="1" result="python" session="order_dependence" source="above"
from sqlframe.duckdb import DuckDBSession

session = DuckDBSession()
sqlframe_df = session.createDataFrame(df_pd)
lf = nw.from_native(sqlframe_df)
result = lf.with_columns(a_cum_sum=nw.col("a").cum_sum().over(order_by="i"))
print(result)
print(result.collect("pandas"))
```

When writing an order-dependent function, if you want it to be executable by `LazyFrame`
(and not just `DataFrame`), make sure that all order-dependent expressions are followed
by `over` with `order_by` specified. If you forget to, don't worry, Narwhals will
give you a loud and clear error message.
