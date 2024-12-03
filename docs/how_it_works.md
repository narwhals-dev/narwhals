# How it works

## Theory

You might think that Narwhals runs on underwater unicorn magic. However, this section exists
to reassure you that there's no such thing. There's only one rule you need to understand in
order to make sense of Narwhals:

> **An expression is a function from a DataFrame to a sequence of Series.**

For example, `nw.col('a')` means "given a dataframe `df`, give me the Series `'a'` from `df`".
Translating this to pandas syntax, we get:

```python exec="1" source="above"
def col_a(df):
    return [df.loc[:, "a"]]
```

Let's step up the complexity. How about `nw.col('a')+1`? We already know what the
`nw.col('a')` part looks like, so we just need to add `1` to each of its outputs:

```python exec="1" source="above"
def col_a(df):
    return [df.loc[:, "a"]]


def col_a_plus_1(df):
    return [x + 1 for x in col_a(df)]
```

Expressions can return multiple Series - for example, `nw.col('a', 'b')` translates to:

```python exec="1" source="above"
def col_a_b(df):
    return [df.loc[:, "a"], df.loc[:, "b"]]
```

Expressions can also take multiple columns as input - for example, `nw.sum_horizontal('a', 'b')`
translates to:

```python exec="1" source="above"
def sum_horizontal_a_b(df):
    return [df.loc[:, "a"] + df.loc[:, "b"]]
```

Note that although an expression may have multiple columns as input,
those columns must all have been derived from the same dataframe. This last sentence was
quite important, you might want to re-read it to make sure it sunk in.

By itself, an expression doesn't produce a value. It only produces a value once you give it to a
DataFrame context. What happens to the value(s) it produces depends on which context you hand
it to:

- `DataFrame.select`: produce a DataFrame with only the result of the given expression
- `DataFrame.with_columns`: produce a DataFrame like the current one, but also with the result of
  the given expression
- `DataFrame.filter`: evaluate the given expression, and if it only returns a single Series, then
  only keep rows where the result is `True`.

Now let's turn our attention to the implementation.

## pandas implementation

The pandas namespace (`pd`) isn't Narwhals-compliant, as the pandas API is very different
from Polars'. So...Narwhals implements a `PandasLikeNamespace`, which includes the top-level
Polars functions included in the Narwhals API:

```python exec="1" source="above", result="python" session="pandas_impl"
import pandas as pd
import narwhals as nw
from narwhals._pandas_like.namespace import PandasLikeNamespace
from narwhals._pandas_like.utils import Implementation
from narwhals.utils import parse_version, Version

pn = PandasLikeNamespace(
    implementation=Implementation.PANDAS,
    backend_version=parse_version(pd.__version__),
    version=Version.MAIN,
)
print(nw.col("a")._call(pn))
```
The result from the last line above is the same as we'd get from `pn.col('a')`, and it's
a `narwhals._pandas_like.expr.PandasLikeExpr` object, which we'll call `PandasLikeExpr` for
short.

`PandasLikeExpr` also has a `_call` method - but this one expects a `PandasLikeDataFrame` as input.
Recall from above that an expression is a function from a dataframe to a sequence of series.
The `_call` method gives us that function! Let's see it in action.

Note: the following examples use `PandasLikeDataFrame` and `PandasLikeSeries`. These are backed
by actual `pandas.DataFrame`s and `pandas.Series` respectively and are Narwhals-compliant. We can access the 
underlying pandas objects via `PandasLikeDataFrame._native_frame` and `PandasLikeSeries._native_series`.

```python exec="1" result="python" session="pandas_impl" source="above"
import narwhals as nw
from narwhals._pandas_like.namespace import PandasLikeNamespace
from narwhals._pandas_like.utils import Implementation
from narwhals._pandas_like.dataframe import PandasLikeDataFrame
from narwhals.utils import parse_version, Version
import pandas as pd

pn = PandasLikeNamespace(
    implementation=Implementation.PANDAS,
    backend_version=parse_version(pd.__version__),
    version=Version.MAIN,
)

df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df = PandasLikeDataFrame(
    df_pd,
    implementation=Implementation.PANDAS,
    backend_version=parse_version(pd.__version__),
    version=Version.MAIN,
)
expression = pn.col("a") + 1
result = expression._call(df)
print(f"length of result: {len(result)}\n")
print("native series of first value of result: ")
print([x._native_series for x in result][0])
```

So indeed, our expression did what it said on the tin - it took some dataframe, took
column 'a', and added 1 to it.

If you search for `def reuse_series_implementation`, you'll see that that's all
expressions do in Narwhals - they just keep rigorously applying the definition of
expression.

It may look like there should be significant overhead to doing it this way - but really,
it's just a few Python calls which get unwinded. From timing tests I've done, there's
no detectable difference - in fact, because the Narwhals API guards against misusing the
pandas API, it's likely that running pandas via Narwhals will in general be more efficient
than running pandas directly.

Further attempts at demistifying Narwhals, refactoring code so it's clearer, and explaining
this section better are 110% welcome.

## Polars and other implementations

Other implementations are similar to the above: they define their own Narwhals-compliant
objects. So, all-in-all, there are a couple of layers here:

- `nw.DataFrame` is backed by a Narwhals-compliant Dataframe, such as:
    - `narwhals._pandas_like.dataframe.PandasLikeDataFrame`
    - `narwhals._arrow.dataframe.ArrowDataFrame`
    - `narwhals._polars.dataframe.PolarsDataFrame`
- each Narwhals-compliant DataFrame is backed by a native Dataframe, for example:
    - `narwhals._pandas_like.dataframe.PandasLikeDataFrame` is backed by a pandas DataFrame
    - `narwhals._arrow.dataframe.ArrowDataFrame` is backed by a PyArrow Table
    - `narwhals._polars.dataframe.PolarsDataFrame` is backed by a Polars DataFrame

Each implementation defines its own objects in subfolders such as `narwhals._pandas_like`,
`narwhals._arrow`, `narwhals._polars`, whereas the top-level modules such as `narwhals.dataframe`
and `narwhals.series` coordinate how to dispatch the Narwhals API to each backend.

## Mapping from API to implementations

If an end user executes some Narwhals code, such as

```python
df.select(nw.col("a") + 1)
```
then how does that get mapped to the underlying dataframe's native API? Let's walk through
this example to see.

Things generally go through a couple of layers:

- The user calls some top-level Narwhals API.
- The Narwhals API forwards the call to a Narwhals-compliant dataframe wrapper, such as
    - `PandasLikeDataFrame` / `ArrowDataFrame` / `PolarsDataFrame` / ...
    - `PandasLikeSeries` / `ArrowSeries` / `PolarsSeries` / ...
    - `PandasLikeExpr` / `ArrowExpr` / `PolarsExpr` / ...
- The dataframe wrapper forwards the call to the underlying library, e.g.:
    - `PandasLikeDataFrame` forwards the call to the underlying pandas/Modin/cuDF dataframe.
    - `ArrowDataFrame` forwards the call to the underlying PyArrow table.
    - `PolarsDataFrame` forwards the call to the underlying Polars DataFrame.

The way you access the Narwhals-compliant wrapper depends on the object:

- `narwhals.DataFrame` and `narwhals.LazyFrame`: use the `._compliant_frame` attribute.
- `narwhals.Series`: use the `._compliant_series` attribute.
- `narwhals.Expr`: call the `._call` method, and pass to it the Narwhals-compliant namespace associated with
  the given backend.

🛑 BUT WAIT! What's a Narwhals-compliant namespace?

Each backend is expected to implement a Narwhals-compliant
namespace (`PandasLikeNamespace`, `ArrowNamespace`, `PolarsNamespace`). These can be used to interact with the Narwhals-compliant
Dataframe and Series objects described above - let's work through the motivating example to see how.

```python exec="1" session="pandas_api_mapping" source="above"
import narwhals as nw
from narwhals._pandas_like.namespace import PandasLikeNamespace
from narwhals._pandas_like.utils import Implementation
from narwhals._pandas_like.dataframe import PandasLikeDataFrame
from narwhals.utils import parse_version, Version
import pandas as pd

pn = PandasLikeNamespace(
    implementation=Implementation.PANDAS,
    backend_version=parse_version(pd.__version__),
    version=Version.MAIN,
)

df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df = nw.from_native(df_pd)
df.select(nw.col("a") + 1)
```

The first thing `narwhals.DataFrame.select` does is to parse each input expression to end up with a compliant expression for the given
backend, and it does so by passing a Narwhals-compliant namespace to `nw.Expr._call`:

```python exec="1" result="python" session="pandas_api_mapping" source="above"
pn = PandasLikeNamespace(
    implementation=Implementation.PANDAS,
    backend_version=parse_version(pd.__version__),
    version=Version.MAIN,
)
expr = (nw.col("a") + 1)._call(pn)
print(expr)
```
If we then extract a Narwhals-compliant dataframe from `df` by
calling `._compliant_frame`, we get a `PandasLikeDataFrame` - and that's an object which we can pass `expr` to!

```python exec="1" session="pandas_api_mapping" source="above"
df_compliant = df._compliant_frame
result = df_compliant.select(expr)
```

We can then view the underlying pandas Dataframe which was produced by calling `._native_frame`:

```python exec="1" result="python" session="pandas_api_mapping" source="above"
print(result._native_frame)
```
which is the same as we'd have obtained by just using the Narwhals API directly:

```python exec="1" result="python" session="pandas_api_mapping" source="above"
print(nw.to_native(df.select(nw.col("a") + 1)))
```

## Group-by

Group-by is probably one of Polars' most significant innovations (on the syntax side) with respect
to pandas. We can write something like
```python
df: pl.DataFrame
df.group_by("a").agg((pl.col("c") > pl.col("b").mean()).max())
```
To do this in pandas, we need to either use `GroupBy.apply` (sloooow), or do some crazy manual
optimisations to get it to work.

In Narwhals, here's what we do:

- if somebody uses a simple group-by aggregation (e.g. `df.group_by('a').agg(nw.col('b').mean())`),
  then on the pandas side we translate it to
  ```python
  df: pd.DataFrame
  df.groupby("a").agg({"b": ["mean"]})
  ```
- if somebody passes a complex group-by aggregation, then we use `apply` and raise a `UserWarning`, warning
  users of the performance penalty and advising them to refactor their code so that the aggregation they perform
  ends up being a simple one.

In order to tell whether an aggregation is simple, Narwhals uses the private `_depth` attribute of `PandasLikeExpr`:

```python exec="1" result="python" session="pandas_impl" source="above"
print(pn.col("a").mean())
print((pn.col("a") + 1).mean())
print(pn.mean("a"))
```

For simple aggregations, Narwhals can just look at `_depth` and `function_name` and figure out
which (efficient) elementary operation this corresponds to in pandas.
