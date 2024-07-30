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

```python exec="1"
def col_a(df):
    return [df.loc[:, "a"]]


def col_a_plus_1(df):
    return [x + 1 for x in col_a(df)]
```

Expressions can return multiple Series - for example, `nw.col('a', 'b')` translates to:

```python exec="1"
def col_a_b(df):
    return [df.loc[:, "a"], df.loc[:, "b"]]
```

Expressions can also take multiple columns as input - for example, `nw.sum_horizontal('a', 'b')`
translates to:

```python exec="1"
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
from narwhals.utils import parse_version

pn = PandasLikeNamespace(
    implementation=Implementation.PANDAS,
    backend_version=parse_version(pd.__version__),
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
underlying pandas objects via `PandasLikeDataFrame._native_dataframe` and `PandasLikeSeries._native_series`.

```python exec="1" result="python" session="pandas_impl" source="above"
import narwhals as nw
from narwhals._pandas_like.namespace import PandasLikeNamespace
from narwhals._pandas_like.utils import Implementation
from narwhals._pandas_like.dataframe import PandasLikeDataFrame
from narwhals.utils import parse_version
import pandas as pd

pn = PandasLikeNamespace(
    implementation=Implementation.PANDAS,
    backend_version=parse_version(pd.__version__),
)

df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df = PandasLikeDataFrame(
    df_pd,
    implementation=Implementation.PANDAS,
    backend_version=parse_version(pd.__version__),
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

Other implementations are similar to the above: their define their own Narwhals-compliant
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

End user works with narwhals apis, it won't use directly the native dataframe APIS,
so narwhals will take care itself of translating
the calls done to the narwhals api to calls to the dataframe wrapper
( `PandasLikeNamespace`, `PandasLikeDataFrame` or `PandasLikeExpr`) which then
forwards the calls to the native implementation. Translation of native narwhals
apis to the dataframe wrapper will usually happen via the namespace.
Usually you will find the namespace referred as
as `plx` which stands for *Polars compliant namespace for Library X*

That usually happens in a few different ways:

- `narwhals.DataFrame` -> `PandasLikeDataFrame` is available via the
  `narwhals.DataFrame._compliant_frame` attribute of the Dataframe.
  For example in case of a `pyarrow` we would have:
  ```python-console
  >>> nwdf = nw.from_native(table)
  >>> type(nwdf)
  <class 'narwhals.stable.v1.DataFrame'>
  >>> nwdf._compliant_frame
  <narwhals._arrow.dataframe.ArrowDataFrame object at 0xe9a1e3f3b050>
  ```
- `narwhals.Expr` -> `PandasLikeExpr` happens via calling `Expr._call` on the
  target namespace. For example in case of `pyarrow` we would have:
  ```python-console
  >>> type(nwdf)
  <class 'narwhals.stable.v1.DataFrame'>
  >>> nw.col("b").len()._call(nwdf.__narwhals_namespace__())
  ArrowExpr(depth=1, function_name=col->len, root_names=['b'], output_names=['b']
  ```
- `narwhals.Series` -> `PandasLikeSeries` happens via the
  `narwhals.Series._compliant_series` attribute of the Series.
  For example, in case of `pyarrow` we would have:
  ```python-console
  >>> nwdf = nw.from_native(table)
  >>> type(nwdf)
  <class 'narwhals.stable.v1.DataFrame'>
  >>> nwseries = nwdf["b"]
  >>> type(colseries)
  <class 'narwhals.stable.v1.Series'>
  >>> colseries._compliant_series
  <narwhals._arrow.series.ArrowSeries object at 0xe913f911d550>
  ```

After a compliant object is returned, the user will keep using and perform
additional operations on the compliant object itself, as the compliant
object implements the same API as the narwhals objects and is accepted
as an argument by all narwhals functions.

For example `nw.col("b")._call(nwdf.__narwhals_namespace__())` on a
`pyarrow.Table` will return an `ArrowExpr`:

```python-console
ArrowExpr(depth=0, function_name=col, root_names=['b'], output_names=['b']
```

but invoking `narwhals.Dataframe.select` on it will work and return
a new `narwhals.Dataframe` as if we passed a `narwhals.Expr` itself:

```python-console
>>> nwexpr = nw.col("b")
>>> type(nwexpr)
<class 'narwhals.stable.v1.Expr'>
>>> nwr = nwdf.select(nw.col("b"))
>>> type(nwr)
<class 'narwhals.stable.v1.DataFrame'>
>>> arrowexpr = nw.col("b")._call(nwdf.__narwhals_namespace__())
>>> type(arrowexpr)
<class 'narwhals._arrow.expr.ArrowExpr'>
>>> nwr = nwdf.select(arrowexpr)
>>> type(nwr)
<class 'narwhals.stable.v1.DataFrame'>
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
