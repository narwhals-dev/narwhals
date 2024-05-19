# How it works

## Theory

You might think that Narwhals runs on underwater unicorn magic. However, this section exists
to reassure you that there's no such thing. There's only one rule you need to understand in
order to make sense of Narwhals:

> **An expression is a function from a DataFrame to a sequence of Series.**

For example, `nw.col('a')` means "given a dataframe `df`, give me the Series `'a'` from `df`".
Translating this to pandas syntax, we get:

```python
def col_a(df):
    return [df.loc[:, 'a']]
```

Let's step up the complexity. How about `nw.col('a')+1`? We already know what the
`nw.col('a')` part looks like, so we just need to add `1` to each of its outputs:

```python
def col_a(df):
    return [df.loc[:, 'a']]

def col_a_plus_1(df):
    return [x+1 for x in col_a(df)]
```

Expressions can return multiple Series - for example, `nw.col('a', 'b')` translates to:

```python
def col_a_b(df):
    return [df.loc[:, 'a'], df.loc[:, 'b']]
```

Expressions can also take multiple columns as input - for example, `nw.sum_horizontal('a', 'b')`
translates to:

```python
def sum_horizontal_a_b(df):
    return [df.loc[:, 'a'] + df.loc[:, 'b']]
```

Note that although an expression may have multiple columns as input,
those columns must all have been derived from the same dataframe.

By itself, an expression doesn't produce a value. It only produces a value once you give it to a
DataFrame context. What happens to the value(s) it produces depends on which context you hand
it to:

- `DataFrame.select`: produce a DataFrame with only the result of the given expression
- `DataFrame.with_columns`: produce a DataFrame like the current one, but also with the result of
  the given expression
- `DataFrame.filter`: evaluate the given expression, and if it only returns a single Series, then
  only keep rows where the result is `True`.

Now let's turn our attention to the implementation.

## Polars implementation

For Polars, Narwhals just "passes everything through". For example consider the following:
```python
import polars as pl
import narwhals as nw

df_pl = pl.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
df = nw.from_native(df_pl)
df.select(nw.col('a')+1)
```

`nw.col('a')` produces a `narwhals.expression.Expr` object, which has a private `_call` method.
We can call `nw.col('a')._call(pl)`, then the result is actually `pl.col('a')`.

We then let Polars do its thing. Which is nice, but also not particularly interesting.
How about translating expressions to pandas? Well, it's
interesting to us, and you're still reading, so maybe it'll be interesting to you too.

## pandas implementation

When we called `nw.col('a')._call(pl)`, we got a Narwhals-compliant Polars namespace.
The pandas namespace (`pd`) isn't Narwhals-compliant, as the pandas API is very different
from Polars'. So...Narwhals implements a `PandasNamespace`, which includes the top-level
Polars functions included in the Narwhals API:

```python
import narwhals as nw
from narwhals._pandas_like.namespace import PandasNamespace

pn = PandasNamespace(implementation='pandas')
nw.col('a')._call(pn)
```
The result from the last line above is the same as we'd get from `pn.col('a')`, and it's
a `narwhals._pandas_like.expression.PandasExpr` object, which we'll call `PandasExpr` for
short.

`PandasExpr` also have a `_call` method - but this one expects a `PandasDataFrame` as input.
Recall from above that an expression is a function from a dataframe to a sequence of series.
The `_call` method gives us that function! Let's see it in action.

Note: the following examples uses `PandasDataFrame` and `PandasSeries`. These are wrappers
around pandas DataFrame and pandas Series, which are Narwhals-compliant. To get the native
pandas objects out from inside them, we access `PandasDataFrame._dataframe` and `PandasSeries._series`.

```python
import narwhals as nw
from narwhals._pandas_like.namespace import PandasNamespace
from narwhals._pandas_like.dataframe import PandasDataFrame
import pandas as pd

pn = PandasNamespace(implementation='pandas')

df_pd = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
df = PandasDataFrame(df_pd, implementation='pandas')
expression = pn.col('a') + 1
result = expression._call(df)
print([x._series for x in result])
```
The first (and only) Series to be output is:
```
0    2
1    3
2    4
Name: a, dtype: int64
```

So indeed, our expression did what it said on the tin - it took some dataframe, took
column 'a', and added 1 to it.

If you search for `def register_expression_call`, you'll see that that's all
expressions do in Narwhals - they just keep rigorously applying the definition of
expression.

It may look like there should be significant overhead to doing it this way - but really,
it's just a few Python calls which get unwinded. From timing tests I've done, there's
no detectable difference - in fact, because the Narwhals API guards against misusing the
pandas API, it's likely that running pandas via Narwhals will in general be more efficient
than running pandas directly.

Further attempts at demistifying Narwhals, refactoring code so it's clearer, and explaining
this section better are 110% welcome.

