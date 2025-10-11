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
from narwhals._utils import parse_version, Version

pn = PandasLikeNamespace(
    implementation=Implementation.PANDAS,
    version=Version.MAIN,
)
print(nw.col("a")._to_compliant_expr(pn))
```

The result from the last line above is the same as we'd get from `pn.col('a')`, and it's
a `narwhals._pandas_like.expr.PandasLikeExpr` object, which we'll call `PandasLikeExpr` for
short.

`PandasLikeExpr` has a `_call` method which expects a `PandasLikeDataFrame` as input.
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
from narwhals._utils import parse_version, Version
import pandas as pd

pn = PandasLikeNamespace(
    implementation=Implementation.PANDAS,
    version=Version.MAIN,
)

df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df = PandasLikeDataFrame(
    df_pd,
    implementation=Implementation.PANDAS,
    version=Version.MAIN,
    validate_column_names=True,
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
- `narwhals.Expr`: call the `._to_compliant_expr` method, and pass to it the Narwhals-compliant namespace associated with
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
from narwhals._utils import parse_version, Version
import pandas as pd

pn = PandasLikeNamespace(
    implementation=Implementation.PANDAS,
    version=Version.MAIN,
)

df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df = nw.from_native(df_pd)
df.select(nw.col("a") + 1)
```

The first thing `narwhals.DataFrame.select` does is to parse each input expression to end up with a compliant expression for the given
backend, and it does so by passing a Narwhals-compliant namespace to `nw.Expr._to_compliant_expr`:

```python exec="1" result="python" session="pandas_api_mapping" source="above"
pn = PandasLikeNamespace(
    implementation=Implementation.PANDAS,
    version=Version.MAIN,
)
expr = (nw.col("a") + 1)._to_compliant_expr(pn)
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

## Nodes

If we have a Narwhals expression, we can look at the operations which make it up by accessing `_nodes`:

```python exec="1" result="python" session="pandas_impl" source="above"
import narwhals as nw

expr = nw.col("a").abs().std(ddof=1) + nw.col("b")
print(expr._nodes)
```

Each node represents an operation. Here, we have 4 operations:

1. Given some dataframe, select column `'a'`.
2. Take its absolute value.
3. Take its standard deviation, with `ddof=1`.
4. Sum column `'b'`.

Let's take a look at a couple of these nodes. Let's start with the third one:

```python exec="1" result="python" session="pandas_impl" source="above"
print(expr._nodes[2].as_dict())
```

This tells us a few things:

- We're performing an aggregation.
- The name of the function is `'std'`. This will be looked up in the compliant object.
- It takes keyword arguments `ddof=1`.
- We'll look at `exprs`, `str_as_lit`, and `allow_multi_output` later.

In order for the evaluation to succeed, then `PandasLikeExpr` must have a `std` method defined
on it, which takes a `ddof` argument. And this is what the `CompliantExpr` Protocol is for: so
long as a backend's implementation complies with the protocol, then Narwhals will be able to
unpack a `ExprNode` and turn it into a valid call.

Let's take a look at the fourth node:

```python exec="1" result="python" session="pandas_impl" source="above"
print(expr._nodes[3].as_dict())
```

Note how now, the `exprs` attribute is populated. Indeed, we are summing another expression: `col('b')`.
The `exprs` parameter holds arguments which are either expressions, or should be interpreted as expressions.
The `str_as_lit` parameter tells us whether string literals should be interpreted as literals (e.g. `lit('foo')`)
or columns (e.g. `col('foo')`). Finally `allow_multi_output` tells us whether multi-outuput expressions
(more on this in the next section) are allowed to appear in `exprs`.

Note that the expression in `exprs` also has its own nodes:

```python exec="1" result="python" session="pandas_impl" source="above"
print(expr._nodes[3].exprs[0]._nodes)
```

It's nodes all the way down!

## Expression Metadata

Let's try printing out some compliant expressions' metadata to see what it shows us:

```python exec="1" result="python" session="pandas_impl" source="above"
import narwhals as nw

print(nw.col("a")._to_compliant_expr(pn)._metadata)
print(nw.col("a").mean()._to_compliant_expr(pn)._metadata)
print(nw.col("a").mean().over("b")._to_compliant_expr(pn)._metadata)
```

This section is all about making sense of what that all means, what the rules are, and what it enables.

Here's a brief description of each piece of metadata:

- `expansion_kind`: How and whether the expression expands to multiple outputs.
  This can be one of:

    - `ExpansionKind.SINGLE`: Only produces a single output. For example, `nw.col('a')`.
    - `ExpansionKind.MULTI_NAMED`: Produces multiple outputs whose names can be
      determined statically, for example `nw.col('a', 'b')`.
    - `ExpansionKind.MULTI_UNNAMED`: Produces multiple outputs whose names depend
      on the input dataframe. For example, `nw.nth(0, 1)` or `nw.selectors.numeric()`.

- `has_windows`: Whether the expression already contains an `over(...)` statement.
- `n_orderable_ops`: How many order-dependent operations the expression contains.
  
    Examples:

    - `nw.col('a')` contains 0 orderable operations.
    - `nw.col('a').diff()` contains 1 orderable operation.
    - `nw.col('a').diff().shift()` contains 2 orderable operation.

- `is_elementwise`: Whether it preserves length and operates on each row independently
  of the rows around it (e.g. `abs`, `is_null`, `round`, ...).
- `preserves_length`: Whether the output of the expression is the same length as
  the dataframe it gets evaluated on.
- `is_scalar_like`: Whether the output of the expression is always length-1.
- `is_literal`: Whether the expression doesn't depend on any column but instead
  only on literal values, like `nw.lit(1)`.
- `nodes`: List of operations which this expression applies when evaluated.

### Chaining

Say we have `expr.expr_method()`. How does `expr`'s `ExprMetadata` change?
This depends on `expr_method`. Details can be found in `narwhals/_expression_parsing`,
in the `ExprMetadata.with_*` methods.

#### Binary operations (e.g. `nw.col('a') + nw.col('b')`)

How do expression kinds change under binary operations? For example,
if we do `expr1 + expr2`, then what can we say about the output kind?
The rules are:

- If one changes the input length (e.g. `Expr.drop_nulls`), then:

    - if the other is scalar-like, then the output also changes length.
    - else, we raise an error.

- If one preserves length and the other is scalar-like, then the output
  preserves length (because of broadcasting).
- If one is scalar-like but not literal and the other is scalar-like,
  the output is scalar-like but not literal.

For n-ary operations such as `nw.sum_horizontal`, the above logic is
extended across inputs. For example, `nw.sum_horizontal(expr1, expr2, expr3)`
is `LITERAL` if all of `expr1`, `expr2`, and `expr3` are.

### "You open a window to another window to another window to another window"

When working with `DataFrame`s, row order is well-defined, as the dataframes
are assumed to be eager and in-memory. Therefore, `n_orderable_ops` is
disregarded.

When working with `LazyFrame`s, on the other hand, row order is undefined.
Therefore, when evaluating an expression, `n_orderable_ops` must be exactly
zero - if it's not, it means that the expression depends on physical row order,
which is not allowed for `LazyFrame`s. The way that `n_orderable_ops` can change
is:

- Orderable window functions like `diff` and `rolling_mean` increase `n_orderable_ops`
  by 1.
- If an orderable window function is immediately followed by `over(order_by=...)`,
  then `n_orderable_ops` is decreased by 1. This is the only way that
  `n_orderable_ops` can decrease.

## Broadcasting

When performing comparisons between columns and aggregations or scalars, we operate as if the
aggregation or scalar was broadcasted to the length of the whole column. For example, if we
have a dataframe with values `{'a': [1, 2, 3]}` and do `nw.col('a') - nw.col('a').mean()`,
then each value from column `'a'` will have its mean subtracted from it, and we will end up
with values `[-1, 0, 1]`.

Different libraries do broadcasting differently. SQL-like libraries require an empty window
function for expressions (e.g. `a - sum(a) over ()`), Polars does its own broadcasting of
length-1 Series, and pandas does its own broadcasting of scalars.

Narwhals triggers a broadcast in these situations:

- In `select` when some values preserve length and others don't, e.g.
  `df.select('a', nw.col('b').mean())`.
- In `with_columns`, all new columns get broadcasted to the length of the dataframe.
- In n-ary operations between expressions, such as `nw.col('a') + nw.col('a').mean()`.

Each backend is then responsible for doing its own broadcasting, as defined in each
`CompliantExpr.broadcast` method.

## Elementwise push-down

SQL is picky about `over` operations. For example:

- `sum(a) over (partition by b)` is valid.
- `sum(abs(a)) over (partition by b)` is valid.
- `abs(sum(a)) over (partition by b)` is not valid.

In Polars, however, all three of

- `pl.col('a').sum().over('b')` is valid.
- `pl.col('a').abs().sum().over('b')` is valid.
- `pl.col('a').sum().abs().over('b')` is valid.

How can we retain Polars' level of flexibility when translating to SQL engines?

The answer is: by rewriting expressions. Specifically, we push down `over` nodes past elementwise ones.
To see this, let's try printing the Narwhals equivalent of the last expression above (the one that SQL rejects):

```python exec="1" result="python" session="pushdown" source="above"
import narwhals as nw

print(nw.col("a").sum().abs().over("b"))
```

Note how Narwhals automatically inserted the `over` operation _before_ the `abs` one. In other words, instead
of doing

- `sum` -> `abs` -> `over`

it did

- `sum` -> `over` -> `abs`

thus allowing the expression to be valid for SQL engines!

This is what we refer to as "pushing down `over` nodes". The idea is:

- Elementwise operations operate row-by-row and don't depend on the rows around them.
- An `over` node partitions or orders a computation.
- Therefore, an elementwise operation followed by an `over` operation is the same
  as doing the `over` operation followed by that same elementwise operation!

Note that the pushdown also applies to any arguments to the elementwise operation.
For example, if we have

```python
(nw.col("a").sum() + nw.col("b").sum()).over("c")
```

then `+` is an elementwise operation and so can be swapped with `over`. We just need
to take care to apply the `over` operation to all the arguments of `+`, so that we
end up with

```python
nw.col("a").sum().over("c") + nw.col("b").sum().over("c")
```

!!! info
    In general, query optimisation is out-of-scope for Narwhals. We consider this
    expression rewrite acceptable because:
      - It's simple.
      - It allows us to evaluate operations which otherwise wouldn't be allowed for certain backends.
