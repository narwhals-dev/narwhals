# Upgrading from `stable.v1` to `stable.v2`

`narwhals.stable.v1` is maintained indefinitely, so this upgrade is optional. What it
buys you is the features which only exist in the newer API; what it costs is a minimum
requirement of `narwhals>=2.0` for your users. See
[Perfect backwards compatibility policy](../backcompat.md) for what the stable API
promises.

## Step 1: require `narwhals>=2.0`

`narwhals.stable.v2` first shipped in Narwhals 2.0, so bump your dependency accordingly:

```toml
# pyproject.toml
dependencies = ["narwhals>=2.0"]
```

## Step 2: swap the imports

Every submodule your code imports needs the same bump:

| `stable.v1` | `stable.v2` |
| --- | --- |
| `import narwhals.stable.v1 as nw` | `import narwhals.stable.v2 as nw` |
| `from narwhals.stable.v1.typing import FrameT` | `from narwhals.stable.v2.typing import FrameT` |
| `from narwhals.stable.v1 import dependencies` | `from narwhals.stable.v2 import dependencies` |
| `from narwhals.stable.v1 import dtypes` | `from narwhals.stable.v2 import dtypes` |
| `from narwhals.stable.v1 import selectors` | `from narwhals.stable.v2 import selectors` |

## Step 3: work through the differences

`stable.v2` matches the main namespace, so everything listed under
[`main` vs `stable.v1` differences](../backcompat.md#main-vs-stablev1-differences) is
something you may need to change.

### `strict` was replaced by `pass_through`

```py
# stable.v1
nw.from_native(df, strict=False)

# stable.v2
nw.from_native(df, pass_through=True)
```

The same applies to `to_native` and `narwhalify`. If you're on `narwhals>=1.13`, you can
make this change before upgrading, since `pass_through` works in both namespaces.

### `native_namespace` was replaced by `backend`

```py
# stable.v1
nw.from_dict(data, native_namespace=pd)

# stable.v2
nw.from_dict(data, backend=pd)
```

`backend` also accepts a string (`"pandas"`) or an `nw.Implementation` member.

### `any_horizontal` and `all_horizontal` require `ignore_nulls`

In `stable.v1` the argument defaults to `False` (Kleene logic). In `stable.v2` you have to
say which behaviour you want:

```python exec="yes" source="material-block" result="python" session="how-to-stable-v2"
import narwhals.stable.v2 as nw
import polars as pl

df = nw.from_native(pl.DataFrame({"a": [True, False], "b": [None, None]}))
print(df.select(nw.any_horizontal("a", "b", ignore_nulls=False)).to_native())
```

`ignore_nulls=False` keeps the `stable.v1` behaviour, and is unsupported by pandas backed
by classic NumPy dtypes. [Boolean logic](../concepts/boolean.md) covers what the two
options mean.

### `LazyFrame.with_row_index` requires `order_by`

Adding a row index is an order-dependent operation, and lazy backends have no inherent
row order, so `stable.v2` makes you name the ordering:

```py
# stable.v1
lf.with_row_index("idx")

# stable.v2
lf.with_row_index("idx", order_by="date")
```

[Order dependence](../concepts/order_dependence.md) explains why.

### Order-dependent expressions are gone from `LazyFrame`

`Expr.head`, `Expr.tail`, `Expr.gather_every`, `Expr.sample`, `Expr.arg_true` and
`Expr.sort` no longer exist in `stable.v2`. The frame-level methods (`DataFrame.head`,
`DataFrame.sort`, ...) are unaffected.

`Expr.arg_min` and `Expr.arg_max` are also gone. `Series.arg_min` and `Series.arg_max`
remain.

### `get_level`, `is_expr` and `InvalidIntoExprError`

- `nw.get_level` no longer exists. `stable.v2` has no `'interchange'` level, and
  `from_native` gives back either a `DataFrame` (eager, full API) or a `LazyFrame` (lazy
  subset), so an `isinstance` check answers the same question.
- `nw.is_expr` no longer exists. Use `isinstance(obj, nw.Expr)`.
- `nw.InvalidIntoExprError` is no longer a top-level name. Reach it through the
  `exceptions` namespace instead:

    ```python exec="yes" source="material-block" result="python" session="how-to-stable-v2"
    print(nw.exceptions.InvalidIntoExprError.__name__)
    ```

### DuckDB and Ibis inputs are lazy

In `stable.v1`, passing a `duckdb.DuckDBPyRelation` or an `ibis.Table` to `from_native`
returned an interchange-level `DataFrame` which only supported `.schema`. In `stable.v2`
you get a full `LazyFrame`.

`eager_or_interchange_only` has been removed from `from_native` and `narwhalify` along
with it. If you were using it to reject lazy inputs, use `eager_only=True`.

### Dtype inference

- pandas' ordered categoricals map to `nw.Enum` rather than `nw.Categorical`, and
  `nw.Enum` requires its `categories` at instantiation.
- An empty or all-null object-dtype pandas Series is inferred as `String`, not `Object`.
- `Datetime` and `Duration` hash using `time_unit` and `time_zone`, so
  `nw.Datetime("us") in {nw.Datetime}` is `False` in `stable.v2`. Compare with `==`
  instead, which behaves the same in both namespaces:

    ```python exec="yes" source="material-block" result="python" session="how-to-stable-v2"
    print(nw.Datetime("us") == nw.Datetime)
    ```

### `Series` is generic

`Series` is generic in its native series, so `s.to_native()` is inferred as
`polars.Series` rather than `Any`. This can surface type errors which `stable.v1` hid, so
the upgrade is worth running your type checker over.

## Step 4: test against every backend you support

Several of the changes above are backend-specific: dtype inference affects pandas, order
dependence affects lazy backends. [Testing dataframe-agnostic code](testing.md) shows how
to parametrise a test suite over backends.
