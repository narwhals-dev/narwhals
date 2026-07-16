# Behind the scenes

## A rich representation for expressions

### Immutable

### ExprIR

### Function

### NamedIR

## Fine-grained protocols, which compose to provide useful defaults

## First-class scalars
[Why? - What about scalars?]: ../why.md#what-about-scalars
[*a function from a DataFrame to a sequence of Series*]: ../why.md#an-expression-is-a-function-from-a-dataframe-to-a-sequence-of-series

!!! tip "Learn more"

    For a deeper understanding of the problem, see [Why? - What about scalars?]

Representing scalar expressions as [*a function from a DataFrame to a sequence of Series*] added complexity to Narwhals. 

<!-- TODO @dangotbanned: Needs more intro-->

=== "CompliantScalar"

    We need to do two things here, and the first (destructuring `ir.Lit`) is the default implementation of `lit`.  
    A backend needs only to fill in `CompliantScalar.from_python`. How do we do that?

    ```py
    class CompliantScalar(CompliantColumn, Protocol):
        @classmethod                              # (1)!
        def lit(cls, node: ir.Lit[PythonLiteral], _frame: Any, name: str, /) -> Self:
            return cls.from_python(node.value, name, dtype=node.dtype)

        @classmethod
        def from_python(
            cls, value: PythonLiteral, name: str = "literal", /, *, dtype: IntoDType | None
        ) -> Self: ... 
    ```

    1.  `_frame` is unused (by default), and we don't need to repeat that in `ArrowScalar`.  
        I think this is an *improvement* but in the future would prefer a signature that supports declaring your dependencies like:  
        `#!py def lit(cls, node: ir.Lit, context: ContextT, /) -> Self: ...`

=== "ArrowScalar"

    [ArrowNamespace.lit]: https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_arrow/namespace.py#L67-L82
    `from_native` is here to show that there is no funny business going on!  
    `from_python` serves the same role as [ArrowNamespace.lit]. But here we create 0 functions per-call. [`pa.ChunkedArray`][pyarrow.ChunkedArray] 
    is nowhere to be seen because [`pa.Scalar`][pyarrow.Scalar] exists so we can use that instead.

    ```py
    class ArrowScalar(CompliantScalar):
        @classmethod
        def from_python(
            cls,
            value: PythonLiteral,
            name: str = "literal",
            /,
            *,
            dtype: IntoDType | None = None,
        ) -> Self:
            unknown = cls.version.dtypes.Unknown # (3)!
            dtype_pa = None if dtype == unknown else fn.dtype_native(dtype, cls.version)
            return cls.from_native(fn.lit(value, dtype_pa), name)

        @classmethod
        def from_native(cls, scalar: NativeScalar, name: str = "literal", /) -> Self: # (1)!
            obj = cls.__new__(cls)
            obj._evaluated = scalar
            obj._name = name # (2)!
            return obj
    ```

    1.  `"literal"` is always the default for creating a scalar expression.
    2.  Every expression **always** has exactly **one** name. 
    3.  See (https://github.com/narwhals-dev/narwhals/issues/2835).


=== "PolarsExpr"

    `from_python` is **nearly identical** to that of `ArrowScalar`. We implement `lit` because polars handles scalars for us.

    ```py
    class PolarsExpr(CompliantExpr):
        @classmethod
        def lit(cls, node: ir.Lit[PythonLiteral], _: Any, name: str, /) -> Self:
            return cls.from_python(node.value, name, dtype=node.dtype)

        @classmethod
        def from_python(
            cls,
            value: PythonLiteral,
            name: str = "literal",
            /,
            *,
            dtype: IntoDType | None = None,
        ) -> Self:
            unknown = cls.version.dtypes.Unknown
            dtype_pl = None if dtype == unknown else dtype_to_native(dtype, cls.version)
            return cls.from_native(fn.lit(value, dtype_pl), name)

        @classmethod
        def from_native(cls, native: pl.Expr, name: str = "", /) -> Self:
            obj = cls.__new__(cls)
            obj._native = native if not name else native.alias(name)
            return obj
    ```

`CompliantExpr` and `CompliantScalar` share `CompliantColumn` as a common base. This separation is represented in Polars with 
`Column::Series` and `Column::Scalar` as variants of the [`Column` enum](https://github.com/pola-rs/polars/blob/9c223540d8c6fffec1c9cdf532d11fca619c6e98/crates/polars-core/src/frame/column/mod.rs#L29-L44). 

Polars and PyArrow are not alone, all backends can represent the two concepts and all but Pandas do so natively:

| Backend | Expr/Series                                                                                              | Scalar                                                                                                                                                                    |
| ------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ibis    | [`ibis.ir.Column`](https://ibis-project.org/reference/expression-generic#ibis.expr.types.generic.Column) | [`ibis.ir.Scalar`](https://ibis-project.org/reference/expression-generic#ibis.expr.types.generic.Scalar)                                                                  |
| DuckDB  | [`duckdb.ColumnExpression`][]                                                                            | [`duckdb.ConstantExpression`][]                                                                                                                                           |
| PySpark | [`pyspark.sql.Column`][]                                                                                 | [`pyspark.sql.functions.lit`][]                                                                                                                                           |
| Dask    | [`dx.Series`][dask.dataframe.Series]                                                                     | [`dx.Scalar`](https://github.com/dask/dask/blob/a54329156cd17a68ef081c83fdc2409ddb423426/dask/dataframe/dask_expr/_collection.py#L4832-L4863)                             |
| Pandas  | [`pd.Series`][pandas.Series]                                                                             | [`np.generic`][numpy.generic], [`PythonLiteral`](https://github.com/narwhals-dev/narwhals/blob/bab7d107138eae80f54f61f7065a43ed7dde91b4/src/narwhals/typing.py#L276) [^1] |

[^1]: Honestly expected to see [`pa.Scalar`][pyarrow.Scalar] here, but didn't find a public API that returns one.

### `lit` in review
> 1 We **know** that the result of this expression will always have the name `"literal"`.

The semantics of this are slightly different in `narwhals._plan`, due to [`NamedIR`][narwhals._plan.expressions.NamedIR].  
What's important here is that by the time `CompliantScalar.lit` has been reached - we already have the final output column `name`.  

If we had the expression `lit(5)` [^2], then `name` is exactly [`Lit.name`][narwhals._plan.expressions.Lit.name]:

[^2]: which does not contain renaming operations

!!! note

    - `CompliantScalar.lit` does not repeat details that we can capture in [`Lit`][narwhals._plan.expressions.Lit]
    - `CompliantScalar.from_python` provides `"literal"` as a default, but accepts a `name` as you may need to [generate a new one](https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_arrow/dataframe.py#L435-L445).


> 2 We **know** that the result will always be a single column.

By expanding our expressions into [`NamedIR`][narwhals._plan.expressions.NamedIR] *before we we passed them to the backend* this was guaranteed for us.  

But what if you're *determined* to break the data model?

``` py
>>> import narwhals._plan as nwp
>>> from narwhals._plan import expressions as ir
>>> data = {"a": [12.1], "b": [42], "c": [4], "d": ["play"]}

>>> df = nwp.from_dict(data, backend="polars")
>>> df._compliant.select((ir.NamedIR("howdy", nwp.nth(3, 4).first()._ir),))
TypeError: 'ByIndex' should not appear at the compliant-level.
Make sure to expand all expressions first, got:
ncs.by_index([3, 4])
```

Every backend **including Polars** behaves the same, none need to handle multiple outputs because [`ExprIR` nodes define what is valid](https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_plan/_expr_ir.py#L687-L700)

> 3 We **know** that the column will be considered scalar (in broadcasting terms).

- If a backend needs to make this distinction? 
    - They implement `CompliantExpr` and `CompliantScalar` (this defines the behavior for scalars e.g. `ArrowScalar`)
- If a backend handles everything natively?
    - They implement `CompliantExpr` (while [declaring that as their scalar type](https://github.com/narwhals-dev/narwhals/blob/79573d9e83451c6448d16b64fc8c3de8e01282e5/src/narwhals/_plan/compliant/classes.py#L179-L195) e.g. `PolarsExpr`)

### Beyond `lit`
[much more]: https://github.com/narwhals-dev/narwhals/blob/expr-ir/docs/fluff-1/src/narwhals/_plan/compliant/scalar.py#L29-L163

`CompliantScalar.lit` is a constructor for `CompliantScalar`. But if you peek at `CompliantScalar`, [many more operations have default implementations] matching the behavior of Polars.

That means we do not need to eagerly [^3] check if expressions are scalar and reject them [^4] - because the behavior is defined to be consistent across backends.  
We can extend or limit the surface of what `CompliantScalar` permits (if needed), but the core idea is to be *more like* Polars and do **less work** work achieve it.

[^3]: Like [here](https://github.com/narwhals-dev/narwhals/blob/92f2e0008bd653a683d7300dd4cdc0e5816632cb/src/narwhals/_expression_parsing.py#L498) 
      and [here](https://github.com/narwhals-dev/narwhals/blob/92f2e0008bd653a683d7300dd4cdc0e5816632cb/src/narwhals/_expression_parsing.py#L517) 
      and [here](https://github.com/narwhals-dev/narwhals/blob/92f2e0008bd653a683d7300dd4cdc0e5816632cb/src/narwhals/_expression_parsing.py#L658)
[^4]: Where Polars does not


## Every backend is a plugin
