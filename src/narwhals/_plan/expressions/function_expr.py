"""`FunctionExpr` and friends.

## Implementation Notes
The default for implementing new expressions should be to add a new `Function`.
It becomes an `ExprIR` when wrapping it via `FunctionExpr`.

This follows `polars`' lead, which simplifies any decision-making for us.

ATOW, [the remaining variants] that would *not* be a new `Function` are:
- `Element` (`pl.element`)
- `DataTypeFunction` (`pl.{dtype_of,self_dtype,struct_with_fields}`)
- `Gather` (`pl.Expr.{gather,get}`)
- `Explode` (`pl.Expr.{explode,list.explode`)
- `Rolling` (`pl.Expr.rolling`)
- `Slice` (`pl.Expr.slice`)
- `Field` (`pl.field`)
- `Eval` (`pl.Expr.list.eval`)
- `StructEval` (`pl.Expr.struct.with_fields`)

Tip:
    If it doesn't look like one of those, it is probably a function

[the remaining variants]: https://github.com/pola-rs/polars/blob/346a793589efd552a6c10c857e0f0434f7e9a7d4/crates/polars-plan/src/dsl/expr/mod.rs#L98-L224
"""

from __future__ import annotations

# mypy: disable-error-code="misc"
# NOTE: Needs to be disabled as `mypy` reports the diagnostic twice, with one not attributed to a line number
# Sadly there's no way to disable  *just* the variance inference part for `function: FunctionT_co`
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, overload

from narwhals._plan._dispatch import FunctionExprDispatch
from narwhals._plan._expr_ir import ExprIR
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._nodes import nodes
from narwhals._plan.typing import (
    FunctionT_co,
    HorizontalT_co,
    Seq,
    Seq1,
    Seq2,
    Seq3,
    StructT_co,
)
from narwhals._utils import unstable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from narwhals._plan._expansion import Expander
    from narwhals._plan._function import (
        BinaryFunction,
        Function,
        TernaryFunction,
        UnaryFunction,
    )
    from narwhals._plan.compliant import typing as ct
    from narwhals._plan.expressions.functions import AsStruct, MapBatches  # noqa: F401
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType, Struct


class FunctionExpr(ExprIR, Generic[FunctionT_co]):
    """An expression wrapping a function and it's arguments.

    By count, the majority of expressions are implemented using this guy or some flavor of it.

    Arguments:
        args: Expression arguments to the function.
        function: The function to apply.

    Note:
        Represents `args` successfully binding to `function`, when constructed via
        `function.to_function_expr(*args)`.

    See Also:
        `narwhals._plan._function.Function`

    Examples:
        Typically, you'll create `FunctionExpr`s indirectly when calling methods on `Expr`:
        >>> import narwhals._plan as nw
        >>> fill_null = nw.col("a").fill_null("b")
        >>> print(fill_null._ir)
        FunctionExpr(args=[Col(name='a'), Lit(dtype=String, value='b')], function=FillNull())

        But for *"fun"*, let's do that manually to get a feel for how it works:
        >>> from narwhals._plan import expressions as ir
        >>> from narwhals._plan.expressions import functions as F
        >>> column = ir.col("a")
        >>> literal = ir.lit("b")
        >>> fill_null_ir = F.FillNull().to_function_expr(column, literal)
        >>> fill_null_ir == fill_null._ir
        True

        `to_function_expr(*args)` will validate that we meet constraints defined by the function:
        >>> F.FillNull().to_function_expr(column, literal, ir.lit("woops"))
        Traceback (most recent call last):
        TypeError: Expected 2 inputs for `fill_null()`, got 3:
          col('a')
          lit('b')
          lit('woops')

        And we can use constraints to reject the kind of input too:
        >>> nw.col("a").max().drop_nulls()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        InvalidOperationError: Cannot use `drop_nulls()` on aggregated expression `col('a').max()`.

        We can inspect these rules using `explain`:
        >>> print(nw.col("a").drop_nulls()._ir.explain(format="long"))
        FunctionExpr[DropNulls]
          Unary(Constraint.DEFAULT)
            col('a')
          FunctionFlags.ROW_SEPARABLE

        As we validate *only* via `to_function_expr` - we can cheaply rewrite our inputs later:
        >>> from narwhals import Int64
        >>> from narwhals._plan._expansion import Expander
        >>> expr = nw.nth(0, -1).fill_null(0)
        >>> expr._ir
        ncs.by_index([0, -1]).fill_null([lit(0)])

        >>> ctx = Expander({"a": Int64(), "b": Int64(), "c": Int64()})
        >>> list(expr._ir.iter_expand(ctx))
        [col('a').fill_null([lit(0)]), col('c').fill_null([lit(0)])]
    """

    __slots__ = ("args", "function")
    args: Seq[ExprIR] = nodes()
    function: FunctionT_co

    __expr_ir_dispatch__: ClassVar[FunctionExprDispatch[Self]] = (
        FunctionExprDispatch.root("FunctionExpr")
    )

    @property
    def flags(self) -> FunctionFlags:
        """Defines properties of the wrapped function.

        Flags tell us how a function transforms the shape of it's input:

            ┌───────────────────┬───────────────┬───────────┐
            │ Flag              ┆ Input         ┆ Output    │
            ╞═══════════════════╪═══════════════╪═══════════╡
            │ AGGREGATION       ┆ Column        ┆ Scalar    │
            │ ROW_SEPARABLE     ┆ Column        ┆ Unknown   │
            │ LENGTH_PRESERVING ┆ Column/Scalar ┆ Preserved │
            │ ELEMENTWISE       ┆ Column/Scalar ┆ Preserved │
            └───────────────────┴───────────────┴───────────┘

        And that's the main nugget we can use to answer the question:
        > Is the function valid *here*?

        See `FunctionFlags` for examples.
        """
        return self.function.__function_flags__

    def is_scalar(self) -> bool:
        return FunctionFlags.AGGREGATION in self.flags

    def is_length_preserving(self) -> bool:
        # NOTE: upstream is `... and all(e.is_length_preserving() for e in self.args)`
        # -but says it's overly conservative.
        # That won't make sense here as this is pre-expansion
        # https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-stream/src/physical_plan/lower_expr.rs#L364-L374
        return self.flags.is_length_preserving()

    def changes_length(self) -> bool:
        return self.flags.changes_length()

    def __repr__(self) -> str:
        return self.function.__expr_ir_repr__(self)

    def resolve_dtype(self, schema: FrozenSchema) -> DType:
        # NOTE: Supported on many functions, but there are important gaps.
        # - Requires [#3396]:
        #   - `{max,min,sum}_horizontal`
        #   - `coalesce`
        #   - `replace_strict(..., dtype=None)`
        #   - `mean_horizontal`   (partial)
        #   - `fill_null(value)`  (partial)
        # - Unlikely to ever be supported:
        #   - `map_batches(..., dtype=None)`
        # [#3396]: https://github.com/narwhals-dev/narwhals/pull/3396
        return self.function.resolve_dtype(self, schema)

    def _pre_undo_aliases(self, schema: FrozenSchema, /) -> ExprIR:
        return self.function._pre_undo_aliases(self, schema)

    def iter_expand(self, ctx: Expander, /) -> Iterator[ExprIR]:
        input_root, *non_root = self.args
        children = tuple(ctx.only(self, child) for child in non_root) if non_root else ()
        for root in input_root.iter_expand(ctx):
            yield self.__replace__(args=(root, *children))

    def dispatch_arg(
        self: FunctionExpr[UnaryFunction],
        ctx: ct.Caller[ct.E, ct.SC],
        frame: ct.FrameAny,
        name: str,
    ) -> ct.E | ct.SC:
        """Dispatch the **only** expression argument to this function.

        Important:
            Exclusive to `UnaryFunction`

        Arguments:
            ctx: An instance that implements `CompliantColumn`.
            frame: A`Compliant*Frame` that shares the same backend as `ctx`.
            name: Output column name, which will typically have originated from `NamedIR.name`.

        Note:
            See `Caller` for how `ctx` differs from `CompliantExpr`.
        """
        node = self.args[0]
        return node.__expr_ir_dispatch__(node, ctx, frame, name)

    @overload
    def dispatch_args(
        self: FunctionExpr[UnaryFunction],
        ctx: ct.Caller[ct.E, ct.SC],
        frame: ct.FrameAny,
        name: str,
    ) -> Seq1[ct.E | ct.SC]: ...
    @overload
    def dispatch_args(
        self: FunctionExpr[BinaryFunction],
        ctx: ct.Caller[ct.E, ct.SC],
        frame: ct.FrameAny,
        name: str,
    ) -> Seq2[ct.E | ct.SC]: ...
    @overload
    def dispatch_args(
        self: FunctionExpr[TernaryFunction],
        ctx: ct.Caller[ct.E, ct.SC],
        frame: ct.FrameAny,
        name: str,
    ) -> Seq3[ct.E | ct.SC]: ...
    @overload
    def dispatch_args(
        self: FunctionExpr[Function],
        ctx: ct.Caller[ct.E, ct.SC],
        frame: ct.FrameAny,
        name: str,
    ) -> Seq[ct.E | ct.SC]: ...
    def dispatch_args(
        self, ctx: ct.Caller[ct.E, ct.SC], frame: ct.FrameAny, name: str
    ) -> Seq[ct.E | ct.SC]:
        """Dispatch **all** expression arguments to this function.

        Arguments:
            ctx: An instance that implements `CompliantColumn`.
            frame: A`Compliant*Frame` that shares the same backend as `ctx`.
            name: Output column name, which will typically have originated from `NamedIR.name`.

        Note:
            See `Caller` for how `ctx` differs from `CompliantExpr`.
        """
        return self.function.__function_parameters__.dispatch_args(self, ctx, frame, name)

    # TODO @dangotbanned: Figure out why this docstring didn't give coverage
    def explain(self, *, format: Literal["short", "long"] = "short") -> str:
        """Create a rich string representation of the expression.

        >>> import narwhals._plan as nw
        >>> a = nw.col("a")
        >>> print(a.shift(5)._ir.explain())
        FunctionExpr[Shift(n=5)]
          Unary(DEFAULT)
            col('a')
          LENGTH_PRESERVING

        >>> print(nw.int_range(nw.col("a").max())._ir.explain(format="long"))
        FunctionExpr[IntRange(step=1, dtype=Int64)]
          Binary(Constraint.SCALAR, Constraint.SCALAR)
            lit(0)
            col('a').max()
          FunctionFlags.DEFAULT
        """
        nl, parens = "\n", "()"
        indent = " " * 2
        f = self.function
        flags = self.flags
        return (
            f"{type(self).__name__}[{str(f).removesuffix(parens)}]"
            f"{nl}{indent}{f.__function_parameters__.explain(format=format)}"
            f"{nl}{nl.join(f'{indent * 2}{e!r}' for e in self.args)}"
            f"{nl}{indent}{flags if format == 'short' else f'{type(flags).__name__}.{flags.name}'}"
        )


class AnonymousExpr(FunctionExpr["MapBatches"]):
    """A user-defined function expression.

    Represents `map_batches`, but could later be adapted to support [`map_elements`].

    [`map_elements`]: https://github.com/narwhals-dev/narwhals/issues/3512
    """

    @property
    def flags(self) -> FunctionFlags:
        # NOTE: Why another `FunctionExpr` subclass?
        # - Every other `Function` has it's flags defined in `type[Function].__dict__`
        # - `MapBatches` haccepts these hints from the caller and stores on the instance
        #   so a common property at a higher level avoids a `__slots__` conflict
        return self.function.flags


class HorizontalExpr(FunctionExpr[HorizontalT_co]):
    """An expression that applies a function *across* columns.

    Special cases of [fold] or [reduce].

    ## Examples
    Horizontal functions use different semantics when expanding selectors.

    Say we have the following schema:
    >>> from tests.plan.utils import Frame
    >>> import narwhals._plan as nw

    >>> df = Frame.from_names("a", "b", "c")
    >>> dict(df.schema)
    {'a': Int64, 'b': Int64, 'c': Int64}

    We expand multiple inputs into a single output:
    >>> before = nw.sum_horizontal(nw.all())
    >>> (reduced,) = df.project(before)
    >>> before._ir
    ncs.all().sum_horizontal()
    >>> reduced
    a=col('a').sum_horizontal([col('b'), col('c')])

    Whereas the more common form of expansion produces multiple outputs:
    >>> before = nw.all().clip("b")
    >>> before._ir
    ncs.all().clip_lower([col('b')])
    >>> df.project(before)  # doctest: +NORMALIZE_WHITESPACE
    (a=col('a').clip_lower([col('b')]),
     b=col('b').clip_lower([col('b')]),
     c=col('c').clip_lower([col('b')]))

    [fold]: https://docs.pola.rs/user-guide/expressions/folds/
    [reduce]: https://mathspp.com/blog/pydonts/the-power-of-reduce
    """

    iter_expand = ExprIR.iter_expand


# TODO @dangotbanned: Rename to hint at selecting from vs `AsStructExpr` which is creating a new struct
class StructExpr(FunctionExpr[StructT_co]):
    """An expression that applies a function to a struct column.

    Note:
        Requires special handling during expression expansion.
    """

    def needs_expansion(self) -> bool:
        return self.function.needs_expansion or super().needs_expansion()

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self


@unstable
class AsStructExpr(HorizontalExpr["AsStruct"]):
    r"""The result of resolving `nw.struct(...)`.

    ## Examples
    >>> import narwhals as nw
    >>> import narwhals._plan as nwp
    >>> from tests.plan.utils import Frame
    >>> frame = Frame.from_mapping({"a": nw.Int64(), "b": nw.String(), "c": nw.Boolean()})
    >>> expr = nwp.struct(nwp.col("a").alias("a_1"), nwp.nth(1).name.suffix("_2")).alias(
    ...     "outer"
    ... )

    `struct` is unique as it has two contexts:
    >>> expr._ir
    struct(col('a').alias('a_1'), ncs.by_index([1]).name.suffix('_2')).alias('outer')

    We start in a similar place to other variadic functions:
    >>> print(type(expr._ir.expr).__name__)
    HorizontalExpr

    But after expansion, we need to handle the inner and outer contexts independently:
    >>> named_ir = frame.project(expr)[0]
    >>> named_ir
    outer=struct(col('a'), col('b'))

    Our outer context resolves into a different class:
    >>> resolved = named_ir.expr
    >>> print(type(resolved).__name__)
    AsStructExpr

    And the output names of the fields are encoded into the dtype:
    >>> print(*zip(resolved.args, resolved.dtype.fields), sep="\n")
    (col('a'), Field('a_1', Int64))
    (col('b'), Field('b_2', String))
    """

    __slots__ = ("dtype",)
    dtype: Struct
    """The resolved struct dtype."""

    def resolve_dtype(self, schema: FrozenSchema) -> Struct:
        return self.dtype
