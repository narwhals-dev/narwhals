# TODO @dangotbanned: Make use of module doc for impl details

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
    from narwhals._plan.expressions.functions import MapBatches  # noqa: F401
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType


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
            Exclusive to `Unary`
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
        """Dispatch **all** expression arguments to this function."""
        return self.function.__function_parameters__.dispatch_args(self, ctx, frame, name)

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


# TODO @dangotbanned: (Docs) Add a note and point to `HorizontalFunction` explainer on expansion
class HorizontalExpr(FunctionExpr[HorizontalT_co]):
    iter_expand = ExprIR.iter_expand


# TODO @dangotbanned: Class doc
class StructExpr(FunctionExpr[StructT_co]):
    """E.g. `col("a").struct.field(...)`.

    Requires special handling during expression expansion.
    """

    def needs_expansion(self) -> bool:
        return self.function.needs_expansion or super().needs_expansion()

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self
