# TODO @dangotbanned: Make use of module doc for impl details

from __future__ import annotations

# mypy: disable-error-code="misc"
# NOTE: Needs to be disabled as `mypy` reports the diagnostic twice, with one not attributed to a line number
# Sadly there's no way to disable  *just* the variance inference part for `function: FunctionT_co`
from typing import TYPE_CHECKING, Generic, overload

from narwhals._plan._expr_ir import ExprIR
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._nodes import nodes
from narwhals._plan.typing import FunctionT_co, HorizontalT_co, RangeT_co, Seq, StructT_co

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
    from narwhals._plan._parameters import Parameters
    from narwhals._plan.compliant import typing as ct
    from narwhals._plan.expressions.functions import MapBatches  # noqa: F401
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType


# TODO @dangotbanned: How painful will a rename for `input` -> `args` be?
# TODO @dangotbanned: Docs should complement `Function`
# - The two are very tightly coupled
class FunctionExpr(ExprIR, Generic[FunctionT_co]):
    """An expression wrapping a function and it's arguments.

    Arguments:
        input: Expression arguments to the function.
        function: The function to apply, which may contain non-expression arguments.

    ## What to doc?
    - What things are functions?
        - (Mostly) non-aggregating functions
        - Data type namespaces
        - Horizontal functions
        - Range functions
        - UDFs
    - So what is a `FunctionExpr` then?
        - ...
    - What behaviors can we describe with this type?
    - Most operations (by count) are represented using this guy or some flavor of it
    """

    __slots__ = ("function", "input")
    input: Seq[ExprIR] = nodes()
    function: FunctionT_co

    # TODO @dangotbanned: (Docs) Maybe just duplicate `Function.__function_flags__`
    @property
    def flags(self) -> FunctionFlags:
        return self.function.__function_flags__

    def is_scalar(self) -> bool:
        return FunctionFlags.AGGREGATION in self.flags

    def is_length_preserving(self) -> bool:
        # NOTE: upstream is `... and all(e.is_length_preserving() for e in self.input)`
        # -but says it's overly conservative.
        # That won't make sense here as this is pre-expansion
        # https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-stream/src/physical_plan/lower_expr.rs#L364-L374
        return self.flags.is_length_preserving()

    def changes_length(self) -> bool:
        return self.flags.changes_length()

    def __repr__(self) -> str:
        if self.input:
            first = self.input[0]
            if len(self.input) >= 2:
                return f"{first!r}.{self.function!r}({list(self.input[1:])!r})"
            return f"{first!r}.{self.function!r}()"
        return f"{self.function!r}()"

    def dispatch(
        self: Self,
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co],
        frame: ct.Frame,
        name: str,
    ) -> ct.ET_co | ct.ST_co:
        return self.function.__expr_ir_dispatch__(self, ctx, frame, name)

    # TODO @dangotbanned: Convert docs into a comment
    def resolve_dtype(self, schema: FrozenSchema) -> DType:
        """NOTE: Supported on many functions, but there are important gaps.

        Requires `get_supertype`:
        - `{max,min,sum}_horizontal`
        - `coalesce`
        - `replace_strict(..., dtype=None)`

        Partially requires `get_supertype`:
        - `mean_horizontal`
        - `fill_null(value)`

        Unlikely to ever be supported:
        - `map_batches(..., dtype=None)`
        """
        return self.function.resolve_dtype(self, schema)

    def iter_expand(self, ctx: Expander, /) -> Iterator[ExprIR]:
        input_root, *non_root = self.input
        children = tuple(ctx.only(self, child) for child in non_root) if non_root else ()
        for root in input_root.iter_expand(ctx):
            yield self.__replace__(input=(root, *children))

    # TODO @dangotbanned: Either document this or remove it and update the `Parameters` doctests
    @property
    def parameters(self) -> Parameters:
        return self.function.__function_parameters__

    # TODO @dangotbanned: (Docs) see `ExprIR.dispatch`
    def dispatch_arg(
        self: FunctionExpr[UnaryFunction],
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co],
        frame: ct.Frame,
        name: str,
    ) -> ct.ET_co | ct.ST_co:
        """Call `ExprIR.dispatch` on the **only** expression argument to this function.

        Important:
            Exclusive to `Unary`
        """
        return self.input[0].dispatch(ctx, frame, name)

    # TODO @dangotbanned: (Docs) see `ExprIR.dispatch`
    @overload
    def dispatch_args(
        self: FunctionExpr[UnaryFunction],
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co],
        frame: ct.Frame,
        name: str,
    ) -> tuple[ct.ET_co | ct.ST_co]: ...
    @overload
    def dispatch_args(
        self: FunctionExpr[BinaryFunction],
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co],
        frame: ct.Frame,
        name: str,
    ) -> tuple[ct.ET_co | ct.ST_co, ct.ET_co | ct.ST_co]: ...
    @overload
    def dispatch_args(
        self: FunctionExpr[TernaryFunction],
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co],
        frame: ct.Frame,
        name: str,
    ) -> tuple[ct.ET_co | ct.ST_co, ct.ET_co | ct.ST_co, ct.ET_co | ct.ST_co]: ...
    @overload
    def dispatch_args(
        self: FunctionExpr[Function],
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co],
        frame: ct.Frame,
        name: str,
    ) -> Seq[ct.ET_co | ct.ST_co]: ...
    def dispatch_args(
        self,
        ctx: ct.DispatchScopeAny[ct.Frame, ct.ET_co, ct.ST_co],
        frame: ct.Frame,
        name: str,
    ) -> Seq[ct.ET_co | ct.ST_co]:
        """Call `ExprIR.dispatch` on **all** expression arguments to this function."""
        return self.function.__function_parameters__.dispatch_args(self, ctx, frame, name)


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


class RangeExpr(FunctionExpr[RangeT_co]):
    """E.g. `int_range(...)`."""

    def __repr__(self) -> str:
        # TODO @dangotbanned: (very low-priority) `Function` could take a format string
        # when subclassing instead of this
        # `dt.timestamp` and `struct.field` have some weirdness too
        return f"{self.function!r}({list(self.input)!r})"


class StructExpr(FunctionExpr[StructT_co]):
    """E.g. `col("a").struct.field(...)`.

    Requires special handling during expression expansion.
    """

    def needs_expansion(self) -> bool:
        return self.function.needs_expansion or super().needs_expansion()

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self
