from __future__ import annotations

# mypy: disable-error-code="misc"
# NOTE: Needs to be disabled as `mypy` reports the diagnostic twice, with one not attributed to a line number
# Sadly there's no way to disable  *just* the variance inference part for `function: FunctionT_co`
from typing import TYPE_CHECKING, Any, Generic, Protocol

from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._expr_ir import ExprIR
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._nodes import nodes
from narwhals._plan.typing import (
    FunctionT,
    FunctionT_co,
    HorizontalT_co,
    RangeT_co,
    Seq,
    StructT_co,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import type_check_only

    from typing_extensions import Self, TypeAlias

    from narwhals._plan import _parameters as params
    from narwhals._plan._expansion import Expander
    from narwhals._plan.compliant.typing import Ctx, FrameT_contra as FrameT, R_co
    from narwhals._plan.expressions.functions import MapBatches  # noqa: F401
    from narwhals._plan.schema import FrozenSchema
    from narwhals._typing_compat import TypeVar
    from narwhals.dtypes import DType

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
renamed = DispatcherOptions.renamed

Incomplete: TypeAlias = Any


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
    """

    __slots__ = ("function", "input")
    input: Seq[ExprIR] = nodes()
    function: FunctionT_co

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

    def dispatch(self: Self, ctx: Ctx[FrameT, R_co], frame: FrameT, name: str) -> R_co:
        return self.function.__expr_ir_dispatch__(self, ctx, frame, name)

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

    if TYPE_CHECKING:

        @property
        @type_check_only
        def __associated_function__(self: FunctionExpr[FunctionT]) -> type[FunctionT]:
            """**Type checking only!**, see `_AssociateParams` doc for details."""
            return self.function.__class__

        @property
        def parameters(self: _AssociateParams[ParamsT_co]) -> ParamsT_co:
            return self.__associated_function__.__function_parameters__

        def dispatch_args(
            self: _AssociateParams[_DispatchArgs[R_co, ArgsR_co]],
            ctx: Ctx[FrameT, R_co],
            frame: FrameT,
            name: str,
        ) -> ArgsR_co:
            """Call `ExprIR.dispatch` on all expression arguments to this function."""
            return self.parameters.dispatch_args(self, ctx, frame, name)
    else:

        @property
        def parameters(self) -> params.Parameters:
            return self.function.__function_parameters__

        def dispatch_args(
            self, ctx: Ctx[FrameT, R_co], frame: FrameT, name: str
        ) -> Seq[R_co]:
            return self.function.__function_parameters__.dispatch_args(
                self, ctx, frame, name
            )


class AnonymousExpr(FunctionExpr["MapBatches"], dispatch=renamed("map_batches")):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L158-L166."""

    @property
    def flags(self) -> FunctionFlags:
        return self.function.flags

    def dispatch(self: Self, ctx: Ctx[FrameT, R_co], frame: FrameT, name: str) -> R_co:
        return self.__expr_ir_dispatch__(self, ctx, frame, name)

    def resolve_dtype(self, schema: FrozenSchema) -> DType:  # pragma: no cover
        if dtype := self.function.return_dtype:
            return dtype
        return super().resolve_dtype(schema)


class HorizontalExpr(FunctionExpr[HorizontalT_co]):
    iter_expand = ExprIR.iter_expand


class RangeExpr(FunctionExpr[RangeT_co]):
    """E.g. `int_range(...)`."""

    def __repr__(self) -> str:
        return f"{self.function!r}({list(self.input)!r})"


class StructExpr(FunctionExpr[StructT_co]):
    """E.g. `col("a").struct.field(...)`.

    Requires special handling during expression expansion.
    """

    def needs_expansion(self) -> bool:
        return self.function.needs_expansion or super().needs_expansion()

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self


if TYPE_CHECKING:
    # NOTE: Associated type magic
    ParamsT_co = TypeVar(
        "ParamsT_co",
        bound="params.Parameters | _DispatchArgs[Any, Seq[Any]]",
        default=params.Parameters,
        covariant=True,
    )
    ArgsR_co = TypeVar("ArgsR_co", bound="Seq[Any]", covariant=True)

    class _HasParams(Protocol[ParamsT_co]):
        @property
        def __function_parameters__(self) -> ParamsT_co: ...

    class _AssociateParams(Protocol[ParamsT_co]):
        """Magic to yoink out the associated `Parameters` type from `FunctionExpr[Function]`.

        Adapts an idea from `scipy-stubs` ([1], [2]) for resolving this conundrum:

            class Function[P: Parameters]:
                # The real thing is an associated type (not a generic)
                __function_parameters_actual__: ClassVar[Parameters]

                # If we try to make it generic ...
                __function_parameters_bad__: ClassVar[P] # "ClassVar" type cannot include type variables

                # Fixes ^, but requires making `Function` generic
                # & somehow handling that in `FunctionExpr`
                @property
                def __function_parameters_better__(self) -> P: ...


            class FunctionExpr[F: Function[Parameters]]:
                # We can't write this as `F[P]` (higher-kinded type)
                function: F

                # Since `Function` isn't generic, this accessor would hide the associated type
                @property
                def parameters_actual(self) -> Parameters:
                    # But if we're *outside*, this long chain would work, e.g.
                    #   FunctionExpr[FillNull].function.__function_parameters_actual__  # "Binary"
                    return self.function.__function_parameters_actual__

                # Might work if `Function` was generic, but doesn't work well with `FunctionExpr` subclasses
                # & scales poorly to how many `Function` subclasses there are
                @property
                def parameters_better[P: Parameters](self: FunctionExpr[Function[P]]) -> P:
                    return self.function.__function_parameters_better__

        [1]: https://github.com/scipy/scipy-stubs/blob/2d6cca6a5e6ee21b6be7008c6c773dd8f12723fb/scipy-stubs/sparse/_base.pyi#L156-L171
        [2]: https://github.com/scipy/scipy-stubs/blob/85bea68bad3924f306586585d46fd2f9ff0d53d4/scipy-stubs/sparse/_typing.pyi#L30-L46
        """

        @property
        def __associated_function__(self) -> _HasParams[ParamsT_co]: ...
        @property
        def parameters(self: _AssociateParams[ParamsT_co]) -> ParamsT_co: ...

    class _DispatchArgs(Protocol[R_co, ArgsR_co]):  # pyright: ignore[reportInvalidTypeVarUse]
        """Magic to pull out the **return `tuple` length** of `Parameters.dispatch_args`.

        Where `ExprIR.dispatch` has the signature:

            Callable[[Ctx[FrameT, R_co], FrameT, str], R_co]

        A reasonable signature for `dispatch_args` would be:

            Callable[[Ctx[FrameT, R_co], FrameT, str], tuple[R_co, ...]]

        But for the majority of functions, we have a more precise return type:

            tuple[R_co]              # Unary
            tuple[R_co, R_co]        # Binary
            tuple[R_co, R_co, R_co]  # Ternary

        This handles exposing those types as `ArgsR_co`, without needing to wait for
        `TypeVarTuple(bound=...)` to be specified ([1], [2]).



        ## Notes
        A downside is that this messes with the variance inference for `R_co` in `Ctx[FrameT, R_co]`.

        AFAIK, the type system doesn't support a way to express:
          `ArgsR_co` has a bound of `tuple[R_co, ...]`

        If that detail could be seen, then it would be inferred as ~~contravariant~~ covariant.


        [1]: https://github.com/python/typing/pull/2215
        [2]: https://github.com/python/cpython/pull/148212
        """

        def dispatch_args(
            self,
            node: FunctionExpr | Incomplete,
            ctx: Ctx[FrameT, R_co],
            frame: FrameT,
            name: str,
        ) -> ArgsR_co: ...
