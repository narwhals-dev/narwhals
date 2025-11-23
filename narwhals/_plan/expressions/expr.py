"""Top-level `Expr` nodes."""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from narwhals._plan._expr_ir import ExprIR, SelectorIR
from narwhals._plan.common import replace
from narwhals._plan.exceptions import (
    function_expr_invalid_operation_error,
    over_order_by_names_error,
)
from narwhals._plan.expressions import selectors as cs
from narwhals._plan.options import ExprIROptions
from narwhals._plan.typing import (
    FunctionT_co,
    Ignored,
    LeftSelectorT,
    LeftT,
    LiteralT,
    OperatorT,
    RangeT_co,
    RightSelectorT,
    RightT,
    RollingT_co,
    SelectorOperatorT,
    SelectorT,
    Seq,
)
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import Self

    from narwhals._plan.compliant.typing import Ctx, FrameT_contra, R_co
    from narwhals._plan.expressions.functions import MapBatches  # noqa: F401
    from narwhals._plan.expressions.literal import LiteralValue
    from narwhals._plan.expressions.window import Window
    from narwhals._plan.options import FunctionOptions, SortMultipleOptions, SortOptions
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType

__all__ = [
    "Alias",
    "AnonymousExpr",
    "BinaryExpr",
    "BinarySelector",
    "Cast",
    "Column",
    "Filter",
    "FunctionExpr",
    "Len",
    "Literal",
    "RollingExpr",
    "RootSelector",
    "SelectorIR",
    "Sort",
    "SortBy",
    "TernaryExpr",
    "WindowExpr",
    "col",
]


def col(name: str, /) -> Column:
    return Column(name=name)


class Alias(ExprIR, child=("expr",), config=ExprIROptions.no_dispatch()):
    __slots__ = ("expr", "name")
    expr: ExprIR
    name: str

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.alias({self.name!r})"


class Column(ExprIR, config=ExprIROptions.namespaced("col")):
    __slots__ = ("name",)
    name: str

    def __repr__(self) -> str:
        return f"col({self.name!r})"

    def to_selector_ir(self) -> RootSelector:
        return cs.ByName.from_name(self.name).to_selector_ir()


class Literal(ExprIR, t.Generic[LiteralT], config=ExprIROptions.namespaced("lit")):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L81."""

    __slots__ = ("value",)
    value: LiteralValue[LiteralT]

    @property
    def is_scalar(self) -> bool:
        return self.value.is_scalar

    @property
    def dtype(self) -> DType:
        return self.value.dtype

    @property
    def name(self) -> str:
        return self.value.name

    def __repr__(self) -> str:
        return f"lit({self.value!r})"

    def unwrap(self) -> LiteralT:
        return self.value.unwrap()


class _BinaryOp(ExprIR, t.Generic[LeftT, OperatorT, RightT]):
    __slots__ = ("left", "op", "right")
    left: LeftT
    op: OperatorT
    right: RightT

    @property
    def is_scalar(self) -> bool:
        return self.left.is_scalar and self.right.is_scalar

    def __repr__(self) -> str:
        return f"[({self.left!r}) {self.op!r} ({self.right!r})]"


class BinaryExpr(
    _BinaryOp[LeftT, OperatorT, RightT],
    t.Generic[LeftT, OperatorT, RightT],
    child=("left", "right"),
):
    """Application of two exprs via an `Operator`."""

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.left.iter_output_name()


class Cast(ExprIR, child=("expr",)):
    __slots__ = ("expr", "dtype")  # noqa: RUF023
    expr: ExprIR
    dtype: DType

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.cast({self.dtype!r})"

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_output_name()


class Sort(ExprIR, child=("expr",)):
    __slots__ = ("expr", "options")
    expr: ExprIR
    options: SortOptions

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        direction = "desc" if self.options.descending else "asc"
        return f"{self.expr!r}.sort({direction})"

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_output_name()


class SortBy(ExprIR, child=("expr", "by")):
    """https://github.com/narwhals-dev/narwhals/issues/2534."""

    __slots__ = ("expr", "by", "options")  # noqa: RUF023
    expr: ExprIR
    by: Seq[ExprIR]
    options: SortMultipleOptions

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.sort_by(by={self.by!r}, options={self.options!r})"

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_output_name()


# mypy: disable-error-code="misc"
class FunctionExpr(ExprIR, t.Generic[FunctionT_co], child=("input",)):
    """**Representing `Expr::Function`**.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L114-L120
    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/function_expr/mod.rs#L123
    """

    __slots__ = ("function", "input", "options")
    input: Seq[ExprIR]
    # NOTE: mypy being mypy - the top error can't be silenced 🤦‍♂️
    # narwhals/_plan/expr.py: error: Cannot use a covariant type variable as a parameter  [misc]
    # narwhals/_plan/expr.py:272:15: error: Cannot use a covariant type variable as a parameter  [misc]
    #         function: FunctionT_co  # noqa: ERA001
    #                   ^
    # Found 2 errors in 1 file (checked 476 source files)
    function: FunctionT_co
    """Operation applied to each element of `input`."""

    options: FunctionOptions
    """Combined flags from chained operations."""

    @property
    def is_scalar(self) -> bool:
        return self.function.is_scalar

    def __repr__(self) -> str:
        if self.input:
            first = self.input[0]
            if len(self.input) >= 2:
                return f"{first!r}.{self.function!r}({list(self.input[1:])!r})"
            return f"{first!r}.{self.function!r}()"
        return f"{self.function!r}()"

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        """When we have multiple inputs, we want the name of the left-most expression.

        For expr:

            col("c").alias("x").fill_null(50)

        We are interested in the name which comes from the root:

            FunctionExpr(..., [Alias(..., name='...'), Literal(...), ...])
            #                  ^^^^^            ^^^
        """
        for e in self.input[:1]:
            yield from e.iter_output_name()
        # NOTE: Covering the empty case doesn't make sense without implementing `FunctionFlags.ALLOW_EMPTY_INPUTS`
        # https://github.com/pola-rs/polars/blob/df69276daf5d195c8feb71eef82cbe9804e0f47f/crates/polars-plan/src/plans/options.rs#L106-L107
        return  # pragma: no cover

    # NOTE: Interacting badly with `pyright` synthesizing the `__replace__` signature
    if not TYPE_CHECKING:

        def __init__(
            self,
            *,
            input: Seq[ExprIR],  # noqa: A002
            function: FunctionT_co,
            options: FunctionOptions,
            **kwds: t.Any,
        ) -> None:
            parent = input[0]
            if parent.is_scalar and not options.is_elementwise():
                raise function_expr_invalid_operation_error(function, parent)
            kwargs = dict(input=input, function=function, options=options, **kwds)
            super().__init__(**kwargs)
    else:  # pragma: no cover
        ...

    def dispatch(
        self: Self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str
    ) -> R_co:
        return self.function.__expr_ir_dispatch__(self, ctx, frame, name)


class RollingExpr(FunctionExpr[RollingT_co]):
    def dispatch(
        self: Self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str
    ) -> R_co:
        return self.__expr_ir_dispatch__(self, ctx, frame, name)


class AnonymousExpr(
    FunctionExpr["MapBatches"], config=ExprIROptions.renamed("map_batches")
):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L158-L166."""

    def dispatch(
        self: Self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str
    ) -> R_co:
        return self.__expr_ir_dispatch__(self, ctx, frame, name)


class RangeExpr(FunctionExpr[RangeT_co]):
    """E.g. `int_range(...)`.

    Special-cased as it is only allowed scalar inputs, and is row_separable.
    """

    def __init__(
        self,
        *,
        input: Seq[ExprIR],  # noqa: A002
        function: RangeT_co,
        options: FunctionOptions,
        **kwds: t.Any,
    ) -> None:
        # NOTE: `IntRange` has 2x scalar inputs, so always triggered error in parent
        if len(input) < 2:
            msg = f"Expected at least 2 inputs for `{function!r}()`, but got `{len(input)}`.\n`{input}`"
            raise InvalidOperationError(msg)
        if not all(e.is_scalar for e in input):
            msg = f"All inputs for `{function!r}()` must be scalar or aggregations, but got \n`{input}`"
            raise InvalidOperationError(msg)
        super(ExprIR, self).__init__(
            **dict(input=input, function=function, options=options, **kwds)
        )

    def __repr__(self) -> str:
        return f"{self.function!r}({list(self.input)!r})"


class Filter(ExprIR, child=("expr", "by")):
    __slots__ = ("expr", "by")  # noqa: RUF023
    expr: ExprIR
    by: ExprIR

    @property
    def is_scalar(self) -> bool:
        return self.expr.is_scalar and self.by.is_scalar

    def __repr__(self) -> str:
        return f"{self.expr!r}.filter({self.by!r})"

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_output_name()


class WindowExpr(
    ExprIR, child=("expr", "partition_by"), config=ExprIROptions.renamed("over")
):
    """A fully specified `.over()`, that occurred after another expression.

    Related:
    - https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L129-L136
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L835-L838
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L840-L876
    """

    __slots__ = ("expr", "partition_by", "options")  # noqa: RUF023
    expr: ExprIR
    """For lazy backends, this should be the only place we allow `rolling_*`, `cum_*`."""
    partition_by: Seq[ExprIR]
    options: Window

    def __repr__(self) -> str:
        return f"{self.expr!r}.over({list(self.partition_by)!r})"

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.expr.iter_output_name()


class OrderedWindowExpr(
    WindowExpr,
    child=("expr", "partition_by", "order_by"),
    config=ExprIROptions.renamed("over_ordered"),
):
    __slots__ = ("order_by", "sort_options")
    expr: ExprIR
    partition_by: Seq[ExprIR]
    order_by: Seq[ExprIR]
    sort_options: SortOptions
    options: Window

    def __repr__(self) -> str:
        order = self.order_by
        if not self.partition_by:
            args = f"order_by={list(order)!r}"
        else:
            args = f"partition_by={list(self.partition_by)!r}, order_by={list(order)!r}"
        return f"{self.expr!r}.over({args})"

    def iter_root_names(self) -> t.Iterator[ExprIR]:
        # NOTE: `order_by` is never considered in `polars`
        # To match that behavior for `root_names` - but still expand in all other cases
        # - this little escape hatch exists
        # https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/plans/iterator.rs#L76-L86
        yield from self.expr.iter_left()
        for e in self.partition_by:
            yield from e.iter_left()
        yield self

    def order_by_names(self) -> Iterator[str]:
        """Yield the names resolved from expanding `order_by`.

        Raises:
            InvalidOperationError: If used *before* expansion, or
                `order_by` contains expressions that do more than select.
        """
        for by in self.order_by:
            if isinstance(by, Column):
                yield by.name
            else:
                raise over_order_by_names_error(self, by)


class Len(ExprIR, config=ExprIROptions.namespaced()):
    @property
    def is_scalar(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "len"

    def __repr__(self) -> str:
        return "len()"


class TernaryExpr(ExprIR, child=("truthy", "falsy", "predicate")):
    """When-Then-Otherwise."""

    __slots__ = ("truthy", "falsy", "predicate")  # noqa: RUF023
    predicate: ExprIR
    truthy: ExprIR
    falsy: ExprIR

    @property
    def is_scalar(self) -> bool:
        return self.predicate.is_scalar and self.truthy.is_scalar and self.falsy.is_scalar

    def __repr__(self) -> str:
        return (
            f".when({self.predicate!r}).then({self.truthy!r}).otherwise({self.falsy!r})"
        )

    def iter_output_name(self) -> t.Iterator[ExprIR]:
        yield from self.truthy.iter_output_name()


class RootSelector(SelectorIR):
    """A single selector expression."""

    __slots__ = ("selector",)
    selector: cs.Selector

    def __repr__(self) -> str:
        return f"{self.selector!r}"

    def iter_expand_names(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        yield from self.selector.iter_expand(schema, ignored_columns)

    def matches(self, dtype: IntoDType) -> bool:
        return self.selector.to_dtype_selector().matches(dtype)

    def to_dtype_selector(self) -> Self:
        return replace(self, selector=self.selector.to_dtype_selector())


class BinarySelector(
    _BinaryOp[LeftSelectorT, SelectorOperatorT, RightSelectorT],
    SelectorIR,
    t.Generic[LeftSelectorT, SelectorOperatorT, RightSelectorT],
):
    """Application of two selector exprs via a set operator."""

    def iter_expand_names(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        # by_name, by_index (upstream) lose their ability to reorder when used as a binary op
        # (As designed) https://github.com/pola-rs/polars/issues/19384
        names = schema.names
        left = frozenset(self.left.iter_expand_names(schema, ignored_columns))
        right = frozenset(self.right.iter_expand_names(schema, ignored_columns))
        remaining: frozenset[str] = self.op(left, right)
        target: Iterable[str]
        if remaining:
            target = (
                names
                if len(remaining) == len(names)
                else (nm for nm in names if nm in remaining)
            )
        else:
            target = ()
        yield from target

    def matches(self, dtype: IntoDType) -> bool:
        left = self.left.matches(dtype)
        right = self.right.matches(dtype)
        return bool(self.op(left, right))

    def to_dtype_selector(self) -> Self:
        return replace(
            self, left=self.left.to_dtype_selector(), right=self.right.to_dtype_selector()
        )


class InvertSelector(SelectorIR, t.Generic[SelectorT]):
    __slots__ = ("selector",)
    selector: SelectorT

    def __repr__(self) -> str:
        return f"~{self.selector!r}"

    def iter_expand_names(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        # by_name, by_index (upstream) lose their ability to reorder when used as a binary op
        # that includes invert, which is implemented as Difference(All, Selector)
        # (As designed) https://github.com/pola-rs/polars/issues/19384
        names = schema.names
        ignore = frozenset(self.selector.iter_expand_names(schema, ignored_columns))
        target: Iterable[str]
        if ignore:
            target = (
                ()
                if len(ignore) == len(names)
                else (nm for nm in names if nm not in ignore)
            )
        else:
            target = names
        yield from target

    def matches(self, dtype: IntoDType) -> bool:
        return not self.selector.to_dtype_selector().matches(dtype)

    def to_dtype_selector(self) -> Self:
        return replace(self, selector=self.selector.to_dtype_selector())
