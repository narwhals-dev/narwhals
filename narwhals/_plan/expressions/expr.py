"""Top-level `Expr` nodes."""

from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Literal
import typing as t

from narwhals._plan.common import ExprIR, SelectorIR, collect
from narwhals._plan.exceptions import function_expr_invalid_operation_error
from narwhals._plan.expressions.aggregation import AggExpr, OrderableAggExpr
from narwhals._plan.expressions.name import KeepName, RenameAlias
from narwhals._plan.options import ExprIROptions
from narwhals._plan.typing import (
    FunctionT_co,
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
from narwhals._utils import flatten
from narwhals.exceptions import InvalidOperationError

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.functions import MapBatches  # noqa: F401
    from narwhals._plan.literal import LiteralValue
    from narwhals._plan.options import FunctionOptions, SortMultipleOptions, SortOptions
    from narwhals._plan.protocols import Ctx, FrameT_contra, R_co
    from narwhals._plan.selectors import Selector
    from narwhals._plan.window import Window
    from narwhals.dtypes import DType

__all__ = [
    "AggExpr",
    "Alias",
    "All",
    "AnonymousExpr",
    "BinaryExpr",
    "BinarySelector",
    "Cast",
    "Column",
    "Columns",
    "Exclude",
    "Filter",
    "FunctionExpr",
    "IndexColumns",
    "KeepName",
    "Len",
    "Literal",
    "Nth",
    "OrderableAggExpr",
    "RenameAlias",
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


def cols(*names: str) -> Columns:
    return Columns(names=names)


def nth(index: int, /) -> Nth:
    return Nth(index=index)


def index_columns(*indices: int) -> IndexColumns:
    return IndexColumns(indices=indices)


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


class _ColumnSelection(ExprIR, config=ExprIROptions.no_dispatch()):
    """Nodes which can resolve to `Column`(s) with a `Schema`."""


class Columns(_ColumnSelection):
    __slots__ = ("names",)
    names: Seq[str]

    def __repr__(self) -> str:
        return f"cols({list(self.names)!r})"


class Nth(_ColumnSelection):
    __slots__ = ("index",)
    index: int

    def __repr__(self) -> str:
        return f"nth({self.index})"


class IndexColumns(_ColumnSelection):
    __slots__ = ("indices",)
    indices: Seq[int]

    def __repr__(self) -> str:
        return f"index_columns({self.indices!r})"


class All(_ColumnSelection):
    def __repr__(self) -> str:
        return "all()"


class Exclude(_ColumnSelection, child=("expr",)):
    __slots__ = ("expr", "names")
    expr: ExprIR
    """Default is `all()`."""
    names: Seq[str]
    """Excluded names."""

    @staticmethod
    def from_names(expr: ExprIR, *names: str | t.Iterable[str]) -> Exclude:
        flat = flatten(names)
        return Exclude(expr=expr, names=collect(flat))

    def __repr__(self) -> str:
        return f"{self.expr!r}.exclude({list(self.names)!r})"


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
        super().__init__(**dict(input=input, function=function, options=options, **kwds))

    def dispatch(
        self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str
    ) -> R_co:
        return self.function.__expr_ir_dispatch__(ctx, t.cast("Self", self), frame, name)  # type: ignore[no-any-return]


class RollingExpr(FunctionExpr[RollingT_co]): ...


class AnonymousExpr(
    FunctionExpr["MapBatches"], config=ExprIROptions.renamed("map_batches")
):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/expr.rs#L158-L166."""

    def dispatch(
        self, ctx: Ctx[FrameT_contra, R_co], frame: FrameT_contra, name: str
    ) -> R_co:
        return self.__expr_ir_dispatch__(ctx, t.cast("Self", self), frame, name)  # type: ignore[no-any-return]


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
    __slots__ = ("expr", "partition_by", "order_by", "sort_options", "options")  # noqa: RUF023
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


class Len(ExprIR, config=ExprIROptions.namespaced()):
    @property
    def is_scalar(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "len"

    def __repr__(self) -> str:
        return "len()"


class RootSelector(SelectorIR):
    """A single selector expression."""

    __slots__ = ("selector",)
    selector: Selector

    def __repr__(self) -> str:
        return f"{self.selector!r}"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return self.selector.matches_column(name, dtype)


class BinarySelector(
    _BinaryOp[LeftSelectorT, SelectorOperatorT, RightSelectorT],
    SelectorIR,
    t.Generic[LeftSelectorT, SelectorOperatorT, RightSelectorT],
):
    """Application of two selector exprs via a set operator."""

    def matches_column(self, name: str, dtype: DType) -> bool:
        left = self.left.matches_column(name, dtype)
        right = self.right.matches_column(name, dtype)
        return bool(self.op(left, right))


class InvertSelector(SelectorIR, t.Generic[SelectorT]):
    __slots__ = ("selector",)
    selector: SelectorT

    def __repr__(self) -> str:
        return f"~{self.selector!r}"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return not self.selector.matches_column(name, dtype)


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
