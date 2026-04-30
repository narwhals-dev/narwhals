"""Home to most of the `ExprIR` subclasses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._expr_ir import ExprIR, SelectorIR
from narwhals._plan._nodes import node, nodes
from narwhals._plan.exceptions import over_order_by_names_error
from narwhals._plan.expressions.selectors import ByName
from narwhals._plan.typing import LeftT_co, OperatorT, RightT_co, Seq

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan._expansion import Expander
    from narwhals._plan.options import SortMultipleOptions, SortOptions
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType


__all__ = (
    "Alias",
    "BinaryExpr",
    "Cast",
    "Column",
    "Filter",
    "Len",
    "Over",
    "Sort",
    "SortBy",
    "TernaryExpr",
    "col",
    "ternary_expr",
)

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
get_dtype = ResolveDType.get_dtype
same_dtype = ResolveDType.expr_ir.same_dtype
namespaced = DispatcherOptions.namespaced


def col(name: str, /) -> Column:
    return Column(name=name)


class Len(ExprIR, dispatch=namespaced(), dtype=dtm.IDX_DTYPE):
    def is_scalar(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "len"

    def __repr__(self) -> str:
        return "len()"


class Alias(ExprIR, dispatch="no_dispatch"):
    """Rename an expression.

    Arguments:
        expr: An expression with a root of exactly one `Column`.
        name: The new name.

    Important:
        All expressions that change the output name are
        resolved and removed following expression expansion.
        This means that you can do arbitrarily complex renames,
        **at the narwhals-level** but there is intentionally no support
        for them at the compliant-level.
    """

    __slots__ = ("expr", "name")
    expr: ExprIR = node()
    name: str

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self

    def __repr__(self) -> str:
        return f"{self.expr!r}.alias({self.name!r})"

    def is_length_preserving(self) -> bool:
        return self.expr.is_length_preserving()


class Column(ExprIR, dispatch=namespaced("col")):
    """An expression that selects exactly one column.

    Arguments:
        name: A single column name.

    Examples:
        >>> import narwhals._plan as nw
        >>> expr = nw.col("one")
        >>> expr._ir
        col('one')
        >>> print(expr._ir)
        Column(name='one')

        A `Column` is not a selector, but can be converted into one:
        >>> expr.meta.as_selector()._ir
        ncs.by_name('one')
    """

    __slots__ = ("name",)
    name: str

    def __repr__(self) -> str:
        return f"col({self.name!r})"

    def to_selector_ir(self) -> SelectorIR:
        return ByName.from_name(self.name)

    def resolve_dtype(self, schema: FrozenSchema) -> DType:
        return schema[self.name]


class Cast(ExprIR, dtype=get_dtype()):
    __slots__ = ("expr", "dtype")  # noqa: RUF023
    expr: ExprIR = node()
    dtype: DType

    def __repr__(self) -> str:
        return f"{self.expr!r}.cast({self.dtype!r})"

    def is_length_preserving(self) -> bool:
        return self.expr.is_length_preserving()


class Sort(ExprIR, dtype=same_dtype()):
    __slots__ = ("expr", "options")
    expr: ExprIR = node()
    options: SortOptions

    def __repr__(self) -> str:
        direction = "desc" if self.options.descending else "asc"
        return f"{self.expr!r}.sort({direction})"

    def is_length_preserving(self) -> bool:
        return self.expr.is_length_preserving()


class SortBy(ExprIR, dtype=same_dtype()):
    __slots__ = ("expr", "by", "options")  # noqa: RUF023
    expr: ExprIR = node()
    by: Seq[ExprIR] = nodes()
    options: SortMultipleOptions

    def __repr__(self) -> str:
        opts = ""
        if any(self.descending):
            opts += f", descending={list(self.descending)}"
        if any(self.nulls_last):
            opts += f", nulls_last={list(self.nulls_last)}"
        return f"{self.expr!r}.sort_by({list(self.by)!r}{opts})"

    @property
    def descending(self) -> Seq[bool]:
        return self.options.descending

    @property
    def nulls_last(self) -> Seq[bool]:
        return self.options.nulls_last

    def is_length_preserving(self) -> bool:
        return self.expr.is_length_preserving()


class Filter(ExprIR, dtype=same_dtype()):
    __slots__ = ("expr", "by")  # noqa: RUF023
    expr: ExprIR = node(observe_scalar=False)
    by: ExprIR = node(observe_scalar=False)

    def __repr__(self) -> str:
        return f"{self.expr!r}.filter({self.by!r})"

    def is_length_preserving(self) -> bool:
        return False

    def changes_length(self) -> bool:
        return True


class Over(ExprIR, dtype=same_dtype()):
    """A fully specified `.over()`, that occurred after another expression.

    Related:
    - https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/expr.rs#L129-L136
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L835-L838
    - https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/mod.rs#L840-L876
    """

    __slots__ = ("expr", "partition_by")
    expr: ExprIR = node(observe_scalar=False)
    """For lazy backends, this should be the only place we allow `rolling_*`, `cum_*`."""
    partition_by: Seq[ExprIR] = nodes()

    def __repr__(self) -> str:
        return f"{self.expr!r}.over({list(self.partition_by)!r})"


class OverOrdered(Over):
    __slots__ = ("order_by", "sort_options")
    order_by: Seq[ExprIR] = nodes()
    sort_options: SortOptions

    def __repr__(self) -> str:
        order = self.order_by
        if not self.partition_by:
            args = f"order_by={list(order)!r}"
        else:
            args = f"partition_by={list(self.partition_by)!r}, order_by={list(order)!r}"
        return f"{self.expr!r}.over({args})"

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


class BinaryExpr(ExprIR, Generic[LeftT_co, OperatorT, RightT_co]):
    """A binary operation applied to two expressions."""

    __slots__ = ("left", "op", "right")
    left: LeftT_co = node()  # type: ignore[misc]
    op: OperatorT
    right: RightT_co = node()  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"[({self.left!r}) {self.op!r} ({self.right!r})]"

    def resolve_dtype(self, schema: FrozenSchema) -> DType:  # pragma: no cover
        """NOTE: Supported on `Logical` and `TrueDivide` operators only.

        Requires `get_supertype`:
        - `Add`
        - `Sub`
        - `Multiply`
        - `FloorDivide`
        - `Modulus`
        """
        return self.op.resolve_dtype(self, schema)

    def iter_expand(self, ctx: Expander, /) -> Iterator[ExprIR]:
        yield from self.__expr_ir_nodes__.iter_expand_by_combination(self, ctx)

    def is_length_preserving(self) -> bool:
        return self.left.is_length_preserving() or self.right.is_length_preserving()


# TODO @dangotbanned: `get_supertype`, `nw.Null`
class TernaryExpr(ExprIR):
    """When-Then-Otherwise."""

    __slots__ = ("truthy", "falsy", "predicate")  # noqa: RUF023
    # `truthy` is defined first because the root is from `when(...).then(<here>)`
    truthy: ExprIR = node()
    predicate: ExprIR = node()
    falsy: ExprIR = node()

    def __repr__(self) -> str:
        return (
            f".when({self.predicate!r}).then({self.truthy!r}).otherwise({self.falsy!r})"
        )

    def resolve_dtype(self, schema: FrozenSchema) -> DType:  # pragma: no cover
        msg = f"Unable to resolve dtype for {(type(self).__name__)!r}:\n{self!r}\n\n"
        "Requires `get_supertype` and `nw.Null`:\n"
        " - https://github.com/narwhals-dev/narwhals/issues/2835\n"
        " - https://github.com/narwhals-dev/narwhals/pull/3396\n\n"
        "See Also: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/schema.rs#L257-L273"
        raise NotImplementedError(msg)

    def iter_expand(self, ctx: Expander, /) -> Iterator[ExprIR]:
        yield from self.__expr_ir_nodes__.iter_expand_by_combination(self, ctx)

    def is_length_preserving(self) -> bool:
        return (
            self.predicate.is_length_preserving()
            or self.truthy.is_length_preserving()
            or self.falsy.is_length_preserving()
        )


def ternary_expr(predicate: ExprIR, truthy: ExprIR, falsy: ExprIR, /) -> TernaryExpr:
    return TernaryExpr(predicate=predicate, truthy=truthy, falsy=falsy)
