from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import TypeVar

from narwhals._compliant.typing import CompliantDataFrameT_co
from narwhals._compliant.typing import CompliantExprT_contra
from narwhals._compliant.typing import CompliantFrameT_co
from narwhals._compliant.typing import CompliantLazyFrameT_co
from narwhals._compliant.typing import DepthTrackingExprAny
from narwhals._compliant.typing import DepthTrackingExprT_contra
from narwhals._compliant.typing import EagerExprT_contra
from narwhals._compliant.typing import LazyExprT_contra
from narwhals._compliant.typing import NativeExprT_co

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


if not TYPE_CHECKING:  # pragma: no cover
    if sys.version_info >= (3, 9):
        from typing import Protocol as Protocol38
    else:
        from typing import Generic as Protocol38
else:  # pragma: no cover
    # TODO @dangotbanned: Remove after dropping `3.8` (#2084)
    # - https://github.com/narwhals-dev/narwhals/pull/2064#discussion_r1965921386
    from typing import Protocol as Protocol38

__all__ = [
    "CompliantGroupBy",
    "DepthTrackingGroupBy",
    "EagerGroupBy",
    "LazyGroupBy",
    "NarwhalsAggregation",
]

NativeAggregationT_co = TypeVar(
    "NativeAggregationT_co", bound="str | Callable[..., Any]", covariant=True
)
NarwhalsAggregation: TypeAlias = Literal[
    "sum", "mean", "median", "max", "min", "std", "var", "len", "n_unique", "count"
]


_RE_LEAF_NAME: re.Pattern[str] = re.compile(r"(\w+->)")


class CompliantGroupBy(Protocol38[CompliantFrameT_co, CompliantExprT_contra]):
    _compliant_frame: Any
    _keys: Sequence[str]

    @property
    def compliant(self) -> CompliantFrameT_co:
        return self._compliant_frame  # type: ignore[no-any-return]

    def __init__(
        self,
        compliant_frame: CompliantFrameT_co,
        keys: Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None: ...

    def agg(self, *exprs: CompliantExprT_contra) -> CompliantFrameT_co: ...


class DataFrameGroupBy(
    CompliantGroupBy[CompliantDataFrameT_co, CompliantExprT_contra],
    Protocol38[CompliantDataFrameT_co, CompliantExprT_contra],
):
    def __iter__(self) -> Iterator[tuple[Any, CompliantDataFrameT_co]]: ...


class DepthTrackingGroupBy(
    CompliantGroupBy[CompliantFrameT_co, DepthTrackingExprT_contra],
    Protocol38[CompliantFrameT_co, DepthTrackingExprT_contra, NativeAggregationT_co],
):
    """`CompliantGroupBy` variant, deals with `Eager` and other backends that utilize `CompliantExpr._depth`."""

    _REMAP_AGGS: ClassVar[Mapping[NarwhalsAggregation, Any]]
    """Mapping from `narwhals` to native representation.

    Note:
    - `Dask` *may* return a `Callable` instead of a `str` referring to one.
    """

    def _ensure_all_simple(self, exprs: Sequence[DepthTrackingExprT_contra]) -> None:
        for expr in exprs:
            if not self._is_simple(expr):
                name = self.compliant._implementation.name.lower()
                msg = (
                    f"Non-trivial complex aggregation found.\n\n"
                    f"Hint: you were probably trying to apply a non-elementary aggregation with a"
                    f"{name!r} table.\n"
                    "Please rewrite your query such that group-by aggregations "
                    "are elementary. For example, instead of:\n\n"
                    "    df.group_by('a').agg(nw.col('b').round(2).mean())\n\n"
                    "use:\n\n"
                    "    df.with_columns(nw.col('b').round(2)).group_by('a').agg(nw.col('b').mean())\n\n"
                )
                raise ValueError(msg)

    @classmethod
    def _is_simple(cls, expr: DepthTrackingExprAny, /) -> bool:
        """Return `True` is we can efficiently use `expr` in a native `group_by` context."""
        return expr._is_elementary() and cls._leaf_name(expr) in cls._REMAP_AGGS

    @classmethod
    def _remap_expr_name(
        cls, name: NarwhalsAggregation | Any, /
    ) -> NativeAggregationT_co:
        """Replace `name`, with some native representation.

        Arguments:
            name: Name of a `nw.Expr` aggregation method.

        Returns:
            A native compatible representation.
        """
        return cls._REMAP_AGGS.get(name, name)

    @classmethod
    def _leaf_name(cls, expr: DepthTrackingExprAny, /) -> NarwhalsAggregation | Any:
        """Return the last function name in the chain defined by `expr`."""
        return _RE_LEAF_NAME.sub("", expr._function_name)


class EagerGroupBy(
    DepthTrackingGroupBy[CompliantDataFrameT_co, EagerExprT_contra, str],
    DataFrameGroupBy[CompliantDataFrameT_co, EagerExprT_contra],
    Protocol38[CompliantDataFrameT_co, EagerExprT_contra],
): ...


class LazyGroupBy(
    CompliantGroupBy[CompliantLazyFrameT_co, LazyExprT_contra],
    Protocol38[CompliantLazyFrameT_co, LazyExprT_contra, NativeExprT_co],
):
    def _evaluate_expr(self, expr: LazyExprT_contra, /) -> Iterator[NativeExprT_co]:
        output_names = expr._evaluate_output_names(self.compliant)
        aliases = (
            expr._alias_output_names(output_names)
            if expr._alias_output_names
            else output_names
        )
        native_exprs = expr(self.compliant)
        if expr._is_multi_output_unnamed():
            for native_expr, name, alias in zip(native_exprs, output_names, aliases):
                if name not in self._keys:
                    yield native_expr.alias(alias)
        else:
            for native_expr, alias in zip(native_exprs, aliases):
                yield native_expr.alias(alias)

    def _evaluate_exprs(
        self, exprs: Iterable[LazyExprT_contra], /
    ) -> Iterator[NativeExprT_co]:
        for expr in exprs:
            yield from self._evaluate_expr(expr)
