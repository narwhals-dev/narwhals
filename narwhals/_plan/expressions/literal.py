# mypy: disable-error-code="misc", warn-unused-configs=True
# see `LiteralExpr.value`
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, final

from narwhals._plan import common
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._expr_ir import ExprIR
from narwhals._plan.typing import (
    LiteralT_co,
    NativeSeriesT,
    NativeSeriesT_co,
    NonNestedLiteralT,
    NonNestedLiteralT_co,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan.series import Series
    from narwhals._utils import Version
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType

__all__ = ["Lit", "LitSeries", "lit", "lit_series"]

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
get_dtype = ResolveDType.get_dtype
namespaced = DispatcherOptions.namespaced


# TODO @dangotbanned: (low-prio) Define `__str__` to use (`value`, `dtype`)-order instead
# - Generated from alphabetical-ordered slots
# - Will break some doctests on a change
# TODO @dangotbanned: Maybe skip this and keep the classes separate?
# - Don't want dispatching for `LiteralExpr`
# - All base class checks have been replaced now
# - Only benefits will be:
#   - a few lines saved
#   - shared slots
#   - sharing attribute docstrings
class LiteralExpr(ExprIR, Generic[LiteralT_co], dtype=get_dtype()):
    __slots__ = ("dtype", "value")
    # NOTE: https://discuss.python.org/t/make-replace-stop-interfering-with-variance-inference/96092
    value: LiteralT_co
    """The literal value."""
    dtype: DType
    """The data type inferred or explicitly given at construction time."""

    @property
    def name(self) -> str:
        return "literal"

    def iter_output_name(self) -> Iterator[ExprIR]:
        yield self


@final
class Lit(LiteralExpr[NonNestedLiteralT_co], dispatch=namespaced()):
    """An expression representing a scalar literal value.

    >>> import narwhals._plan as nw
    >>> expr = nw.lit(1)
    >>> expr._ir
    lit(int: 1)
    >>> expr.meta.is_literal()
    True
    >>> expr._ir.is_scalar()
    True
    """

    def is_scalar(self) -> bool:
        return True

    def __repr__(self) -> str:
        v = self.value
        return f"lit({'null' if v is None else f'{type(v).__name__}: {v!s}'})"


@final
class LitSeries(LiteralExpr["Series[NativeSeriesT_co]"], dispatch=namespaced()):
    """An expression representing a series literal.

    >>> import narwhals._plan as nw
    >>> series = nw.Series.from_iterable([1], backend="pyarrow")
    >>> expr = nw.lit(series)
    >>> expr._ir
    lit(Series)
    >>> expr.meta.is_literal()
    True

    A length-1 series literal is not considered scalar
    >>> expr._ir.is_scalar()
    False
    """

    def is_scalar(self) -> bool:
        return False

    @staticmethod
    def from_series(series: Series[NativeSeriesT], /) -> LitSeries[NativeSeriesT]:
        return LitSeries(value=series, dtype=series.dtype)

    @property
    def name(self) -> str:
        return self.value.name

    @property
    def native(self) -> NativeSeriesT_co:
        return self.value.to_native()

    @property
    def version(self) -> Version:
        return self.value.version

    # TODO @dangotbanned: Maybe show more detail on origin (not values)?
    # - `lit(Series[polars])`
    # - `lit(Series[pl.Series])`
    # - `lit(Series<Implementation.POLARS: 'polars'>)`
    def __repr__(self) -> str:
        return "lit(Series)"

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        # NOTE: Adding `Series.__eq__` means this needed a manual override
        yield from (self.name, self.dtype, id(self.value))


def lit(
    value: NonNestedLiteralT, dtype: IntoDType | None = None
) -> Lit[NonNestedLiteralT]:
    if dtype is None:
        dtype = common.py_to_narwhals_dtype(value)
    else:
        dtype = common.into_dtype(dtype)
    return Lit(value=value, dtype=dtype)


lit_series = LitSeries.from_series
