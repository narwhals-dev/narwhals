# mypy: disable-error-code="misc", warn-unused-configs=True
# see `LiteralExpr.value`
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from functools import cache
from typing import TYPE_CHECKING, Any, Generic, final

from narwhals._plan import common
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._expr_ir import ExprIR
from narwhals._plan._guards import is_python_literal_type
from narwhals._plan.exceptions import literal_type_error
from narwhals._plan.typing import (
    LiteralT_co,
    NativeSeriesT,
    NativeSeriesT_co,
    PythonLiteralT,
    PythonLiteralT_co,
)
from narwhals._utils import Version
from narwhals.dtypes import Field, List, Struct, Unknown

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan.series import Series
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType, NonNestedDType, PythonLiteral

__all__ = ("Lit", "LitSeries", "lit", "lit_series")

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
get_dtype = ResolveDType.get_dtype


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

    def is_length_preserving(self) -> bool:
        return False

    def changes_length(self) -> bool:
        return False


@final
class Lit(
    LiteralExpr[PythonLiteralT_co], dispatch=DispatcherOptions.constructor("scalar")
):
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

    # TODO @dangotbanned: Use quotes for `lit("string")`
    # TODO @dangotbanned: (Noisy repr change) `lit(int: 1)` -> `lit[int](1)`
    # - Vaguely resembles C-style typing, but there's not an identifier (just a value)
    def __repr__(self) -> str:
        v: Any = self.value
        if v is None:
            return "lit(null)"
        name = type(v).__name__
        if self.dtype.is_nested():
            if isinstance(v, dict):
                name = f"struct[{len(v)}]"
            elif isinstance(v, (tuple)):
                name, v = "list", list(v)
        return f"lit({name}: {v})"

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        dtype = self.dtype
        value: Any = self.value
        yield from (id(value) if dtype.is_nested() else value, dtype)


@final
class LitSeries(
    LiteralExpr["Series[NativeSeriesT_co]"],
    dispatch=DispatcherOptions.constructor("expr"),
):
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
    def version(self) -> Version:  # pragma: no cover
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


lit_series = LitSeries.from_series


def lit(value: PythonLiteralT, dtype: IntoDType | None = None) -> Lit[PythonLiteralT]:
    dtype = (
        _py_value_to_dtype(value, Version.MAIN, allow_null=True)
        if dtype is None
        else common.into_dtype(dtype)
    )
    return Lit(value=value, dtype=dtype)


def _py_value_to_dtype(
    obj: PythonLiteral, version: Version = Version.MAIN, *, allow_null: bool
) -> DType:
    # NOTE: Surely mypy must have fixed `_lru_cache_wrapper` hashable in a new version?
    if dtype := _py_type_to_dtype(type(obj), version):  # type: ignore[arg-type]
        if allow_null or not isinstance(dtype, Unknown):
            return dtype
        msg = "Nested dtypes containing nulls are not yet supported"
        raise TypeError(msg)
    if not isinstance(obj, (list, dict, tuple)):
        # Just a type narrowing issue
        msg = f"Expected unreachable, got {obj!r}"
        raise NotImplementedError(msg)

    if not obj:
        msg = "Cannot infer dtype for empty nested structure. Please provide an explicit dtype parameter."
        raise TypeError(msg)
    if not isinstance(obj, dict):
        first_value = next((el for el in obj if el is not None), None)
        return List(_py_value_to_dtype(first_value, version, allow_null=False))
    return Struct(
        [
            Field(k, _py_value_to_dtype(v, version, allow_null=False))
            for k, v in obj.items()
        ]
    )


@cache
def _py_type_to_dtype(
    py_type: type[PythonLiteral], version: Version = Version.MAIN, /
) -> NonNestedDType | None:
    """SAFETY.

    Cache size is bound by these dimensions:

        n_valid_py_types = len(non_nested) + len([list, dict, tuple])
        maxsize = n_valid_py_types * len(Version)
    """
    dtypes = version.dtypes
    non_nested: dict[type[PythonLiteral], type[NonNestedDType]] = {
        int: dtypes.Int64,
        float: dtypes.Float64,
        str: dtypes.String,
        bool: dtypes.Boolean,
        dt.datetime: dtypes.Datetime,
        dt.date: dtypes.Date,
        dt.time: dtypes.Time,
        dt.timedelta: dtypes.Duration,
        bytes: dtypes.Binary,
        Decimal: dtypes.Decimal,
        type(None): dtypes.Unknown,
    }
    if dtype := non_nested.get(py_type):
        return dtype()
    if not is_python_literal_type(py_type):
        raise literal_type_error(py_type)
    return None
