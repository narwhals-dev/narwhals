"""Almost entirely complete, generic `selectors` implementation.

- Focusing on eager-only for now
"""

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING
from typing import Callable
from typing import Collection
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Protocol
from typing import Sequence
from typing import TypeVar
from typing import overload

from narwhals.typing import CompliantExpr
from narwhals.utils import _parse_time_unit_and_time_zone
from narwhals.utils import dtype_matches_time_unit_and_time_zone
from narwhals.utils import get_column_names
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from datetime import timezone

    from typing_extensions import Self
    from typing_extensions import TypeAlias
    from typing_extensions import TypeIs

    from narwhals.dtypes import DType
    from narwhals.typing import CompliantDataFrame
    from narwhals.typing import CompliantSeries
    from narwhals.typing import TimeUnit
    from narwhals.utils import Implementation
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

    # NOTE: Plugging the gap of this not being defined in `CompliantSeries`
    class CompliantSeriesWithDType(CompliantSeries, Protocol):
        @property
        def dtype(self) -> DType: ...


SeriesT = TypeVar("SeriesT", bound="CompliantSeriesWithDType")
DataFrameT = TypeVar("DataFrameT", bound="CompliantDataFrame")
SelectorOrExpr: TypeAlias = (
    "CompliantSelector[DataFrameT, SeriesT] | CompliantExpr[SeriesT]"
)
EvalSeries: TypeAlias = Callable[[DataFrameT], Sequence[SeriesT]]
EvalNames: TypeAlias = Callable[[DataFrameT], Sequence[str]]


# NOTE: Pretty much finished generic for eager backends
class CompliantSelectorNamespace(Generic[DataFrameT, SeriesT], Protocol):
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    # TODO @dangotbanned: push for adding to public API for `DataFrame`
    # Only need internally, but it plugs so many holes that it must be useful beyond that
    # https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.iter_columns.html
    def _iter_columns(self, df: DataFrameT, /) -> Iterator[SeriesT]: ...

    def _selector(
        self,
        context: _FullContext,
        call: EvalSeries[DataFrameT, SeriesT],
        evaluate_output_names: EvalNames[DataFrameT],
        /,
    ) -> CompliantSelector[DataFrameT, SeriesT]: ...

    def _is_dtype(
        self: CompliantSelectorNamespace[DataFrameT, SeriesT], dtype: type[DType], /
    ) -> CompliantSelector[DataFrameT, SeriesT]:
        def series(df: DataFrameT) -> Sequence[SeriesT]:
            return [ser for ser in self._iter_columns(df) if isinstance(ser.dtype, dtype)]

        def names(df: DataFrameT) -> Sequence[str]:
            return [
                ser.name for ser in self._iter_columns(df) if isinstance(ser.dtype, dtype)
            ]

        return self._selector(self, series, names)

    def by_dtype(
        self: Self, dtypes: Collection[DType | type[DType]]
    ) -> CompliantSelector[DataFrameT, SeriesT]:
        def series(df: DataFrameT) -> Sequence[SeriesT]:
            return [ser for ser in self._iter_columns(df) if ser.dtype in dtypes]

        def names(df: DataFrameT) -> Sequence[str]:
            return [ser.name for ser in self._iter_columns(df) if ser.dtype in dtypes]

        return self._selector(self, series, names)

    def matches(self: Self, pattern: str) -> CompliantSelector[DataFrameT, SeriesT]:
        p = re.compile(pattern)

        def series(df: DataFrameT) -> Sequence[SeriesT]:
            return [df.get_column(col) for col in df.columns if p.search(col)]

        def names(df: DataFrameT) -> Sequence[str]:
            return [col for col in df.columns if p.search(col)]

        return self._selector(self, series, names)

    def numeric(self: Self) -> CompliantSelector[DataFrameT, SeriesT]:
        def series(df: DataFrameT) -> Sequence[SeriesT]:
            return [ser for ser in self._iter_columns(df) if ser.dtype.is_numeric()]

        def names(df: DataFrameT) -> Sequence[str]:
            return [ser.name for ser in self._iter_columns(df) if ser.dtype.is_numeric()]

        return self._selector(self, series, names)

    def categorical(self: Self) -> CompliantSelector[DataFrameT, SeriesT]:
        return self._is_dtype(import_dtypes_module(self._version).Categorical)

    def string(self: Self) -> CompliantSelector[DataFrameT, SeriesT]:
        return self._is_dtype(import_dtypes_module(self._version).String)

    def boolean(self: Self) -> CompliantSelector[DataFrameT, SeriesT]:
        return self._is_dtype(import_dtypes_module(self._version).Boolean)

    def all(self: Self) -> CompliantSelector[DataFrameT, SeriesT]:
        def series(df: DataFrameT) -> Sequence[SeriesT]:
            return list(self._iter_columns(df))

        return self._selector(self, series, get_column_names)

    def datetime(
        self: Self,
        time_unit: TimeUnit | Iterable[TimeUnit] | None,
        time_zone: str | timezone | Iterable[str | timezone | None] | None,
    ) -> CompliantSelector[DataFrameT, SeriesT]:
        time_units, time_zones = _parse_time_unit_and_time_zone(time_unit, time_zone)
        matches = partial(
            dtype_matches_time_unit_and_time_zone,
            dtypes=import_dtypes_module(version=self._version),
            time_units=time_units,
            time_zones=time_zones,
        )

        def series(df: DataFrameT) -> Sequence[SeriesT]:
            return [ser for ser in self._iter_columns(df) if matches(ser.dtype)]

        def names(df: DataFrameT) -> Sequence[str]:
            return [ser.name for ser in self._iter_columns(df) if matches(ser.dtype)]

        return self._selector(self, series, names)

    def __init__(self: Self, context: _FullContext, /) -> None:
        self._implementation = context._implementation
        self._backend_version = context._backend_version
        self._version = context._version


# NOTE: CompliantExpr already provides `_implementation`, `_backend_version`
# https://github.com/narwhals-dev/narwhals/pull/2060
class CompliantSelector(CompliantExpr[SeriesT], Generic[DataFrameT, SeriesT], Protocol):
    _version: Version

    @property
    def selectors(self) -> CompliantSelectorNamespace[DataFrameT, SeriesT]: ...
    def __repr__(self: Self) -> str: ...
    def _to_expr(self: Self) -> CompliantExpr[SeriesT]: ...

    def _is_selector(
        self: Self,
        other: Self | CompliantExpr[SeriesT],
    ) -> TypeIs[CompliantSelector[DataFrameT, SeriesT]]:
        return isinstance(other, type(self))

    @overload
    def __sub__(self: Self, other: Self) -> Self: ...
    @overload
    def __sub__(self: Self, other: CompliantExpr[SeriesT]) -> CompliantExpr[SeriesT]: ...
    def __sub__(
        self: Self, other: SelectorOrExpr[DataFrameT, SeriesT]
    ) -> SelectorOrExpr[DataFrameT, SeriesT]:
        if self._is_selector(other):

            def series(df: DataFrameT) -> Sequence[SeriesT]:
                lhs_names, rhs_names = _eval_lhs_rhs(df, self, other)
                return [
                    x for x, name in zip(self(df), lhs_names) if name not in rhs_names
                ]

            def names(df: DataFrameT) -> Sequence[str]:
                lhs_names, rhs_names = _eval_lhs_rhs(df, self, other)
                return [x for x in lhs_names if x not in rhs_names]

            return self.selectors._selector(self, series, names)
        else:
            return self._to_expr() - other

    @overload
    def __or__(self: Self, other: Self) -> Self: ...
    @overload
    def __or__(self: Self, other: CompliantExpr[SeriesT]) -> CompliantExpr[SeriesT]: ...
    def __or__(
        self: Self, other: SelectorOrExpr[DataFrameT, SeriesT]
    ) -> SelectorOrExpr[DataFrameT, SeriesT]:
        if self._is_selector(other):

            def names(df: DataFrameT) -> Sequence[SeriesT]:
                lhs_names, rhs_names = _eval_lhs_rhs(df, self, other)
                return [
                    *(x for x, name in zip(self(df), lhs_names) if name not in rhs_names),
                    *other(df),
                ]

            def series(df: DataFrameT) -> Sequence[str]:
                lhs_names, rhs_names = _eval_lhs_rhs(df, self, other)
                return [*(x for x in lhs_names if x not in rhs_names), *rhs_names]

            return self.selectors._selector(self, names, series)
        else:
            return self._to_expr() | other

    @overload
    def __and__(self: Self, other: Self) -> Self: ...
    @overload
    def __and__(self: Self, other: CompliantExpr[SeriesT]) -> CompliantExpr[SeriesT]: ...
    def __and__(
        self: Self, other: SelectorOrExpr[DataFrameT, SeriesT]
    ) -> SelectorOrExpr[DataFrameT, SeriesT]:
        if self._is_selector(other):

            def series(df: DataFrameT) -> Sequence[SeriesT]:
                lhs_names, rhs_names = _eval_lhs_rhs(df, self, other)
                return [x for x, name in zip(self(df), lhs_names) if name in rhs_names]

            def names(df: DataFrameT) -> Sequence[str]:
                lhs_names, rhs_names = _eval_lhs_rhs(df, self, other)
                return [x for x in lhs_names if x in rhs_names]

            return self.selectors._selector(self, series, names)
        else:
            return self._to_expr() & other

    def __invert__(
        self: Self,
    ) -> CompliantSelector[DataFrameT, SeriesT]:
        return self.selectors.all() - self


# NOTE: Should probably be a `DataFrame` method
# Using `Expr` because this doesn't require `Selector` attrs/methods
def _eval_lhs_rhs(
    df: CompliantDataFrame, lhs: CompliantExpr, rhs: CompliantExpr
) -> tuple[Sequence[str], Sequence[str]]:
    return lhs._evaluate_output_names(df), rhs._evaluate_output_names(df)
