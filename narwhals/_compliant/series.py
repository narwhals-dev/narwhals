from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Protocol
from typing import Sequence

from narwhals._compliant.any_namespace import CatNamespace
from narwhals._compliant.any_namespace import DateTimeNamespace
from narwhals._compliant.any_namespace import ListNamespace
from narwhals._compliant.any_namespace import StringNamespace
from narwhals._compliant.any_namespace import StructNamespace
from narwhals._compliant.typing import CompliantSeriesT_co
from narwhals._compliant.typing import EagerSeriesT_co
from narwhals._compliant.typing import NativeSeriesT
from narwhals._compliant.typing import NativeSeriesT_co
from narwhals._translate import FromIterable
from narwhals._translate import FromNative
from narwhals._translate import NumpyConvertible
from narwhals.utils import _StoresCompliant
from narwhals.utils import _StoresNative
from narwhals.utils import is_compliant_series
from narwhals.utils import is_sequence_like_ints
from narwhals.utils import is_slice_none
from narwhals.utils import unstable

if TYPE_CHECKING:
    from types import ModuleType

    import pandas as pd
    import polars as pl
    from typing_extensions import Self

    from narwhals._arrow.typing import ArrowArray
    from narwhals._compliant.dataframe import CompliantDataFrame
    from narwhals._compliant.expr import CompliantExpr
    from narwhals._compliant.expr import EagerExpr
    from narwhals._compliant.namespace import CompliantNamespace
    from narwhals._compliant.namespace import EagerNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import ClosedInterval
    from narwhals.typing import FillNullStrategy
    from narwhals.typing import Into1DArray
    from narwhals.typing import NonNestedLiteral
    from narwhals.typing import NumericLiteral
    from narwhals.typing import RankMethod
    from narwhals.typing import RollingInterpolationMethod
    from narwhals.typing import TemporalLiteral
    from narwhals.typing import _1DArray
    from narwhals.typing import _IntIndexer
    from narwhals.utils import Implementation
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

__all__ = ["CompliantSeries", "EagerSeries"]


class CompliantSeries(
    NumpyConvertible["_1DArray", "Into1DArray"],
    FromIterable,
    FromNative[NativeSeriesT],
    Protocol[NativeSeriesT],
):
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    @property
    def dtype(self) -> DType: ...
    @property
    def name(self) -> str: ...
    @property
    def native(self) -> NativeSeriesT: ...
    def __narwhals_series__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> CompliantNamespace[Any, Any]: ...
    def __native_namespace__(self) -> ModuleType: ...
    def __array__(self, dtype: Any, *, copy: bool | None) -> _1DArray: ...
    def __contains__(self, other: Any) -> bool: ...
    def __getitem__(self, item: Any) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int:
        return len(self.native)

    def _with_native(self, series: Any) -> Self: ...
    def _with_version(self, version: Version) -> Self: ...
    def _to_expr(self) -> CompliantExpr[Any, Self]: ...
    @classmethod
    def from_native(cls, data: NativeSeriesT, /, *, context: _FullContext) -> Self: ...
    @classmethod
    def from_numpy(cls, data: Into1DArray, /, *, context: _FullContext) -> Self: ...
    @classmethod
    def from_iterable(
        cls,
        data: Iterable[Any],
        /,
        *,
        context: _FullContext,
        name: str = "",
        dtype: DType | type[DType] | None = None,
    ) -> Self: ...

    # Operators
    def __add__(self, other: Any) -> Self: ...
    def __and__(self, other: Any) -> Self: ...
    def __eq__(self, other: object) -> Self: ...  # type: ignore[override]
    def __floordiv__(self, other: Any) -> Self: ...
    def __ge__(self, other: Any) -> Self: ...
    def __gt__(self, other: Any) -> Self: ...
    def __invert__(self) -> Self: ...
    def __le__(self, other: Any) -> Self: ...
    def __lt__(self, other: Any) -> Self: ...
    def __mod__(self, other: Any) -> Self: ...
    def __mul__(self, other: Any) -> Self: ...
    def __ne__(self, other: object) -> Self: ...  # type: ignore[override]
    def __or__(self, other: Any) -> Self: ...
    def __pow__(self, other: Any) -> Self: ...
    def __radd__(self, other: Any) -> Self: ...
    def __rand__(self, other: Any) -> Self: ...
    def __rfloordiv__(self, other: Any) -> Self: ...
    def __rmod__(self, other: Any) -> Self: ...
    def __rmul__(self, other: Any) -> Self: ...
    def __ror__(self, other: Any) -> Self: ...
    def __rpow__(self, other: Any) -> Self: ...
    def __rsub__(self, other: Any) -> Self: ...
    def __rtruediv__(self, other: Any) -> Self: ...
    def __sub__(self, other: Any) -> Self: ...
    def __truediv__(self, other: Any) -> Self: ...

    def abs(self) -> Self: ...
    def alias(self, name: str) -> Self: ...
    def all(self) -> bool: ...
    def any(self) -> bool: ...
    def arg_max(self) -> int: ...
    def arg_min(self) -> int: ...
    def arg_true(self) -> Self: ...
    def cast(self, dtype: DType | type[DType]) -> Self: ...
    def clip(
        self,
        lower_bound: Self | NumericLiteral | TemporalLiteral | None,
        upper_bound: Self | NumericLiteral | TemporalLiteral | None,
    ) -> Self: ...
    def count(self) -> int: ...
    def cum_count(self, *, reverse: bool) -> Self: ...
    def cum_max(self, *, reverse: bool) -> Self: ...
    def cum_min(self, *, reverse: bool) -> Self: ...
    def cum_prod(self, *, reverse: bool) -> Self: ...
    def cum_sum(self, *, reverse: bool) -> Self: ...
    def diff(self) -> Self: ...
    def drop_nulls(self) -> Self: ...
    def ewm_mean(
        self,
        *,
        com: float | None,
        span: float | None,
        half_life: float | None,
        alpha: float | None,
        adjust: bool,
        min_samples: int,
        ignore_nulls: bool,
    ) -> Self: ...
    def fill_null(
        self,
        value: Self | NonNestedLiteral,
        strategy: FillNullStrategy | None,
        limit: int | None,
    ) -> Self: ...
    def filter(self, predicate: Any) -> Self: ...
    def gather_every(self, n: int, offset: int) -> Self: ...
    @unstable
    def hist(
        self,
        bins: list[float | int] | None,
        *,
        bin_count: int | None,
        include_breakpoint: bool,
    ) -> CompliantDataFrame[Self, Any, Any]: ...
    def head(self, n: int) -> Self: ...
    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: ClosedInterval
    ) -> Self: ...
    def is_finite(self) -> Self: ...
    def is_first_distinct(self) -> Self: ...
    def is_in(self, other: Any) -> Self: ...
    def is_last_distinct(self) -> Self: ...
    def is_nan(self) -> Self: ...
    def is_null(self) -> Self: ...
    def is_sorted(self, *, descending: bool) -> bool: ...
    def is_unique(self) -> Self: ...
    def item(self, index: int | None) -> Any: ...
    def len(self) -> int: ...
    def max(self) -> Any: ...
    def mean(self) -> float: ...
    def median(self) -> float: ...
    def min(self) -> Any: ...
    def mode(self) -> Self: ...
    def n_unique(self) -> int: ...
    def null_count(self) -> int: ...
    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod
    ) -> float: ...
    def rank(self, method: RankMethod, *, descending: bool) -> Self: ...
    def replace_strict(
        self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any],
        *,
        return_dtype: DType | type[DType] | None,
    ) -> Self: ...
    def rolling_mean(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
    ) -> Self: ...
    def rolling_std(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
        ddof: int,
    ) -> Self: ...
    def rolling_sum(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
    ) -> Self: ...
    def rolling_var(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
        ddof: int,
    ) -> Self: ...
    def round(self, decimals: int) -> Self: ...
    def sample(
        self,
        n: int | None,
        *,
        fraction: float | None,
        with_replacement: bool,
        seed: int | None,
    ) -> Self: ...
    def scatter(self, indices: int | Sequence[int], values: Any) -> Self: ...
    def shift(self, n: int) -> Self: ...
    def skew(self) -> float | None: ...
    def sort(self, *, descending: bool, nulls_last: bool) -> Self: ...
    def std(self, *, ddof: int) -> float: ...
    def sum(self) -> float: ...
    def tail(self, n: int) -> Self: ...
    def to_arrow(self) -> ArrowArray: ...
    def to_dummies(
        self, *, separator: str, drop_first: bool
    ) -> CompliantDataFrame[Self, Any, Any]: ...
    def to_frame(self) -> CompliantDataFrame[Self, Any, Any]: ...
    def to_list(self) -> list[Any]: ...
    def to_pandas(self) -> pd.Series[Any]: ...
    def to_polars(self) -> pl.Series: ...
    def unique(self, *, maintain_order: bool) -> Self: ...
    def value_counts(
        self,
        *,
        sort: bool,
        parallel: bool,
        name: str | None,
        normalize: bool,
    ) -> CompliantDataFrame[Self, Any, Any]: ...
    def var(self, *, ddof: int) -> float: ...
    def zip_with(self, mask: Any, other: Any) -> Self: ...

    @property
    def str(self) -> Any: ...
    @property
    def dt(self) -> Any: ...
    @property
    def cat(self) -> Any: ...
    @property
    def list(self) -> Any: ...
    @property
    def struct(self) -> Any: ...


class EagerSeries(CompliantSeries[NativeSeriesT], Protocol[NativeSeriesT]):
    _native_series: Any
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version
    _broadcast: bool

    def _from_scalar(self, value: Any) -> Self:
        return self.from_iterable([value], name=self.name, context=self)

    def _with_native(
        self, series: NativeSeriesT, *, preserve_broadcast: bool = False
    ) -> Self:
        """Return a new `CompliantSeries`, wrapping the native `series`.

        In cases when operations are known to not affect whether a result should
        be broadcast, we can pass `preverse_broadcast=True`.
        Set this with care - it should only be set for unary expressions which don't
        change length or order, such as `.alias` or `.fill_null`. If in doubt, don't
        set it, you probably don't need it.
        """
        ...

    def __narwhals_namespace__(
        self,
    ) -> EagerNamespace[Any, Self, Any, Any, NativeSeriesT]: ...

    def _to_expr(self) -> EagerExpr[Any, Any]:
        return self.__narwhals_namespace__()._expr._from_series(self)  # type: ignore[no-any-return]

    def _gather(self, indices: _IntIndexer) -> Self: ...
    def _gather_slice(self, indices: slice | range) -> Self: ...

    def __getitem__(self, item: Any) -> Self:
        if is_slice_none(item):
            return self
        if isinstance(item, int):
            return self._gather([item])
        elif isinstance(item, (slice, range)):
            return self._gather_slice(item)
        elif is_compliant_series(item):
            return self._gather(item.native)
        elif is_sequence_like_ints(item):
            return self._gather(item)
        else:
            msg = "Unreachable code"
            raise AssertionError(msg)

    @property
    def str(self) -> EagerSeriesStringNamespace[Self, NativeSeriesT]: ...
    @property
    def dt(self) -> EagerSeriesDateTimeNamespace[Self, NativeSeriesT]: ...
    @property
    def cat(self) -> EagerSeriesCatNamespace[Self, NativeSeriesT]: ...
    @property
    def list(self) -> EagerSeriesListNamespace[Self, NativeSeriesT]: ...
    @property
    def struct(self) -> EagerSeriesStructNamespace[Self, NativeSeriesT]: ...


class _SeriesNamespace(  # type: ignore[misc]
    _StoresCompliant[CompliantSeriesT_co],
    _StoresNative[NativeSeriesT_co],
    Protocol[CompliantSeriesT_co, NativeSeriesT_co],
):
    _compliant_series: CompliantSeriesT_co

    @property
    def compliant(self) -> CompliantSeriesT_co:
        return self._compliant_series

    @property
    def native(self) -> NativeSeriesT_co:
        return self._compliant_series.native  # type: ignore[no-any-return]

    def with_native(self, series: Any, /) -> CompliantSeriesT_co:
        return self.compliant._with_native(series)


class EagerSeriesNamespace(
    _SeriesNamespace[EagerSeriesT_co, NativeSeriesT_co],
    Generic[EagerSeriesT_co, NativeSeriesT_co],
):
    _compliant_series: EagerSeriesT_co

    def __init__(self, series: EagerSeriesT_co, /) -> None:
        self._compliant_series = series


class EagerSeriesCatNamespace(  # type: ignore[misc]
    _SeriesNamespace[EagerSeriesT_co, NativeSeriesT_co],
    CatNamespace[EagerSeriesT_co],
    Protocol[EagerSeriesT_co, NativeSeriesT_co],
): ...


class EagerSeriesDateTimeNamespace(  # type: ignore[misc]
    _SeriesNamespace[EagerSeriesT_co, NativeSeriesT_co],
    DateTimeNamespace[EagerSeriesT_co],
    Protocol[EagerSeriesT_co, NativeSeriesT_co],
): ...


class EagerSeriesListNamespace(  # type: ignore[misc]
    _SeriesNamespace[EagerSeriesT_co, NativeSeriesT_co],
    ListNamespace[EagerSeriesT_co],
    Protocol[EagerSeriesT_co, NativeSeriesT_co],
): ...


class EagerSeriesStringNamespace(  # type: ignore[misc]
    _SeriesNamespace[EagerSeriesT_co, NativeSeriesT_co],
    StringNamespace[EagerSeriesT_co],
    Protocol[EagerSeriesT_co, NativeSeriesT_co],
): ...


class EagerSeriesStructNamespace(  # type: ignore[misc]
    _SeriesNamespace[EagerSeriesT_co, NativeSeriesT_co],
    StructNamespace[EagerSeriesT_co],
    Protocol[EagerSeriesT_co, NativeSeriesT_co],
): ...
