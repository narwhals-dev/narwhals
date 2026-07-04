from __future__ import annotations

import math
import operator
from collections import Counter
from itertools import accumulate, compress, count, pairwise, repeat
from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerSeries
from narwhals._typing_compat import assert_never
from narwhals._utils import Implementation, not_implemented
from narwhals.exceptions import ShapeError
from narwhals_dict.utils import binary_op, cast_values, infer_dtype, is_native_column

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from types import ModuleType

    from typing_extensions import Self, TypeIs

    from narwhals._utils import Version, _LimitedContext
    from narwhals.dtypes import DType
    from narwhals.typing import (
        ClosedInterval,
        FillNullStrategy,
        Into1DArray,
        IntoDType,
        PythonLiteral,
        SizedMultiIndexSelector,
        _1DArray,
        _SliceIndex,
    )
    from narwhals_dict.dataframe import DictDataFrame
    from narwhals_dict.namespace import DictNamespace
    from narwhals_dict.series_dt import DictSeriesDateTimeNamespace
    from narwhals_dict.series_str import DictSeriesStringNamespace
    from narwhals_dict.typing import NativeSeries


class DictSeries(EagerSeries["NativeSeries"]):  # type: ignore[type-var]
    _implementation = Implementation.UNKNOWN

    def __init__(
        self, native_series: NativeSeries, *, name: str, version: Version
    ) -> None:
        self._name = name
        self._native_series: list[Any] = (
            native_series if isinstance(native_series, list) else list(native_series)
        )
        self._version = version
        self._broadcast = False

    @property
    def native(self) -> list[Any]:
        return self._native_series

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> DType:
        return infer_dtype(self.native, self._version)

    def __native_namespace__(self) -> ModuleType:
        import builtins

        return builtins

    def __narwhals_namespace__(self) -> DictNamespace:
        from narwhals_dict.namespace import DictNamespace

        return DictNamespace(version=self._version)

    # Constructors and internal helpers
    @staticmethod
    def _is_native(obj: NativeSeries | Any) -> TypeIs[NativeSeries]:
        return is_native_column(obj)

    @classmethod
    def from_native(
        cls, data: NativeSeries, /, *, context: _LimitedContext, name: str = ""
    ) -> Self:
        return cls(data, name=name, version=context._version)

    @classmethod
    def from_iterable(
        cls,
        data: Iterable[Any],
        /,
        *,
        context: _LimitedContext,
        name: str = "",
        dtype: IntoDType | None = None,
    ) -> Self:
        result = cls(list(data), name=name, version=context._version)
        return result.cast(dtype) if dtype is not None else result

    @classmethod
    def from_numpy(cls, data: Into1DArray, /, *, context: _LimitedContext) -> Self:
        return cls.from_iterable(data.tolist(), context=context)

    def _with_version(self, version: Version) -> Self:
        return self.__class__(self.native, name=self.name, version=version)

    def _with_native(
        self, series: NativeSeries, *, preserve_broadcast: bool = False
    ) -> Self:
        result = self.__class__(series, name=self.name, version=self._version)
        if preserve_broadcast:
            result._broadcast = self._broadcast
        return result

    @classmethod
    def _align_full_broadcast(cls, *series: Self) -> Sequence[Self]:
        lengths = {len(s) for s in series if not s._broadcast}
        if not lengths:
            return series
        if len(lengths) > 1:
            msg = f"Expected series of equal lengths, got: {sorted(lengths)}."
            raise ShapeError(msg)
        target_length = lengths.pop()
        return [
            s._with_native([s.native[0]] * target_length)
            if s._broadcast and len(s) != target_length
            else s
            for s in series
        ]

    def alias(self, name: str) -> Self:
        result = self.__class__(self.native, name=name, version=self._version)
        result._broadcast = self._broadcast
        return result

    def _non_null(self) -> list[Any]:
        return [value for value in self.native if value is not None]

    def _extract_comparand(self, other: Any) -> tuple[Any, bool]:
        """Return `(values_or_scalar, is_scalar)` for use in elementwise operations."""
        if isinstance(other, DictSeries):
            if other._broadcast:
                return (other.native[0] if other.native else None), True
            return other.native, False
        return other, True

    def _with_binary(self, op: Callable[[Any, Any], Any], other: Any) -> Self:
        rhs, rhs_is_scalar = self._extract_comparand(other)
        preserve_broadcast = self._broadcast and getattr(other, "_broadcast", True)
        if self._broadcast and not rhs_is_scalar:
            # e.g. `nw.lit(1) + nw.col("a")`: broadcast the scalar left-hand side.
            lhs = self.native[0] if self.native else None
            result = [
                None if (lhs is None or value is None) else op(lhs, value)
                for value in rhs
            ]
        else:
            result = binary_op(op, self.native, rhs, is_scalar=rhs_is_scalar)
        return self._with_native(result, preserve_broadcast=preserve_broadcast)

    def _with_binary_right(self, op: Callable[[Any, Any], Any], other: Any) -> Self:
        return self._with_binary(lambda lhs, rhs: op(rhs, lhs), other)

    def _with_unary(
        self, fn: Callable[[Any], Any], *, preserve_broadcast: bool = True
    ) -> Self:
        return self._with_native(
            [None if value is None else fn(value) for value in self.native],
            preserve_broadcast=preserve_broadcast,
        )

    # Binary operations
    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        return self._with_binary(operator.eq, other)

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        return self._with_binary(operator.ne, other)

    def __ge__(self, other: Any) -> Self:
        return self._with_binary(operator.ge, other)

    def __gt__(self, other: Any) -> Self:
        return self._with_binary(operator.gt, other)

    def __le__(self, other: Any) -> Self:
        return self._with_binary(operator.le, other)

    def __lt__(self, other: Any) -> Self:
        return self._with_binary(operator.lt, other)

    def __and__(self, other: Any) -> Self:
        return self._with_binary(operator.and_, other)

    def __rand__(self, other: Any) -> Self:
        return self._with_binary_right(operator.and_, other)

    def __or__(self, other: Any) -> Self:
        return self._with_binary(operator.or_, other)

    def __ror__(self, other: Any) -> Self:
        return self._with_binary_right(operator.or_, other)

    def __add__(self, other: Any) -> Self:
        return self._with_binary(operator.add, other)

    def __radd__(self, other: Any) -> Self:
        return self._with_binary_right(operator.add, other)

    def __sub__(self, other: Any) -> Self:
        return self._with_binary(operator.sub, other)

    def __rsub__(self, other: Any) -> Self:
        return self._with_binary_right(operator.sub, other)

    def __mul__(self, other: Any) -> Self:
        return self._with_binary(operator.mul, other)

    def __rmul__(self, other: Any) -> Self:
        return self._with_binary_right(operator.mul, other)

    def __truediv__(self, other: Any) -> Self:
        return self._with_binary(operator.truediv, other)

    def __rtruediv__(self, other: Any) -> Self:
        return self._with_binary_right(operator.truediv, other)

    def __floordiv__(self, other: Any) -> Self:
        return self._with_binary(operator.floordiv, other)

    def __rfloordiv__(self, other: Any) -> Self:
        return self._with_binary_right(operator.floordiv, other)

    def __pow__(self, other: Any) -> Self:
        return self._with_binary(operator.pow, other)

    def __rpow__(self, other: Any) -> Self:
        return self._with_binary_right(operator.pow, other)

    def __mod__(self, other: Any) -> Self:
        return self._with_binary(operator.mod, other)

    def __rmod__(self, other: Any) -> Self:
        return self._with_binary_right(operator.mod, other)

    # Unary operations
    def __invert__(self) -> Self:
        return self._with_unary(operator.not_)

    def __neg__(self) -> Self:
        return self._with_unary(operator.neg)

    def abs(self) -> Self:
        return self._with_unary(operator.abs)

    def exp(self) -> Self:
        def _exp(value: float) -> float:
            try:
                return math.exp(value)
            except OverflowError:
                return float("inf")

        return self._with_unary(_exp)

    def sqrt(self) -> Self:
        def _sqrt(value: float) -> float:
            return math.sqrt(value) if value >= 0 else float("nan")

        return self._with_unary(_sqrt)

    def log(self, base: float) -> Self:
        def _log(value: float) -> float:
            if value > 0:
                return math.log(value, base)
            return float("-inf") if value == 0 else float("nan")

        return self._with_unary(_log)

    def sin(self) -> Self:
        return self._with_unary(math.sin)

    def cos(self) -> Self:
        return self._with_unary(math.cos)

    def floor(self) -> Self:
        return self._with_unary(math.floor)

    def ceil(self) -> Self:
        return self._with_unary(math.ceil)

    def round(self, decimals: int) -> Self:
        return self._with_unary(lambda value: round(value, decimals))

    def cast(self, dtype: IntoDType) -> Self:
        return self._with_native(
            cast_values(self.native, dtype, self._version), preserve_broadcast=True
        )

    def clip(self, lower_bound: Any, upper_bound: Any) -> Self:
        return self.clip_lower(lower_bound).clip_upper(upper_bound)

    def clip_lower(self, lower_bound: Any) -> Self:
        return self._with_binary(max, lower_bound)

    def clip_upper(self, upper_bound: Any) -> Self:
        return self._with_binary(min, upper_bound)

    # TODO(FBruzzesi): Do not fallback to super implementation
    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: ClosedInterval
    ) -> Self:
        lo, lo_is_scalar = self._extract_comparand(lower_bound)
        hi, hi_is_scalar = self._extract_comparand(upper_bound)
        if not (lo_is_scalar and hi_is_scalar) or lo is None or hi is None:
            # Series or null bounds: reuse the operator-based default,
            # which already handles alignment and null propagation.
            return super().is_between(lower_bound, upper_bound, closed)
        values = self.native
        if closed == "left":
            result = [None if v is None else lo <= v < hi for v in values]
        elif closed == "right":
            result = [None if v is None else lo < v <= hi for v in values]
        elif closed == "none":
            result = [None if v is None else lo < v < hi for v in values]
        elif closed == "both":
            result = [None if v is None else lo <= v <= hi for v in values]
        else:
            assert_never(closed)
        return self._with_native(result, preserve_broadcast=True)

    # Aggregations (return Python scalars, `None` for empty/all-null where applicable)
    def len(self) -> int:
        return len(self.native)

    def count(self) -> int:
        return len(self._non_null())

    def null_count(self) -> int:
        return len(self.native) - len(self._non_null())

    def n_unique(self) -> int:
        return len(set(self.native))

    def sum(self) -> float:
        return sum(self._non_null())

    def min(self) -> Any:
        values = self._non_null()
        return min(values) if values else None

    def max(self) -> Any:
        values = self._non_null()
        return max(values) if values else None

    def mean(self) -> float:
        values = self._non_null()
        return sum(values) / len(values) if values else None  # type: ignore[return-value]

    def median(self) -> float:
        import statistics

        values = self._non_null()
        return statistics.median(values) if values else None  # type: ignore[return-value]

    def std(self, *, ddof: int) -> float:
        variance = self.var(ddof=ddof)
        return math.sqrt(variance) if variance is not None else None  # pyright: ignore[reportReturnType]

    def var(self, *, ddof: int) -> float:
        values = self._non_null()
        if len(values) - ddof <= 0:
            return None  # type: ignore[return-value]
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / (len(values) - ddof)

    def any(self) -> bool:
        return any(self._non_null())

    def all(self) -> bool:
        return all(self._non_null())

    def arg_min(self) -> int:
        return self.native.index(min(self.native))

    def arg_max(self) -> int:
        return self.native.index(max(self.native))

    def first(self) -> PythonLiteral:
        return self.native[0] if self.native else None

    def last(self) -> PythonLiteral:
        return self.native[-1] if self.native else None

    def any_value(self, *, ignore_nulls: bool) -> PythonLiteral:
        if ignore_nulls:
            values = self._non_null()
            return values[0] if values else None
        return self.first()

    def item(self, index: int | None) -> Any:
        if index is None:
            if len(self.native) != 1:
                msg = f"can only call '.item()' if the Series is of length 1, got: {len(self.native)}"
                raise ValueError(msg)
            return self.native[0]
        return self.native[index]

    # Element-wise transformations
    def is_null(self) -> Self:
        return self._with_native(
            [value is None for value in self.native], preserve_broadcast=True
        )

    def is_nan(self) -> Self:
        return self._with_unary(
            lambda value: isinstance(value, float) and math.isnan(value)
        )

    def is_finite(self) -> Self:
        return self._with_unary(math.isfinite)

    def is_in(self, other: Any) -> Self:
        return self._with_unary(lambda value: value in other)

    def is_unique(self) -> Self:
        counts = Counter(self.native)
        return self._with_native([counts[value] == 1 for value in self.native])

    def is_first_distinct(self) -> Self:
        seen: set[Any] = set()
        result = []
        for value in self.native:
            result.append(value not in seen)
            seen.add(value)
        return self._with_native(result)

    def is_last_distinct(self) -> Self:
        seen: set[Any] = set()
        result = []
        for value in reversed(self.native):
            result.append(value not in seen)
            seen.add(value)
        return self._with_native(result[::-1])

    def fill_null(
        self, value: Any, strategy: FillNullStrategy | None, limit: int | None
    ) -> Self:
        if strategy is not None or limit is not None:
            msg = "`fill_null` with `strategy`/`limit` is not supported for the dict backend."
            raise NotImplementedError(msg)
        fill, is_scalar = self._extract_comparand(value)
        if is_scalar:
            result = [fill if current is None else current for current in self.native]
        else:
            result = [
                new if current is None else current
                for current, new in zip(self.native, fill, strict=True)
            ]
        return self._with_native(result, preserve_broadcast=True)

    def zip_with(self, mask: Self, other: Self) -> Self:
        return self._with_native(
            [
                current if flag else replacement
                for flag, current, replacement in zip(
                    mask.native, self.native, other.native, strict=True
                )
            ]
        )

    def shift(self, n: int) -> Self:
        if n == 0:
            return self._with_native(list(self.native))
        if n > 0:
            return self._with_native([*repeat(None, n), *self.native[:-n]])
        return self._with_native([*self.native[-n:], *repeat(None, -n)])

    def diff(self) -> Self:
        deltas = binary_op(
            operator.sub, self.native[1:], self.native[:-1], is_scalar=False
        )
        return self._with_native([None, *deltas])

    def _cum_agg(self, op: Callable[[Any, Any], Any], *, reverse: bool) -> Self:
        values = self.native[::-1] if reverse else self.native
        # Accumulate skipping nulls, then restore nulls at their original positions.
        accumulated: Iterator[Any] = accumulate(
            values,
            lambda acc, value: (
                acc if value is None else value if acc is None else op(acc, value)
            ),
        )
        result = [
            None if value is None else acc
            for value, acc in zip(values, accumulated, strict=True)
        ]
        return self._with_native(result[::-1] if reverse else result)

    def cum_sum(self, *, reverse: bool) -> Self:
        return self._cum_agg(operator.add, reverse=reverse)

    def cum_prod(self, *, reverse: bool) -> Self:
        return self._cum_agg(operator.mul, reverse=reverse)

    def cum_max(self, *, reverse: bool) -> Self:
        return self._cum_agg(max, reverse=reverse)

    def cum_min(self, *, reverse: bool) -> Self:
        return self._cum_agg(min, reverse=reverse)

    def cum_count(self, *, reverse: bool) -> Self:
        values = self.native[::-1] if reverse else self.native
        counts = accumulate(values, lambda n, value: n + (value is not None), initial=0)
        result = list(counts)[1:]
        return self._with_native(result[::-1] if reverse else result)

    # Filtering, gathering, sorting
    def filter(self, predicate: Any) -> Self:
        mask = predicate.native if isinstance(predicate, DictSeries) else list(predicate)
        if len(mask) != len(self.native):
            msg = f"Expected mask of length {len(self.native)}, got: {len(mask)}."
            raise ShapeError(msg)
        return self._with_native(list(compress(self.native, mask)))

    def drop_nulls(self) -> Self:
        return self._with_native(self._non_null())

    def head(self, n: int) -> Self:
        return self._with_native(self.native[:n])

    def tail(self, n: int) -> Self:
        if n >= 0:
            return self._with_native(self.native[max(len(self.native) - n, 0) :])
        return self._with_native(self.native[-n:])

    def gather_every(self, n: int, offset: int) -> Self:
        return self._with_native(self.native[offset::n])

    def _gather(self, rows: SizedMultiIndexSelector[NativeSeries]) -> Self:
        return self._with_native([self.native[i] for i in rows])

    def _gather_slice(self, rows: _SliceIndex | range) -> Self:
        return self._with_native(self.native[rows.start : rows.stop : rows.step])

    def scatter(self, indices: Self, values: Self) -> Self:
        result = list(self.native)
        for index, value in zip(indices, values, strict=True):
            result[index] = value
        return self._with_native(result)

    def arg_true(self) -> Self:
        return self._with_native(list(compress(count(), self.native)))

    def unique(self, *, maintain_order: bool = False) -> Self:
        return self._with_native(list(dict.fromkeys(self.native)))

    def sort(self, *, descending: bool, nulls_last: bool) -> Self:
        nulls = [value for value in self.native if value is None]
        rest = sorted(self._non_null(), reverse=descending)
        return self._with_native(rest + nulls if nulls_last else nulls + rest)

    def is_sorted(self, *, descending: bool) -> bool:
        op = operator.ge if descending else operator.le
        return all(op(lhs, rhs) for lhs, rhs in pairwise(self._non_null()))

    # Conversions
    def __contains__(self, other: Any) -> bool:
        return other in self.native

    def __iter__(self) -> Iterator[Any]:
        return iter(self.native)

    def to_list(self) -> list[Any]:
        return list(self.native)

    def to_frame(self) -> DictDataFrame:
        from narwhals_dict.dataframe import DictDataFrame

        return DictDataFrame({self.name: self.native}, version=self._version)

    def __array__(self, dtype: Any, *, copy: bool | None) -> _1DArray:
        import numpy as np

        return np.asarray(self.native, dtype=dtype)

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray:
        return self.__array__(dtype, copy=copy)

    # Not implemented (yet): fill in incrementally.
    ewm_mean = not_implemented()
    hist_from_bin_count = not_implemented()
    hist_from_bins = not_implemented()
    kurtosis = not_implemented()
    mode = not_implemented()
    quantile = not_implemented()
    rank = not_implemented()
    replace_strict = not_implemented()
    rolling_mean = not_implemented()
    rolling_std = not_implemented()
    rolling_sum = not_implemented()
    rolling_var = not_implemented()
    sample = not_implemented()
    skew = not_implemented()
    to_arrow = not_implemented()
    to_dummies = not_implemented()
    to_pandas = not_implemented()
    to_polars = not_implemented()
    value_counts = not_implemented()
    fill_nan = not_implemented()

    # Namespaces
    @property
    def str(self) -> DictSeriesStringNamespace:
        from narwhals_dict.series_str import DictSeriesStringNamespace

        return DictSeriesStringNamespace(self)

    @property
    def dt(self) -> DictSeriesDateTimeNamespace:
        from narwhals_dict.series_dt import DictSeriesDateTimeNamespace

        return DictSeriesDateTimeNamespace(self)

    # Namespaces: not implemented (yet).
    cat: Any = not_implemented()
    list: Any = not_implemented()
    struct: Any = not_implemented()
