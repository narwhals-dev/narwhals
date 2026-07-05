from __future__ import annotations

import math
import operator
import random
import statistics
from collections import Counter
from itertools import accumulate, compress, count, pairwise, repeat
from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerSeries
from narwhals._typing_compat import assert_never
from narwhals._utils import NO_DEFAULT, Implementation, not_implemented
from narwhals.exceptions import InvalidOperationError, ShapeError
from narwhals_dict.utils import binary_op, cast_values, infer_dtype, is_native_column

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from types import ModuleType

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self, TypeIs

    from narwhals._typing import NoDefault
    from narwhals._utils import Version, _LimitedContext
    from narwhals.dtypes import DType
    from narwhals.typing import (
        ClosedInterval,
        FillNullStrategy,
        Into1DArray,
        IntoDType,
        ModeKeepStrategy,
        PythonLiteral,
        RankMethod,
        RollingInterpolationMethod,
        SizedMultiIndexSelector,
        _1DArray,
        _SliceIndex,
    )
    from narwhals_dict.dataframe import DictDataFrame
    from narwhals_dict.namespace import DictNamespace
    from narwhals_dict.series_dt import DictSeriesDateTimeNamespace
    from narwhals_dict.series_list import DictSeriesListNamespace
    from narwhals_dict.series_str import DictSeriesStringNamespace
    from narwhals_dict.typing import DictFrame, NativeSeries


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
        values = data.tolist()
        # `tolist` on a 0-d array or numpy scalar returns a plain scalar.
        return cls.from_iterable(
            values if isinstance(values, list) else [values], context=context
        )

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
            result = binary_op(
                lambda value, scalar: op(scalar, value), rhs, lhs, is_scalar=True
            )
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

    @staticmethod
    def _truediv(numerator: float, denominator: float) -> float:
        # Division by zero: match Polars/IEEE-754 (`±inf`, `0/0 -> NaN`)
        # instead of Python's `ZeroDivisionError`.
        try:
            return numerator / denominator
        except ZeroDivisionError:
            if numerator:
                return math.copysign(float("inf"), numerator)
            return float("nan")

    @staticmethod
    def _floordiv(numerator: float, denominator: float) -> float | None:
        # Floor division by zero: Polars returns null.
        try:
            return numerator // denominator
        except ZeroDivisionError:
            return None

    def __truediv__(self, other: Self) -> Self:
        return self._with_binary(self._truediv, other)

    def __rtruediv__(self, other: Self) -> Self:
        return self._with_binary_right(self._truediv, other)

    def __floordiv__(self, other: Self) -> Self:
        return self._with_binary(self._floordiv, other)

    def __rfloordiv__(self, other: Self) -> Self:
        return self._with_binary_right(self._floordiv, other)

    @staticmethod
    def _pow(base: float, exponent: float) -> float:
        result = base**exponent
        # Negative base with fractional exponent: match float semantics (Polars
        # returns NaN), not Python's complex result.
        return float("nan") if isinstance(result, complex) else result

    def __pow__(self, other: Self) -> Self:
        return self._with_binary(self._pow, other)

    def __rpow__(self, other: Self) -> Self:
        return self._with_binary_right(self._pow, other)

    def __mod__(self, other: Self) -> Self:
        return self._with_binary(operator.mod, other)

    def __rmod__(self, other: Self) -> Self:
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
        if not (values := self._non_null()):
            return None  # type: ignore[return-value]
        if not self.dtype.is_numeric():
            msg = "`median` operation not supported for non-numeric input type."
            raise InvalidOperationError(msg)
        return statistics.median(values)

    def std(self, *, ddof: int) -> float:
        variance = self.var(ddof=ddof)
        return math.sqrt(variance) if variance is not None else None  # pyright: ignore[reportReturnType]

    def var(self, *, ddof: int) -> float:
        values = self._non_null()
        if len(values) - ddof <= 0:
            return None  # type: ignore[return-value]
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / (len(values) - ddof)

    @staticmethod
    def _central_moments(values: Sequence[Any]) -> tuple[float, float, float]:
        """Biased central moments `(m2, m3, m4)`, in a single pass over `values`."""
        n = len(values)
        mean = sum(values) / n
        m2 = m3 = m4 = 0.0
        for value in values:
            delta = value - mean
            delta2 = delta * delta
            m2 += delta2
            m3 += delta2 * delta
            m4 += delta2 * delta2
        return m2 / n, m3 / n, m4 / n

    def skew(self) -> float | None:
        if not (values := self._non_null()):
            return None
        if len(values) == 1:
            return float("nan")
        m2, m3, _ = self._central_moments(values)
        return m3 / m2**1.5 if m2 else float("nan")

    def kurtosis(self) -> float | None:
        if not (values := self._non_null()):
            return None
        if len(values) == 1:
            return float("nan")
        m2, _, m4 = self._central_moments(values)
        return m4 / m2**2 - 3 if m2 else float("nan")

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod
    ) -> float:
        if not (values := sorted(self._non_null())):
            return None  # type: ignore[return-value]
        position = (len(values) - 1) * quantile
        lower = values[math.floor(position)]
        higher = values[math.ceil(position)]
        if interpolation == "lower":
            return lower
        if interpolation == "higher":
            return higher
        if interpolation == "midpoint":
            return (lower + higher) / 2
        if interpolation == "linear":
            return lower + (higher - lower) * (position - math.floor(position))
        if interpolation == "nearest":
            return values[round(position)]
        assert_never(interpolation)

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
        if not self.dtype.is_numeric():
            msg = f"`.is_nan` only supported for numeric dtype and not {self.dtype}, did you mean `.is_null`?"
            raise InvalidOperationError(msg)
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
        if strategy is not None:
            values = self.native if strategy == "forward" else self.native[::-1]
            result: list[Any] = []
            last: Any = None
            gap = 0
            for current in values:
                if current is not None:
                    last, gap = current, 0
                    result.append(current)
                else:
                    gap += 1
                    result.append(last if limit is None or gap <= limit else None)
            if strategy == "backward":
                result.reverse()
            return self._with_native(result, preserve_broadcast=True)
        fill, is_scalar = self._extract_comparand(value)
        if is_scalar:
            result = [fill if current is None else current for current in self.native]
        else:
            result = [
                new if current is None else current
                for current, new in zip(self.native, fill, strict=True)
            ]
        return self._with_native(result, preserve_broadcast=True)

    def fill_nan(self, value: float | None) -> Self:
        return self._with_native(
            [
                value if isinstance(current, float) and math.isnan(current) else current
                for current in self.native
            ],
            preserve_broadcast=True,
        )

    def replace_strict(
        self,
        default: Self | NoDefault,
        old: Sequence[Any],
        new: Sequence[Any],
        *,
        return_dtype: IntoDType | None,
    ) -> Self:
        # Casting the replacements (and default) up front keeps the pass over
        # the data itself to a single `dict.get` per value.
        if return_dtype is not None:
            new = cast_values(new, return_dtype, self._version)
        mapping = dict(zip(old, new, strict=True))
        if default is NO_DEFAULT:
            unset = object()
            unmatched: dict[Any, None] = {}
            result = []
            for value in self.native:
                replacement = mapping.get(value, unset)
                if replacement is unset:
                    if value is not None:
                        unmatched[value] = None
                    replacement = None
                result.append(replacement)
            if unmatched:
                msg = (
                    "replace_strict did not replace all non-null values.\n\n"
                    f"The following did not get replaced: {list(unmatched)}"
                )
                raise InvalidOperationError(msg)
        else:
            fill, fill_is_scalar = self._extract_comparand(default)
            if return_dtype is not None:
                fill = (
                    cast_values([fill], return_dtype, self._version)[0]
                    if fill_is_scalar
                    else cast_values(fill, return_dtype, self._version)
                )
            if fill_is_scalar:
                # Keys mapped to `None` are still hits, so `dict.get` alone
                # distinguishes matched from defaulted.
                result = [mapping.get(value, fill) for value in self.native]
            else:
                unset = object()
                result = [
                    fill_value
                    if (replacement := mapping.get(value, unset)) is unset
                    else replacement
                    for value, fill_value in zip(self.native, fill, strict=True)
                ]
        return self._with_native(result)

    def mode(self, *, keep: ModeKeepStrategy) -> Self:
        if keep == "all":
            return self._with_native(statistics.multimode(self.native))
        return self._with_native([statistics.mode(self.native)] if self.native else [])

    def rank(self, method: RankMethod, *, descending: bool) -> Self:
        ordered = sorted(
            (
                (value, index)
                for index, value in enumerate(self.native)
                if value is not None
            ),
            key=operator.itemgetter(0),
            reverse=descending,
        )
        result: list[Any] = [None] * len(self.native)
        start = 0
        dense_rank = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end][0] == ordered[start][0]:
                end += 1
            dense_rank += 1
            for offset in range(end - start):
                if method == "average":
                    rank_value: float = (start + 1 + end) / 2
                elif method == "min":
                    rank_value = start + 1
                elif method == "max":
                    rank_value = end
                elif method == "dense":
                    rank_value = dense_rank
                elif method == "ordinal":
                    rank_value = start + offset + 1
                else:
                    assert_never(method)
                result[ordered[start + offset][1]] = rank_value
            start = end
        return self._with_native(result)

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

    # Rolling and exponentially-weighted windows
    def _rolling_agg(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
        agg: Callable[[int, Any, Any], Any],
        needs_squares: bool = False,
    ) -> Self:
        """Apply `agg(count, total, total_of_squares)` over each rolling window.

        Windows are position-based: nulls occupy a slot but are excluded from
        `count`/`total`, mirroring `pandas`/Polars semantics. The prefix sum of
        squares is only accumulated when `needs_squares` is set.
        """
        values = self.native
        counts, totals, total_sqs = [0], [0], [0]
        for value in values:
            is_obs = value is not None
            counts.append(counts[-1] + is_obs)
            totals.append(totals[-1] + (value if is_obs else 0))
            if needs_squares:
                total_sqs.append(total_sqs[-1] + (value * value if is_obs else 0))
        if center:
            # Window [i - window_size // 2, i + (window_size - 1) // 2].
            offsets = (-(window_size // 2), (window_size - 1) // 2 + 1)
        else:
            offsets = (-window_size + 1, 1)
        result = []
        for index in range(len(values)):
            start = max(index + offsets[0], 0)
            end = min(index + offsets[1], len(values))
            count = counts[end] - counts[start]
            if count < min_samples:
                result.append(None)
            else:
                result.append(
                    agg(
                        count,
                        totals[end] - totals[start],
                        total_sqs[end] - total_sqs[start] if needs_squares else 0,
                    )
                )
        return self._with_native(result)

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._rolling_agg(
            window_size,
            min_samples=min_samples,
            center=center,
            agg=lambda _count, total, _total_sq: total,
        )

    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._rolling_agg(
            window_size,
            min_samples=min_samples,
            center=center,
            agg=lambda count, total, _total_sq: total / count if count else None,
        )

    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        def variance(count: int, total: Any, total_sq: Any) -> float | None:
            if count <= ddof:
                return None
            return max(total_sq - total * total / count, 0.0) / (count - ddof)

        return self._rolling_agg(
            window_size,
            min_samples=min_samples,
            center=center,
            agg=variance,
            needs_squares=True,
        )

    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self.rolling_var(
            window_size, min_samples=min_samples, center=center, ddof=ddof
        )._with_unary(math.sqrt)

    @staticmethod
    def _ewm_alpha(
        com: float | None,
        span: float | None,
        half_life: float | None,
        alpha: float | None,
    ) -> float:
        params = [p for p in (com, span, half_life, alpha) if p is not None]
        if len(params) > 1:
            msg = "`com`, `span`, `half_life`, and `alpha` are mutually exclusive."
            raise ValueError(msg)
        if com is not None:
            return 1.0 / (1.0 + com)
        if span is not None:
            return 2.0 / (span + 1.0)
        if half_life is not None:
            return 1.0 - math.exp(-math.log(2.0) / half_life)
        if alpha is not None:
            return float(alpha)
        msg = "One of `com`, `span`, `half_life`, or `alpha` must be provided."
        raise ValueError(msg)

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
    ) -> Self:
        alpha_ = self._ewm_alpha(com, span, half_life, alpha)
        # Weighted running mean, matching the `pandas` EWM algorithm: `old_wt`
        # decays by `1 - alpha` at every step (`ignore_nulls=False`) or only at
        # observations (`ignore_nulls=True`).
        old_wt_factor = 1.0 - alpha_
        new_wt = 1.0 if adjust else alpha_
        weighted: float | None = None
        old_wt = 1.0
        observations = 0
        result: list[float | None] = []
        for value in self.native:
            is_obs = value is not None
            observations += is_obs
            if weighted is None:
                if is_obs:
                    weighted = float(value)
            elif is_obs or not ignore_nulls:
                old_wt *= old_wt_factor
                if is_obs:
                    if weighted != value:
                        weighted = (old_wt * weighted + new_wt * value) / (
                            old_wt + new_wt
                        )
                    old_wt = old_wt + new_wt if adjust else 1.0
            result.append(weighted if is_obs and observations >= min_samples else None)
        return self._with_native(result)

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

    def sample(self, n: int, *, with_replacement: bool, seed: int | None) -> Self:
        rng = random.Random(seed)  # noqa: S311
        native = (
            rng.choices(self.native, k=n)
            if with_replacement
            else rng.sample(self.native, k=n)
        )
        return self._with_native(native)

    def unique(self, *, maintain_order: bool = False) -> Self:
        return self._with_native(list(dict.fromkeys(self.native)))

    def sort(self, *, descending: bool, nulls_last: bool) -> Self:
        nulls = [value for value in self.native if value is None]
        rest = sorted(self._non_null(), reverse=descending)
        return self._with_native(rest + nulls if nulls_last else nulls + rest)

    def is_sorted(self, *, descending: bool) -> bool:
        if not isinstance(descending, bool):
            msg = f"argument 'descending' should be a bool, got: {type(descending).__name__}"
            raise TypeError(msg)
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

    def value_counts(
        self, *, sort: bool, parallel: bool, name: str | None, normalize: bool
    ) -> DictDataFrame:
        """`parallel` is unused, exists for compatibility."""
        from narwhals_dict.dataframe import DictDataFrame

        value_name = name or ("proportion" if normalize else "count")
        counter = Counter(self.native)
        items = counter.most_common() if sort else list(counter.items())
        total = len(self.native)
        return DictDataFrame(
            {
                self.name: [value for value, _ in items],
                value_name: [count / total if normalize else count for _, count in items],
            },
            version=self._version,
        )

    def to_dummies(self, *, separator: str, drop_first: bool) -> DictDataFrame:
        from narwhals_dict.dataframe import DictDataFrame

        null_name = f"{self.name}{separator}null"
        length = len(self.native)
        columns: dict[Any, list[int]] = {}
        names: dict[Any, str] = {}
        for index, value in enumerate(self.native):
            column = columns.get(value)
            if column is None:
                column = columns[value] = [0] * length
                names[value] = (
                    null_name if value is None else f"{self.name}{separator}{value}"
                )
            column[index] = 1
        columns = {names[value]: column for value, column in columns.items()}
        non_null = sorted(name for name in columns if name != null_name)
        ordered = ([null_name] if null_name in columns else []) + non_null[
            int(drop_first) :
        ]
        return DictDataFrame(
            {name: columns[name] for name in ordered}, version=self._version
        )

    @staticmethod
    def _hist_data(
        values: Iterable[Any], bins: Sequence[float], *, include_breakpoint: bool
    ) -> DictFrame:
        """Count values per bin: intervals are `(lo, hi]`, the first also includes `lo`."""
        from bisect import bisect_left

        counts = [0] * (len(bins) - 1)
        lo, hi = bins[0], bins[-1]
        for value in values:
            if (
                value is None
                or (isinstance(value, float) and math.isnan(value))
                or not lo <= value <= hi
            ):
                continue
            counts[0 if value == lo else bisect_left(bins, value) - 1] += 1
        if include_breakpoint:
            return {"breakpoint": [float(edge) for edge in bins[1:]], "count": counts}
        return {"count": counts}

    def hist_from_bins(
        self, bins: list[float], *, include_breakpoint: bool
    ) -> DictDataFrame:
        from narwhals_dict.dataframe import DictDataFrame

        data: DictFrame
        if len(bins) <= 1:
            data = (
                {"breakpoint": [], "count": []} if include_breakpoint else {"count": []}
            )
        else:
            data = self._hist_data(
                self.native, bins, include_breakpoint=include_breakpoint
            )
        return DictDataFrame(data, version=self._version)

    def hist_from_bin_count(
        self, bin_count: int, *, include_breakpoint: bool
    ) -> DictDataFrame:
        from narwhals_dict.dataframe import DictDataFrame

        if bin_count == 0:
            data: DictFrame = (
                {"breakpoint": [], "count": []} if include_breakpoint else {"count": []}
            )
            return DictDataFrame(data, version=self._version)
        values = [
            value
            for value in self._non_null()
            if not (isinstance(value, float) and math.isnan(value))
        ]
        if not values:
            # Match Polars: an all-null series gets `bin_count` bins spanning [0, 1].
            lower, upper = 0.0, 1.0
        else:
            lower, upper = float(min(values)), float(max(values))
            if lower == upper:
                lower -= 0.5
                upper += 0.5
        width = (upper - lower) / bin_count
        bins = [lower + width * index for index in range(bin_count)] + [upper]
        return DictDataFrame(
            self._hist_data(values, bins, include_breakpoint=include_breakpoint),
            version=self._version,
        )

    def __array__(self, dtype: Any, *, copy: bool | None) -> _1DArray:
        import numpy as np

        values = self.native
        if dtype is None:
            nw_dtype = self.dtype
            if isinstance(nw_dtype, self._version.dtypes.Datetime):
                # NumPy has no time-zone support: convert to UTC, then drop it.
                from datetime import timezone

                values = [
                    value.astimezone(timezone.utc).replace(tzinfo=None)
                    if value is not None and value.tzinfo is not None
                    else value
                    for value in values
                ]
                return np.asarray(values, dtype="datetime64[us]")
            if None in values and nw_dtype.is_numeric():
                # Match other backends: numeric columns with nulls become
                # float64 with NaN, not an object array holding `None`.
                values = [float("nan") if value is None else value for value in values]
        return np.asarray(values, dtype=dtype)

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray:
        return self.__array__(dtype, copy=copy)

    def to_arrow(self) -> pa.Array[Any]:
        import pyarrow as pa

        return pa.array(self.native)

    def to_pandas(self) -> pd.Series[Any]:
        import pandas as pd

        return pd.Series(self.native, name=self.name)

    def to_polars(self) -> pl.Series:
        import polars as pl

        return pl.Series(self.name, self.native)

    # Namespaces
    @property
    def str(self) -> DictSeriesStringNamespace:
        from narwhals_dict.series_str import DictSeriesStringNamespace

        return DictSeriesStringNamespace(self)

    @property
    def dt(self) -> DictSeriesDateTimeNamespace:
        from narwhals_dict.series_dt import DictSeriesDateTimeNamespace

        return DictSeriesDateTimeNamespace(self)

    @property
    def list(self) -> DictSeriesListNamespace:
        from narwhals_dict.series_list import DictSeriesListNamespace

        return DictSeriesListNamespace(self)

    # Namespaces: not implemented (yet).
    cat: Any = not_implemented()
    struct: Any = not_implemented()
