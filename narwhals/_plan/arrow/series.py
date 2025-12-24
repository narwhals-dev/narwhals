from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc

from narwhals._arrow.utils import narwhals_to_native_dtype, native_to_narwhals_dtype
from narwhals._plan.arrow import compat, functions as fn, options
from narwhals._plan.arrow.common import ArrowFrameSeries as FrameSeries
from narwhals._plan.compliant.accessors import SeriesStructNamespace as StructNamespace
from narwhals._plan.compliant.series import CompliantSeries
from narwhals._plan.compliant.typing import namespace
from narwhals._plan.expressions import functions as F
from narwhals._utils import Version, generate_repr
from narwhals.dependencies import is_numpy_array_1d
from narwhals.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.arrow.dataframe import ArrowDataFrame as DataFrame
    from narwhals._plan.arrow.namespace import ArrowNamespace as Namespace
    from narwhals._plan.arrow.typing import ChunkedArrayAny
    from narwhals.dtypes import DType
    from narwhals.typing import (
        FillNullStrategy,
        Into1DArray,
        IntoDType,
        NonNestedLiteral,
        PythonLiteral,
        _1DArray,
    )

Incomplete: TypeAlias = Any


class ArrowSeries(FrameSeries["ChunkedArrayAny"], CompliantSeries["ChunkedArrayAny"]):
    _name: str

    def __repr__(self) -> str:
        return generate_repr(f"nw.{type(self).__name__}", self.native.__repr__())

    def _with_native(self, native: ChunkedArrayAny) -> Self:
        return self.from_native(native, self.name, version=self.version)

    def to_frame(self) -> DataFrame:
        return namespace(self)._dataframe.from_dict({self.name: self.native})

    def to_list(self) -> list[Any]:
        return self.native.to_pylist()

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray:
        return self.native.to_numpy()

    def to_polars(self) -> pl.Series:
        import polars as pl  # ignore-banned-import
        # NOTE: Recommended in https://github.com/pola-rs/polars/issues/22921#issuecomment-2908506022

        return pl.Series(self.name, self.native)

    def __len__(self) -> int:
        return self.native.length()

    @property
    def dtype(self) -> DType:
        return native_to_narwhals_dtype(self.native.type, self._version)

    @classmethod
    def from_numpy(
        cls, data: Into1DArray, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self:
        return cls.from_iterable(
            data if is_numpy_array_1d(data) else [data], name=name, version=version
        )

    @classmethod
    def from_iterable(
        cls,
        data: Iterable[Any],
        *,
        version: Version,
        name: str = "",
        dtype: IntoDType | None = None,
    ) -> Self:
        dtype_pa = fn.dtype_native(dtype, version)
        return cls.from_native(fn.chunked_array([data], dtype_pa), name, version=version)

    def cast(self, dtype: IntoDType) -> Self:
        dtype_pa = narwhals_to_native_dtype(dtype, self.version)
        return self._with_native(fn.cast(self.native, dtype_pa))

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        opts = options.array_sort(descending=descending, nulls_last=nulls_last)
        indices = pc.array_sort_indices(self.native, options=opts)
        return self._with_native(self._gather(indices))

    def scatter(self, indices: Self, values: Self) -> Self:
        mask = fn.is_in(fn.int_range(len(self), chunked=False), indices.native)
        replacements = values._gather(pc.sort_indices(indices.native))
        return self._with_native(fn.replace_with_mask(self.native, mask, replacements))

    def is_in(self, other: Self) -> Self:
        return self._with_native(fn.is_in(self.native, other.native))

    def is_nan(self) -> Self:
        return self._with_native(fn.is_nan(self.native))

    def is_null(self) -> Self:
        return self._with_native(fn.is_null(self.native))

    def is_not_nan(self) -> Self:
        return self._with_native(fn.is_not_nan(self.native))

    def is_not_null(self) -> Self:
        return self._with_native(fn.is_not_null(self.native))

    def has_nulls(self) -> bool:
        return bool(self.native.null_count)

    def null_count(self) -> int:
        return self.native.null_count

    __add__ = fn.bin_op(fn.add)
    __and__ = fn.bin_op(fn.and_)
    __eq__ = fn.bin_op(fn.eq)
    __floordiv__ = fn.bin_op(fn.floordiv)
    __ge__ = fn.bin_op(fn.gt_eq)
    __gt__ = fn.bin_op(fn.gt)
    __le__ = fn.bin_op(fn.lt_eq)
    __lt__ = fn.bin_op(fn.lt)
    __mod__ = fn.bin_op(fn.modulus)
    __mul__ = fn.bin_op(fn.multiply)
    __ne__ = fn.bin_op(fn.not_eq)
    __or__ = fn.bin_op(fn.or_)
    __pow__ = fn.bin_op(fn.power)
    __rfloordiv__ = fn.bin_op(fn.floordiv, reflect=True)
    __radd__ = fn.bin_op(fn.add, reflect=True)
    __rand__ = fn.bin_op(fn.and_, reflect=True)
    __rmod__ = fn.bin_op(fn.modulus, reflect=True)
    __rmul__ = fn.bin_op(fn.multiply, reflect=True)
    __ror__ = fn.bin_op(fn.or_, reflect=True)
    __rpow__ = fn.bin_op(fn.power, reflect=True)
    __rsub__ = fn.bin_op(fn.sub, reflect=True)
    __rtruediv__ = fn.bin_op(fn.truediv, reflect=True)
    __rxor__ = fn.bin_op(fn.xor, reflect=True)
    __sub__ = fn.bin_op(fn.sub)
    __truediv__ = fn.bin_op(fn.truediv)
    __xor__ = fn.bin_op(fn.xor)

    def __invert__(self) -> Self:
        return self._with_native(pc.invert(self.native))

    def cum_sum(self, *, reverse: bool = False) -> Self:
        if not reverse:
            return self._with_native(fn.cum_sum(self.native))
        return self._with_native(fn.cumulative(self.native, F.CumSum(reverse=reverse)))

    def cum_count(self, *, reverse: bool = False) -> Self:
        if not reverse:
            return self._with_native(fn.cum_count(self.native))
        return self._with_native(fn.cumulative(self.native, F.CumCount(reverse=reverse)))

    def cum_max(self, *, reverse: bool = False) -> Self:
        if not reverse:
            return self._with_native(fn.cum_max(self.native))
        return self._with_native(fn.cumulative(self.native, F.CumMax(reverse=reverse)))

    def cum_min(self, *, reverse: bool = False) -> Self:
        if not reverse:
            return self._with_native(fn.cum_min(self.native))
        return self._with_native(fn.cumulative(self.native, F.CumMin(reverse=reverse)))

    def cum_prod(self, *, reverse: bool = False) -> Self:
        if not reverse:
            return self._with_native(fn.cum_prod(self.native))
        return self._with_native(fn.cumulative(self.native, F.CumProd(reverse=reverse)))

    def fill_nan(self, value: float | Self | None) -> Self:
        fill_value = value.native if isinstance(value, ArrowSeries) else value
        return self._with_native(fn.fill_nan(self.native, fill_value))

    def fill_null(self, value: NonNestedLiteral | Self) -> Self:
        fill_value = value.native if isinstance(value, ArrowSeries) else value
        return self._with_native(fn.fill_null(self.native, fill_value))

    def fill_null_with_strategy(
        self, strategy: FillNullStrategy, limit: int | None = None
    ) -> Self:
        return self._with_native(fn.fill_null_with_strategy(self.native, strategy, limit))

    def diff(self, n: int = 1) -> Self:
        return self._with_native(fn.diff(self.native, n))

    def shift(self, n: int, *, fill_value: NonNestedLiteral = None) -> Self:
        return self._with_native(fn.shift(self.native, n, fill_value=fill_value))

    def _rolling_center(self, window_size: int) -> tuple[Self, int]:
        """Think this is similar to [`polars_core::chunked_array::ops::rolling_window::inner_mod::window_edges`].

        On `main`, this is `narwhals._arrow.utils.pad_series`.

        [`polars_core::chunked_array::ops::rolling_window::inner_mod::window_edges`]: https://github.com/pola-rs/polars/blob/e1d6f294218a36497255e2d872c223e19a47e2ec/crates/polars-core/src/chunked_array/ops/rolling_window.rs#L64-L77
        """
        offset_left = window_size // 2
        # subtract one if window_size is even
        offset_right = offset_left - (window_size % 2 == 0)
        native = self.native
        arrays = (
            fn.nulls_like(offset_left, native),
            *native.chunks,
            fn.nulls_like(offset_right, native),
        )
        offset = offset_left + offset_right
        return self._with_native(fn.concat_vertical(arrays)), offset

    def _rolling_sum(self, window_size: int, /) -> Self:
        cum_sum = self.cum_sum().fill_null_with_strategy("forward")
        return cum_sum.diff(window_size).fill_null(cum_sum)

    def _rolling_count(self, window_size: int, /) -> Self:
        cum_count = self.cum_count()
        return cum_count.diff(window_size).fill_null(cum_count)

    def rolling_sum(
        self, window_size: int, *, min_samples: int, center: bool = False
    ) -> Self:
        s, offset = self, 0
        if center:
            s, offset = self._rolling_center(window_size)
        rolling_count = s._rolling_count(window_size)
        keep = rolling_count >= min_samples
        result = s._rolling_sum(window_size).zip_with(keep, None)
        return result.slice(offset) if offset else result

    def rolling_mean(
        self, window_size: int, *, min_samples: int, center: bool = False
    ) -> Self:
        s, offset = self, 0
        if center:
            s, offset = self._rolling_center(window_size)
        rolling_count = s._rolling_count(window_size)
        keep = rolling_count >= min_samples
        result = (s._rolling_sum(window_size).zip_with(keep, None)) / rolling_count
        return result.slice(offset) if offset else result

    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool = False, ddof: int = 1
    ) -> Self:
        s, offset = self, 0
        if center:
            s, offset = self._rolling_center(window_size)
        rolling_count = s._rolling_count(window_size)
        keep = rolling_count >= min_samples

        # NOTE: Yes, these two are different
        sq_rolling_sum = s.pow(2)._rolling_sum(window_size)
        rolling_sum_sq = s._rolling_sum(window_size).pow(2)

        # NOTE: Please somebody rename these two to *something else*!
        rolling_something = sq_rolling_sum - (rolling_sum_sq / rolling_count)
        denominator = s._with_native(fn.max_horizontal((rolling_count - ddof).native, 0))
        result = rolling_something.zip_with(keep, None) / denominator
        return result.slice(offset) if offset else result

    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool = False, ddof: int = 1
    ) -> Self:
        return self.rolling_var(
            window_size, min_samples=min_samples, center=center, ddof=ddof
        ).pow(0.5)

    def zip_with(self, mask: Self, other: Self | None) -> Self:
        predicate = mask.native.combine_chunks()
        right = other.native if other is not None else other
        return self._with_native(fn.when_then(predicate, self.native, right))

    def all(self) -> bool:
        return fn.all_(self.native).as_py()

    def any(self) -> bool:
        return fn.any_(self.native).as_py()

    def sum(self) -> float:
        result: float = fn.sum_(self.native).as_py()
        return result

    def count(self) -> int:
        return fn.count(self.native).as_py()

    def unique(self, *, maintain_order: bool = False) -> Self:
        return self._with_native(self.native.unique())

    def drop_nulls(self) -> Self:
        return self._with_native(self.native.drop_null())

    def drop_nans(self) -> Self:
        predicate: Incomplete = fn.is_not_nan(self.native)
        return self._with_native(
            self.native.filter(predicate, null_selection_behavior="emit_null")
        )

    def explode(self, *, empty_as_null: bool = True, keep_nulls: bool = True) -> Self:
        exploder = fn.ExplodeBuilder(empty_as_null=empty_as_null, keep_nulls=keep_nulls)
        return self._with_native(exploder.explode(self.native))

    def first(self) -> PythonLiteral:
        return self.native[0].as_py() if len(self) else None

    def last(self) -> PythonLiteral:
        ca = self.native
        return ca[height - 1].as_py() if (height := len(ca)) else None

    @property
    def struct(self) -> SeriesStructNamespace:
        return SeriesStructNamespace(self)


class SeriesStructNamespace(StructNamespace["DataFrame", ArrowSeries]):
    def __init__(self, compliant: ArrowSeries, /) -> None:
        self._compliant: ArrowSeries = compliant

    @property
    def compliant(self) -> ArrowSeries:
        return self._compliant

    @property
    def native(self) -> ChunkedArrayAny:
        return self.compliant.native

    def __narwhals_namespace__(self) -> Namespace:
        return namespace(self.compliant)

    @property
    def version(self) -> Version:
        return self.compliant.version

    def with_native(self, native: ChunkedArrayAny, name: str, /) -> ArrowSeries:
        return self.compliant.from_native(native, name, version=self.version)

    def unnest(self) -> DataFrame:
        native = cast("pa.ChunkedArray[pa.StructScalar]", self.native)
        if compat.HAS_FROM_TO_STRUCT_ARRAY:
            if len(native):
                table = pa.Table.from_struct_array(native)
            else:
                table = fn.struct_schema(native).empty_table()
        else:  # pragma: no cover
            # NOTE: Too strict, doesn't allow `Array[StructScalar]`
            rec_batch: Incomplete = pa.RecordBatch.from_struct_array
            batches = (rec_batch(chunk) for chunk in native.chunks)
            table = pa.Table.from_batches(batches, fn.struct_schema(native))
        return namespace(self)._dataframe.from_native(table, self.version)

    # name overriding *may* be wrong
    def field(self, name: str) -> ArrowSeries:
        return self.with_native(fn.struct_field(self.native, name), name)

    @property
    def schema(self) -> Schema:
        return Schema.from_arrow(fn.struct_schema(self.native))
