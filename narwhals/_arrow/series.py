from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Sequence
from typing import overload

from narwhals._arrow.utils import cast_for_truediv
from narwhals._arrow.utils import floordiv_compat
from narwhals._arrow.utils import narwhals_to_native_dtype
from narwhals._arrow.utils import translate_dtype
from narwhals._arrow.utils import validate_column_comparand
from narwhals.utils import Implementation
from narwhals.utils import generate_unique_token

if TYPE_CHECKING:
    from types import ModuleType

    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals.dtypes import DType


class ArrowSeries:
    def __init__(
        self,
        native_series: pa.ChunkedArray,
        *,
        name: str,
        backend_version: tuple[int, ...],
    ) -> None:
        self._name = name
        self._native_series = native_series
        self._implementation = Implementation.PYARROW
        self._backend_version = backend_version

    def _from_native_series(self, series: Any) -> Self:
        import pyarrow as pa  # ignore-banned-import()

        if isinstance(series, pa.Array):
            series = pa.chunked_array([series])
        return self.__class__(
            series,
            name=self._name,
            backend_version=self._backend_version,
        )

    @classmethod
    def _from_iterable(
        cls: type[Self],
        data: Iterable[Any],
        name: str,
        *,
        backend_version: tuple[int, ...],
    ) -> Self:
        import pyarrow as pa  # ignore-banned-import()

        return cls(
            pa.chunked_array([data]),
            name=name,
            backend_version=backend_version,
        )

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._arrow.namespace import ArrowNamespace

        return ArrowNamespace(backend_version=self._backend_version)

    def __len__(self) -> int:
        return len(self._native_series)

    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        import pyarrow.compute as pc

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.equal(ser, other))

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.not_equal(ser, other))

    def __ge__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.greater_equal(ser, other))

    def __gt__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.greater(ser, other))

    def __le__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.less_equal(ser, other))

    def __lt__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.less(ser, other))

    def __and__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.and_kleene(ser, other))

    def __rand__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.and_kleene(other, ser))

    def __or__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.or_kleene(ser, other))

    def __ror__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.or_kleene(other, ser))

    def __add__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        other = validate_column_comparand(other)
        return self._from_native_series(pc.add(self._native_series, other))

    def __radd__(self, other: Any) -> Self:
        return self + other  # type: ignore[no-any-return]

    def __sub__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        other = validate_column_comparand(other)
        return self._from_native_series(pc.subtract(self._native_series, other))

    def __rsub__(self, other: Any) -> Self:
        return (self - other) * (-1)  # type: ignore[no-any-return]

    def __mul__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        other = validate_column_comparand(other)
        return self._from_native_series(pc.multiply(self._native_series, other))

    def __rmul__(self, other: Any) -> Self:
        return self * other  # type: ignore[no-any-return]

    def __pow__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.power(ser, other))

    def __rpow__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.power(other, ser))

    def __floordiv__(self, other: Any) -> Self:
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(floordiv_compat(ser, other))

    def __rfloordiv__(self, other: Any) -> Self:
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(floordiv_compat(other, ser))

    def __truediv__(self, other: Any) -> Self:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        if not isinstance(other, (pa.Array, pa.ChunkedArray)):
            # scalar
            other = pa.scalar(other)
        return self._from_native_series(pc.divide(*cast_for_truediv(ser, other)))

    def __rtruediv__(self, other: Any) -> Self:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        if not isinstance(other, (pa.Array, pa.ChunkedArray)):
            # scalar
            other = pa.scalar(other)
        return self._from_native_series(pc.divide(*cast_for_truediv(other, ser)))

    def __mod__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        floor_div = (self // other)._native_series
        res = pc.subtract(ser, pc.multiply(floor_div, other))
        return self._from_native_series(res)

    def __rmod__(self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        other = validate_column_comparand(other)
        floor_div = (other // self)._native_series
        res = pc.subtract(other, pc.multiply(floor_div, ser))
        return self._from_native_series(res)

    def __invert__(self) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(pc.invert(self._native_series))

    def len(self) -> int:
        return len(self._native_series)

    def filter(self, other: Any) -> Self:
        if not (isinstance(other, list) and all(isinstance(x, bool) for x in other)):
            other = validate_column_comparand(other)
        return self._from_native_series(self._native_series.filter(other))

    def mean(self) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return pc.mean(self._native_series)  # type: ignore[no-any-return]

    def min(self) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return pc.min(self._native_series)  # type: ignore[no-any-return]

    def max(self) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return pc.max(self._native_series)  # type: ignore[no-any-return]

    def sum(self) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return pc.sum(self._native_series)  # type: ignore[no-any-return]

    def drop_nulls(self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(pc.drop_null(self._native_series))

    def shift(self, n: int) -> Self:
        import pyarrow as pa  # ignore-banned-import()

        ca = self._native_series

        if n > 0:
            result = pa.concat_arrays([pa.nulls(n, ca.type), *ca[:-n].chunks])
        elif n < 0:
            result = pa.concat_arrays([*ca[-n:].chunks, pa.nulls(-n, ca.type)])
        else:
            result = ca
        return self._from_native_series(result)

    def std(self, ddof: int = 1) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return pc.stddev(self._native_series, ddof=ddof)  # type: ignore[no-any-return]

    def count(self) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return pc.count(self._native_series)  # type: ignore[no-any-return]

    def n_unique(self) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        unique_values = pc.unique(self._native_series)
        return pc.count(unique_values, mode="all")  # type: ignore[no-any-return]

    def __native_namespace__(self: Self) -> ModuleType:
        if self._implementation is Implementation.PYARROW:
            return self._implementation.to_native_namespace()

        msg = f"Expected pyarrow, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    @property
    def name(self) -> str:
        return self._name

    def __narwhals_series__(self) -> Self:
        return self

    @overload
    def __getitem__(self, idx: int) -> Any: ...

    @overload
    def __getitem__(self, idx: slice | Sequence[int]) -> Self: ...

    def __getitem__(self, idx: int | slice | Sequence[int]) -> Any | Self:
        if isinstance(idx, int):
            return self._native_series[idx]
        if isinstance(idx, Sequence):
            return self._from_native_series(self._native_series.take(idx))
        return self._from_native_series(self._native_series[idx])

    def scatter(self, indices: int | Sequence[int], values: Any) -> Self:
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import
        import pyarrow.compute as pc  # ignore-banned-import

        ca = self._native_series
        mask = np.zeros(len(ca), dtype=bool)
        mask[indices] = True
        if isinstance(values, self.__class__):
            values = validate_column_comparand(values)
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        if not isinstance(values, pa.Array):
            values = pa.array(values)
        result = pc.replace_with_mask(ca, mask, values.take(indices))
        return self._from_native_series(result)

    def to_list(self) -> Any:
        return self._native_series.to_pylist()

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        return self._native_series.__array__(dtype=dtype, copy=copy)

    def to_numpy(self) -> Any:
        return self._native_series.to_numpy()

    def alias(self, name: str) -> Self:
        return self.__class__(
            self._native_series,
            name=name,
            backend_version=self._backend_version,
        )

    @property
    def dtype(self: Self) -> DType:
        return translate_dtype(self._native_series.type)

    def abs(self) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(pc.abs(self._native_series))

    def cum_sum(self) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(pc.cumulative_sum(self._native_series))

    def round(self, decimals: int) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(
            pc.round(self._native_series, decimals, round_mode="half_towards_infinity")
        )

    def diff(self) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(
            pc.pairwise_diff(self._native_series.combine_chunks())
        )

    def any(self) -> bool:
        import pyarrow.compute as pc  # ignore-banned-import()

        return pc.any(self._native_series)  # type: ignore[no-any-return]

    def all(self) -> bool:
        import pyarrow.compute as pc  # ignore-banned-import()

        return pc.all(self._native_series)  # type: ignore[no-any-return]

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        if closed == "left":
            ge = pc.greater_equal(ser, lower_bound)
            lt = pc.less(ser, upper_bound)
            res = pc.and_kleene(ge, lt)
        elif closed == "right":
            gt = pc.greater(ser, lower_bound)
            le = pc.less_equal(ser, upper_bound)
            res = pc.and_kleene(gt, le)
        elif closed == "none":
            gt = pc.greater(ser, lower_bound)
            lt = pc.less(ser, upper_bound)
            res = pc.and_kleene(gt, lt)
        elif closed == "both":
            ge = pc.greater_equal(ser, lower_bound)
            le = pc.less_equal(ser, upper_bound)
            res = pc.and_kleene(ge, le)
        else:  # pragma: no cover
            raise AssertionError
        return self._from_native_series(res)

    def is_empty(self) -> bool:
        return len(self) == 0

    def is_null(self) -> Self:
        ser = self._native_series
        return self._from_native_series(ser.is_null())

    def cast(self, dtype: DType) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        dtype = narwhals_to_native_dtype(dtype)
        return self._from_native_series(pc.cast(ser, dtype))

    def null_count(self: Self) -> int:
        return self._native_series.null_count  # type: ignore[no-any-return]

    def head(self, n: int) -> Self:
        ser = self._native_series
        if n >= 0:
            return self._from_native_series(ser.slice(0, n))
        else:
            num_rows = len(ser)
            return self._from_native_series(ser.slice(0, max(0, num_rows + n)))

    def tail(self, n: int) -> Self:
        ser = self._native_series
        if n >= 0:
            num_rows = len(ser)
            return self._from_native_series(ser.slice(max(0, num_rows - n)))
        else:
            return self._from_native_series(ser.slice(abs(n)))

    def is_in(self, other: Any) -> Self:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        value_set = pa.array(other)
        ser = self._native_series
        return self._from_native_series(pc.is_in(ser, value_set=value_set))

    def arg_true(self) -> Self:
        import numpy as np  # ignore-banned-import

        ser = self._native_series
        res = np.flatnonzero(ser)
        return self._from_iterable(
            res, name=self.name, backend_version=self._backend_version
        )

    def item(self: Self, index: int | None = None) -> Any:
        if index is None:
            if len(self) != 1:
                msg = (
                    "can only call '.item()' if the Series is of length 1,"
                    f" or an explicit index is provided (Series is of length {len(self)})"
                )
                raise ValueError(msg)
            return self._native_series[0]
        return self._native_series[index]

    def value_counts(
        self: Self,
        *,
        sort: bool = False,
        parallel: bool = False,
        name: str | None = None,
        normalize: bool = False,
    ) -> ArrowDataFrame:
        """Parallel is unused, exists for compatibility"""
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        from narwhals._arrow.dataframe import ArrowDataFrame

        index_name_ = "index" if self._name is None else self._name
        value_name_ = name or ("proportion" if normalize else "count")

        val_count = pc.value_counts(self._native_series)
        values = val_count.field("values")
        counts = val_count.field("counts")

        if normalize:
            counts = pc.divide(*cast_for_truediv(counts, pc.sum(counts)))

        val_count = pa.Table.from_arrays(
            [values, counts], names=[index_name_, value_name_]
        )

        if sort:
            val_count = val_count.sort_by([(value_name_, "descending")])

        return ArrowDataFrame(
            val_count,
            backend_version=self._backend_version,
        )

    def zip_with(self: Self, mask: Self, other: Self) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        mask = mask._native_series.combine_chunks()
        return self._from_native_series(
            pc.if_else(
                mask,
                self._native_series,
                other._native_series,
            )
        )

    def sample(
        self: Self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> Self:
        import numpy as np  # ignore-banned-import
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        num_rows = len(self)

        if n is None and fraction is not None:
            n = int(num_rows * fraction)

        rng = np.random.default_rng(seed=seed)
        idx = np.arange(0, num_rows)
        mask = rng.choice(idx, size=n, replace=with_replacement)

        return self._from_native_series(pc.take(ser, mask))

    def fill_null(self: Self, value: Any) -> Self:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        dtype = ser.type

        return self._from_native_series(pc.fill_null(ser, pa.scalar(value, dtype)))

    def to_frame(self: Self) -> ArrowDataFrame:
        import pyarrow as pa  # ignore-banned-import()

        from narwhals._arrow.dataframe import ArrowDataFrame

        df = pa.Table.from_arrays([self._native_series], names=[self.name])
        return ArrowDataFrame(df, backend_version=self._backend_version)

    def to_pandas(self: Self) -> Any:
        import pandas as pd  # ignore-banned-import()

        return pd.Series(self._native_series, name=self.name)

    def is_duplicated(self: Self) -> ArrowSeries:
        return self.to_frame().is_duplicated().alias(self.name)

    def is_unique(self: Self) -> ArrowSeries:
        return self.to_frame().is_unique().alias(self.name)

    def is_first_distinct(self: Self) -> Self:
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        row_number = pa.array(np.arange(len(self)))
        col_token = generate_unique_token(n_bytes=8, columns=[self.name])
        first_distinct_index = (
            pa.Table.from_arrays([self._native_series], names=[self.name])
            .append_column(col_token, row_number)
            .group_by(self.name)
            .aggregate([(col_token, "min")])
            .column(f"{col_token}_min")
        )

        return self._from_native_series(pc.is_in(row_number, first_distinct_index))

    def is_last_distinct(self: Self) -> Self:
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        row_number = pa.array(np.arange(len(self)))
        col_token = generate_unique_token(n_bytes=8, columns=[self.name])
        last_distinct_index = (
            pa.Table.from_arrays([self._native_series], names=[self.name])
            .append_column(col_token, row_number)
            .group_by(self.name)
            .aggregate([(col_token, "max")])
            .column(f"{col_token}_max")
        )

        return self._from_native_series(pc.is_in(row_number, last_distinct_index))

    def is_sorted(self: Self, *, descending: bool = False) -> bool:
        if not isinstance(descending, bool):
            msg = f"argument 'descending' should be boolean, found {type(descending)}"
            raise TypeError(msg)
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        if descending:
            return pc.all(pc.greater_equal(ser[:-1], ser[1:]))  # type: ignore[no-any-return]
        else:
            return pc.all(pc.less_equal(ser[:-1], ser[1:]))  # type: ignore[no-any-return]

    def unique(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(pc.unique(self._native_series))

    def sort(
        self: Self, *, descending: bool = False, nulls_last: bool = False
    ) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        series = self._native_series
        order = "descending" if descending else "ascending"
        null_placement = "at_end" if nulls_last else "at_start"
        sorted_indices = pc.array_sort_indices(
            series, order=order, null_placement=null_placement
        )

        return self._from_native_series(pc.take(series, sorted_indices))

    def to_dummies(
        self: Self, *, separator: str = "_", drop_first: bool = False
    ) -> ArrowDataFrame:
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import()

        from narwhals._arrow.dataframe import ArrowDataFrame

        series = self._native_series
        da = series.dictionary_encode().combine_chunks()

        columns = np.zeros((len(da.dictionary), len(da)), np.uint8)
        columns[da.indices, np.arange(len(da))] = 1
        names = [f"{self._name}{separator}{v}" for v in da.dictionary]

        return ArrowDataFrame(
            pa.Table.from_arrays(columns, names=names),
            backend_version=self._backend_version,
        ).select(*sorted(names)[int(drop_first) :])

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Any:
        import pyarrow.compute as pc  # ignore-banned-import()

        return pc.quantile(self._native_series, q=quantile, interpolation=interpolation)[
            0
        ]

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        return self._from_native_series(self._native_series[offset::n])

    def clip(
        self: Self, lower_bound: Any | None = None, upper_bound: Any | None = None
    ) -> Self:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._native_series
        arr = pc.max_element_wise(arr, pa.scalar(lower_bound, type=arr.type))
        arr = pc.min_element_wise(arr, pa.scalar(upper_bound, type=arr.type))

        return self._from_native_series(arr)

    def to_arrow(self: Self) -> pa.Array:
        return self._native_series.combine_chunks()

    def mode(self: Self) -> ArrowSeries:
        plx = self.__narwhals_namespace__()
        col_token = generate_unique_token(n_bytes=8, columns=[self.name])
        return self.value_counts(name=col_token, normalize=False).filter(
            plx.col(col_token) == plx.col(col_token).max()
        )[self.name]

    def __iter__(self: Self) -> Iterator[Any]:
        yield from self._native_series.__iter__()

    @property
    def shape(self) -> tuple[int]:
        return (len(self._native_series),)

    @property
    def dt(self) -> ArrowSeriesDateTimeNamespace:
        return ArrowSeriesDateTimeNamespace(self)

    @property
    def cat(self) -> ArrowSeriesCatNamespace:
        return ArrowSeriesCatNamespace(self)

    @property
    def str(self) -> ArrowSeriesStringNamespace:
        return ArrowSeriesStringNamespace(self)


class ArrowSeriesDateTimeNamespace:
    def __init__(self: Self, series: ArrowSeries) -> None:
        self._arrow_series = series

    def to_string(self: Self, format: str) -> ArrowSeries:  # noqa: A002
        import pyarrow.compute as pc  # ignore-banned-import()

        # PyArrow differs from other libraries in that %S also prints out
        # the fractional part of the second...:'(
        # https://arrow.apache.org/docs/python/generated/pyarrow.compute.strftime.html
        format = format.replace("%S.%f", "%S").replace("%S%.f", "%S")
        return self._arrow_series._from_native_series(
            pc.strftime(self._arrow_series._native_series, format)
        )

    def date(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            self._arrow_series._native_series.cast(pa.date64())
        )

    def year(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.year(self._arrow_series._native_series)
        )

    def month(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.month(self._arrow_series._native_series)
        )

    def day(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.day(self._arrow_series._native_series)
        )

    def hour(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.hour(self._arrow_series._native_series)
        )

    def minute(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.minute(self._arrow_series._native_series)
        )

    def second(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.second(self._arrow_series._native_series)
        )

    def millisecond(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.millisecond(self._arrow_series._native_series)
        )

    def microsecond(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        result = pc.add(pc.multiply(pc.millisecond(arr), 1000), pc.microsecond(arr))

        return self._arrow_series._from_native_series(result)

    def nanosecond(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        result = pc.add(
            pc.multiply(self.microsecond()._native_series, 1000), pc.nanosecond(arr)
        )
        return self._arrow_series._from_native_series(result)

    def ordinal_day(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.day_of_year(self._arrow_series._native_series)
        )

    def total_minutes(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        unit = arr.type.unit

        unit_to_minutes_factor = {
            "s": 60,  # seconds
            "ms": 60 * 1e3,  # milli
            "us": 60 * 1e6,  # micro
            "ns": 60 * 1e9,  # nano
        }

        factor = pa.scalar(unit_to_minutes_factor[unit], type=pa.int64())
        return self._arrow_series._from_native_series(
            pc.cast(pc.divide(arr, factor), pa.int64())
        )

    def total_seconds(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        unit = arr.type.unit

        unit_to_seconds_factor = {
            "s": 1,  # seconds
            "ms": 1e3,  # milli
            "us": 1e6,  # micro
            "ns": 1e9,  # nano
        }
        factor = pa.scalar(unit_to_seconds_factor[unit], type=pa.int64())

        return self._arrow_series._from_native_series(
            pc.cast(pc.divide(arr, factor), pa.int64())
        )

    def total_milliseconds(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        unit = arr.type.unit

        unit_to_milli_factor = {
            "s": 1e3,  # seconds
            "ms": 1,  # milli
            "us": 1e3,  # micro
            "ns": 1e6,  # nano
        }

        factor = pa.scalar(unit_to_milli_factor[unit], type=pa.int64())

        if unit == "s":
            return self._arrow_series._from_native_series(
                pc.cast(pc.multiply(arr, factor), pa.int64())
            )

        return self._arrow_series._from_native_series(
            pc.cast(pc.divide(arr, factor), pa.int64())
        )

    def total_microseconds(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        unit = arr.type.unit

        unit_to_micro_factor = {
            "s": 1e6,  # seconds
            "ms": 1e3,  # milli
            "us": 1,  # micro
            "ns": 1e3,  # nano
        }

        factor = pa.scalar(unit_to_micro_factor[unit], type=pa.int64())

        if unit in {"s", "ms"}:
            return self._arrow_series._from_native_series(
                pc.cast(pc.multiply(arr, factor), pa.int64())
            )
        return self._arrow_series._from_native_series(
            pc.cast(pc.divide(arr, factor), pa.int64())
        )

    def total_nanoseconds(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        unit = arr.type.unit

        unit_to_nano_factor = {
            "s": 1e9,  # seconds
            "ms": 1e6,  # milli
            "us": 1e3,  # micro
            "ns": 1,  # nano
        }

        factor = pa.scalar(unit_to_nano_factor[unit], type=pa.int64())

        return self._arrow_series._from_native_series(
            pc.cast(pc.multiply(arr, factor), pa.int64())
        )


class ArrowSeriesCatNamespace:
    def __init__(self, series: ArrowSeries) -> None:
        self._arrow_series = series

    def get_categories(self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()

        ca = self._arrow_series._native_series
        # TODO(Unassigned): this looks potentially expensive - is there no better way?
        # https://github.com/narwhals-dev/narwhals/issues/464
        out = pa.chunked_array(
            [pa.concat_arrays([x.dictionary for x in ca.chunks]).unique()]
        )
        return self._arrow_series._from_native_series(out)


class ArrowSeriesStringNamespace:
    def __init__(self: Self, series: ArrowSeries) -> None:
        self._arrow_series = series

    def len_chars(self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.utf8_length(self._arrow_series._native_series)
        )

    def replace(
        self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        method = "replace_substring" if literal else "replace_substring_regex"
        return self._arrow_series._from_native_series(
            getattr(pc, method)(
                self._arrow_series._native_series,
                pattern=pattern,
                replacement=value,
                max_replacements=n,
            )
        )

    def replace_all(
        self, pattern: str, value: str, *, literal: bool = False
    ) -> ArrowSeries:
        return self.replace(pattern, value, literal=literal, n=-1)

    def strip_chars(self: Self, characters: str | None = None) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        whitespace = " \t\n\r\v\f"
        return self._arrow_series._from_native_series(
            pc.utf8_trim(
                self._arrow_series._native_series,
                characters or whitespace,
            )
        )

    def starts_with(self: Self, prefix: str) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.equal(self.slice(0, len(prefix))._native_series, prefix)
        )

    def ends_with(self: Self, suffix: str) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.equal(self.slice(-len(suffix))._native_series, suffix)
        )

    def contains(self: Self, pattern: str, *, literal: bool = False) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        check_func = pc.match_substring if literal else pc.match_substring_regex
        return self._arrow_series._from_native_series(
            check_func(self._arrow_series._native_series, pattern)
        )

    def slice(self: Self, offset: int, length: int | None = None) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        stop = offset + length if length else None
        return self._arrow_series._from_native_series(
            pc.utf8_slice_codeunits(
                self._arrow_series._native_series, start=offset, stop=stop
            ),
        )

    def to_datetime(self: Self, format: str | None = None) -> ArrowSeries:  # noqa: A002
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.strptime(self._arrow_series._native_series, format=format, unit="us")
        )

    def to_uppercase(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.utf8_upper(self._arrow_series._native_series),
        )

    def to_lowercase(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.utf8_lower(self._arrow_series._native_series),
        )
