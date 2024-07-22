from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence
from typing import overload

from narwhals._arrow.utils import cast_for_truediv
from narwhals._arrow.utils import floordiv_compat
from narwhals._arrow.utils import reverse_translate_dtype
from narwhals._arrow.utils import translate_dtype
from narwhals._arrow.utils import validate_column_comparand
from narwhals.dependencies import get_numpy
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_pyarrow
from narwhals.dependencies import get_pyarrow_compute
from narwhals.utils import generate_unique_token

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals.dtypes import DType


class ArrowSeries:
    def __init__(
        self, native_series: Any, *, name: str, backend_version: tuple[int, ...]
    ) -> None:
        self._name = name
        self._native_series = native_series
        self._implementation = "arrow"  # for compatibility with PandasLikeSeries
        self._backend_version = backend_version

    def _from_native_series(self, series: Any) -> Self:
        pa = get_pyarrow()
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
        pa = get_pyarrow()
        return cls(
            pa.chunked_array([data]),
            name=name,
            backend_version=backend_version,
        )

    def __len__(self) -> int:
        return len(self._native_series)

    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.equal(ser, other))

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.not_equal(ser, other))

    def __ge__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.greater_equal(ser, other))

    def __gt__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.greater(ser, other))

    def __le__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.less_equal(ser, other))

    def __lt__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.less(ser, other))

    def __and__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.and_kleene(ser, other))

    def __or__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.or_kleene(ser, other))

    def __add__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        other = validate_column_comparand(other)
        return self._from_native_series(pc.add(self._native_series, other))

    def __radd__(self, other: Any) -> Self:
        return self + other  # type: ignore[no-any-return]

    def __sub__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        other = validate_column_comparand(other)
        return self._from_native_series(pc.subtract(self._native_series, other))

    def __rsub__(self, other: Any) -> Self:
        return (self - other) * (-1)  # type: ignore[no-any-return]

    def __mul__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        other = validate_column_comparand(other)
        return self._from_native_series(pc.multiply(self._native_series, other))

    def __rmul__(self, other: Any) -> Self:
        return self * other  # type: ignore[no-any-return]

    def __pow__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        return self._from_native_series(pc.power(ser, other))

    def __rpow__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
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
        pa = get_pyarrow()
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        if not isinstance(other, (pa.Array, pa.ChunkedArray)):
            # scalar
            other = pa.scalar(other)
        return self._from_native_series(pc.divide(*cast_for_truediv(ser, other)))

    def __rtruediv__(self, other: Any) -> Self:
        pa = get_pyarrow()
        pc = get_pyarrow_compute()
        ser = self._native_series
        other = validate_column_comparand(other)
        if not isinstance(other, (pa.Array, pa.ChunkedArray)):
            # scalar
            other = pa.scalar(other)
        return self._from_native_series(pc.divide(*cast_for_truediv(other, ser)))

    def __invert__(self) -> Self:
        pc = get_pyarrow_compute()
        return self._from_native_series(pc.invert(self._native_series))

    def len(self) -> int:
        return len(self._native_series)

    def filter(self, other: Any) -> Self:
        other = validate_column_comparand(other)
        return self._from_native_series(self._native_series.filter(other))

    def mean(self) -> int:
        pc = get_pyarrow_compute()
        return pc.mean(self._native_series)  # type: ignore[no-any-return]

    def min(self) -> int:
        pc = get_pyarrow_compute()
        return pc.min(self._native_series)  # type: ignore[no-any-return]

    def max(self) -> int:
        pc = get_pyarrow_compute()
        return pc.max(self._native_series)  # type: ignore[no-any-return]

    def sum(self) -> int:
        pc = get_pyarrow_compute()
        return pc.sum(self._native_series)  # type: ignore[no-any-return]

    def drop_nulls(self) -> ArrowSeries:
        pc = get_pyarrow_compute()
        return self._from_native_series(pc.drop_null(self._native_series))

    def std(self, ddof: int = 1) -> int:
        pc = get_pyarrow_compute()
        return pc.stddev(self._native_series, ddof=ddof)  # type: ignore[no-any-return]

    def count(self) -> int:
        pc = get_pyarrow_compute()
        return pc.count(self._native_series)  # type: ignore[no-any-return]

    def n_unique(self) -> int:
        pc = get_pyarrow_compute()
        unique_values = pc.unique(self._native_series)
        count_unique = pc.count(unique_values, mode="all")
        return count_unique.as_py()  # type: ignore[no-any-return]

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._arrow.namespace import ArrowNamespace

        return ArrowNamespace(backend_version=self._backend_version)

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
        return self._from_native_series(self._native_series[idx])

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
    def dtype(self) -> DType:
        return translate_dtype(self._native_series.type)

    def abs(self) -> Self:
        pc = get_pyarrow_compute()
        return self._from_native_series(pc.abs(self._native_series))

    def cum_sum(self) -> Self:
        pc = get_pyarrow_compute()
        return self._from_native_series(pc.cumulative_sum(self._native_series))

    def diff(self) -> Self:
        pc = get_pyarrow_compute()
        return self._from_native_series(
            pc.pairwise_diff(self._native_series.combine_chunks())
        )

    def any(self) -> bool:
        pc = get_pyarrow_compute()
        return pc.any(self._native_series)  # type: ignore[no-any-return]

    def all(self) -> bool:
        pc = get_pyarrow_compute()
        return pc.all(self._native_series)  # type: ignore[no-any-return]

    def is_empty(self) -> bool:
        return len(self) == 0

    def is_null(self) -> Self:
        ser = self._native_series
        return self._from_native_series(ser.is_null())

    def cast(self, dtype: DType) -> Self:
        pc = get_pyarrow_compute()
        ser = self._native_series
        dtype = reverse_translate_dtype(dtype)
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
        pc = get_pyarrow_compute()
        pa = get_pyarrow()
        value_set = pa.array(other)
        ser = self._native_series
        return self._from_native_series(pc.is_in(ser, value_set=value_set))

    def arg_true(self) -> Self:
        np = get_numpy()
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
            return self._native_series[0].as_py()
        return self._native_series[index].as_py()

    def value_counts(self: Self, *, sort: bool = False, parallel: bool = False) -> Any:  # noqa: ARG002
        """Parallel is unused, exists for compatibility"""
        from narwhals._arrow.dataframe import ArrowDataFrame

        pc = get_pyarrow_compute()
        pa = get_pyarrow()

        name_ = (
            "index" if self._native_series._name is None else self._native_series._name
        )

        val_count = pc.value_counts(self._native_series)
        val_count = pa.Table.from_arrays(
            [val_count.field("values"), val_count.field("counts")], names=[name_, "count"]
        )

        if sort:
            val_count = val_count.sort_by([("count", "descending")])

        return ArrowDataFrame(
            val_count,
            backend_version=self._backend_version,
        )

    def zip_with(self: Self, mask: Self, other: Self) -> Self:
        pc = get_pyarrow_compute()

        return self._from_native_series(
            pc.replace_with_mask(
                self._native_series.combine_chunks(),
                pc.invert(mask._native_series.combine_chunks()),
                other._native_series.combine_chunks(),
            )
        )

    def sample(
        self: Self,
        n: int | None = None,
        fraction: float | None = None,
        *,
        with_replacement: bool = False,
    ) -> Self:
        np = get_numpy()
        pc = get_pyarrow_compute()
        ser = self._native_series
        num_rows = len(self)

        if n is None and fraction is not None:
            n = int(num_rows * fraction)

        idx = np.arange(0, num_rows)
        mask = np.random.choice(idx, size=n, replace=with_replacement)
        return self._from_native_series(pc.take(ser, mask))

    def fill_null(self: Self, value: Any) -> Self:
        pa = get_pyarrow()
        pc = get_pyarrow_compute()
        ser = self._native_series
        dtype = ser.type

        return self._from_native_series(pc.fill_null(ser, pa.scalar(value, dtype)))

    def to_frame(self: Self) -> ArrowDataFrame:
        from narwhals._arrow.dataframe import ArrowDataFrame

        pa = get_pyarrow()
        df = pa.Table.from_arrays([self._native_series], names=[self.name])
        return ArrowDataFrame(df, backend_version=self._backend_version)

    def to_pandas(self: Self) -> Any:
        pd = get_pandas()
        return pd.Series(self._native_series, name=self.name)

    def is_duplicated(self: Self) -> ArrowSeries:
        return self.to_frame().is_duplicated().alias(self.name)

    def is_unique(self: Self) -> ArrowSeries:
        return self.to_frame().is_unique().alias(self.name)

    def is_first_distinct(self: Self) -> Self:
        np = get_numpy()
        pa = get_pyarrow()
        pc = get_pyarrow_compute()

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
        np = get_numpy()
        pa = get_pyarrow()
        pc = get_pyarrow_compute()

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
        pc = get_pyarrow_compute()
        ser = self._native_series
        if descending:
            return pc.all(pc.greater_equal(ser[:-1], ser[1:])).as_py()  # type: ignore[no-any-return]
        else:
            return pc.all(pc.less_equal(ser[:-1], ser[1:])).as_py()  # type: ignore[no-any-return]

    def unique(self: Self) -> ArrowSeries:
        pc = get_pyarrow_compute()
        return self._from_native_series(pc.unique(self._native_series))

    def sort(self: Self, *, descending: bool = False) -> ArrowSeries:
        pc = get_pyarrow_compute()
        series = self._native_series
        order = "descending" if descending else "ascending"
        sorted_indices = pc.array_sort_indices(
            series, order=order, null_placement="at_start"
        )

        return self._from_native_series(pc.take(series, sorted_indices))

    def to_dummies(
        self: Self, *, separator: str = "_", drop_first: bool = False
    ) -> ArrowDataFrame:
        from narwhals._arrow.dataframe import ArrowDataFrame

        pa = get_pyarrow()
        pc = get_pyarrow_compute()
        series = self._native_series
        unique_values = self.unique().sort()._native_series
        columns = [pc.cast(pc.equal(series, v), pa.uint8()) for v in unique_values][
            int(drop_first) :
        ]
        names = [f"{self._name}{separator}{v}" for v in unique_values][int(drop_first) :]

        return ArrowDataFrame(
            pa.Table.from_arrays(columns, names=names),
            backend_version=self._backend_version,
        )

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
    def __init__(self, series: ArrowSeries) -> None:
        self._arrow_series = series

    def to_string(self, format: str) -> ArrowSeries:  # noqa: A002
        pc = get_pyarrow_compute()
        # PyArrow differs from other libraries in that %S also prints out
        # the fractional part of the second...:'(
        # https://arrow.apache.org/docs/python/generated/pyarrow.compute.strftime.html
        format = format.replace("%S.%f", "%S").replace("%S%.f", "%S")
        return self._arrow_series._from_native_series(
            pc.strftime(self._arrow_series._native_series, format)
        )


class ArrowSeriesCatNamespace:
    def __init__(self, series: ArrowSeries) -> None:
        self._arrow_series = series

    def get_categories(self) -> ArrowSeries:
        pa = get_pyarrow()
        ca = self._arrow_series._native_series
        # TODO(Unassigned): this looks potentially expensive - is there no better way?
        out = pa.chunked_array(
            [pa.concat_arrays([x.dictionary for x in ca.chunks]).unique()]
        )
        return self._arrow_series._from_native_series(out)


class ArrowSeriesStringNamespace:
    def __init__(self: Self, series: ArrowSeries) -> None:
        self._arrow_series = series

    def starts_with(self: Self, prefix: str) -> ArrowSeries:
        pc = get_pyarrow_compute()
        return self._arrow_series._from_native_series(
            pc.equal(self.slice(0, len(prefix))._native_series, prefix)
        )

    def ends_with(self: Self, suffix: str) -> ArrowSeries:
        pc = get_pyarrow_compute()
        return self._arrow_series._from_native_series(
            pc.equal(self.slice(-len(suffix))._native_series, suffix)
        )

    def contains(self: Self, pattern: str, *, literal: bool = False) -> ArrowSeries:
        pc = get_pyarrow_compute()
        check_func = pc.match_substring if literal else pc.match_substring_regex
        return self._arrow_series._from_native_series(
            check_func(self._arrow_series._native_series, pattern)
        )

    def slice(self: Self, offset: int, length: int | None = None) -> ArrowSeries:
        pc = get_pyarrow_compute()
        stop = offset + length if length else None
        return self._arrow_series._from_native_series(
            pc.utf8_slice_codeunits(
                self._arrow_series._native_series, start=offset, stop=stop
            ),
        )

    def to_datetime(self: Self, format: str | None = None) -> ArrowSeries:  # noqa: A002
        pc = get_pyarrow_compute()
        return self._arrow_series._from_native_series(
            pc.strptime(self._arrow_series._native_series, format=format, unit="us")
        )

    def to_uppercase(self: Self) -> ArrowSeries:
        pc = get_pyarrow_compute()
        return self._arrow_series._from_native_series(
            pc.utf8_upper(self._arrow_series._native_series),
        )

    def to_lowercase(self: Self) -> ArrowSeries:
        pc = get_pyarrow_compute()
        return self._arrow_series._from_native_series(
            pc.utf8_lower(self._arrow_series._native_series),
        )
