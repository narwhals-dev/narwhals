from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

from narwhals._arrow.namespace import ArrowNamespace
from narwhals._arrow.utils import item
from narwhals._arrow.utils import reverse_translate_dtype
from narwhals._arrow.utils import translate_dtype
from narwhals._arrow.utils import validate_column_comparand
from narwhals._pandas_like.utils import native_series_from_iterable
from narwhals.dependencies import get_pyarrow
from narwhals.dependencies import get_pyarrow_compute

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType


class ArrowSeries:
    def __init__(
        self, native_series: Any, *, name: str, backend_version: tuple[int, ...]
    ) -> None:
        self._name = name
        self._native_series = native_series
        self._implementation = "arrow"  # for compatibility with PandasSeries
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
        return cls(
            native_series_from_iterable(
                data,
                name=name,
                index=None,
                implementation="arrow",
            ),
            name=name,
            backend_version=backend_version,
        )

    def __len__(self) -> int:
        return len(self._native_series)

    def __add__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        other = validate_column_comparand(other)
        return self._from_native_series(pc.add(self._native_series, other))

    def __sub__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        other = validate_column_comparand(other)
        return self._from_native_series(pc.subtract(self._native_series, other))

    def __mul__(self, other: Any) -> Self:
        pc = get_pyarrow_compute()
        other = validate_column_comparand(other)
        return self._from_native_series(pc.multiply(self._native_series, other))

    def mean(self) -> int:
        pc = get_pyarrow_compute()
        return item(self._backend_version, pc.mean(self._native_series))  # type: ignore[no-any-return]

    def std(self, ddof: int = 1) -> int:
        pc = get_pyarrow_compute()
        return item(self._backend_version, pc.stddev(self._native_series, ddof=ddof))  # type: ignore[no-any-return]

    def __narwhals_namespace__(self) -> ArrowNamespace:
        return ArrowNamespace(backend_version=self._backend_version)

    @property
    def name(self) -> str:
        return self._name

    def __narwhals_series__(self) -> Self:
        return self

    def __getitem__(self, idx: int) -> Any:
        return self._native_series[idx]

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

    def cast(self, dtype: DType) -> Self:
        pc = get_pyarrow_compute()
        ser = self._native_series
        dtype = reverse_translate_dtype(dtype)
        return self._from_native_series(pc.cast(ser, dtype))

    @property
    def shape(self) -> tuple[int]:
        return (len(self._native_series),)

    @property
    def dt(self) -> ArrowSeriesDateTimeNamespace:
        return ArrowSeriesDateTimeNamespace(self)

    @property
    def cat(self) -> ArrowSeriesCatNamespace:
        return ArrowSeriesCatNamespace(self)


class ArrowSeriesDateTimeNamespace:
    def __init__(self, series: ArrowSeries) -> None:
        self._series = series

    def to_string(self, format: str) -> ArrowSeries:  # noqa: A002
        pc = get_pyarrow_compute()
        # PyArrow differs from other libraries in that %S also prints out
        # the fractional part of the second...:'(
        # https://arrow.apache.org/docs/python/generated/pyarrow.compute.strftime.html
        format = format.replace("%S.%f", "%S").replace("%S%.f", "%S")
        return self._series._from_native_series(
            pc.strftime(self._series._native_series, format)
        )


class ArrowSeriesCatNamespace:
    def __init__(self, series: ArrowSeries) -> None:
        self._series = series

    def get_categories(self) -> ArrowSeries:
        pa = get_pyarrow()
        ca = self._series._native_series
        # todo: this looks potentially expensive - is there no better way?
        out = pa.chunked_array(
            [pa.concat_arrays([x.dictionary for x in ca.chunks]).unique()]
        )
        return self._series._from_native_series(out)
