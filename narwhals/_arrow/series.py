from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

from narwhals._arrow.namespace import ArrowNamespace
from narwhals._arrow.utils import reverse_translate_dtype
from narwhals._arrow.utils import translate_dtype
from narwhals._pandas_like.utils import native_series_from_iterable
from narwhals.dependencies import get_pyarrow
from narwhals.dependencies import get_pyarrow_compute

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType


class ArrowSeries:
    def __init__(
        self,
        series: Any,
        *,
        name: str,
    ) -> None:
        self._name = name
        self._series = series
        self._implementation = "arrow"  # for compatibility with PandasSeries

    def _from_series(self, series: Any) -> Self:
        return self.__class__(
            series,
            name=self._name,
        )

    @classmethod
    def _from_iterable(cls: type[Self], data: Iterable[Any], name: str) -> Self:
        return cls(
            native_series_from_iterable(
                data, name=name, index=None, implementation="arrow"
            ),
            name=name,
        )

    def __len__(self) -> int:
        return len(self._series)

    def __narwhals_namespace__(self) -> ArrowNamespace:
        return ArrowNamespace()

    @property
    def name(self) -> str:
        return self._name

    def __narwhals_series__(self) -> Self:
        return self

    def __getitem__(self, idx: int) -> Any:
        return self._series[idx]

    def to_list(self) -> Any:
        return self._series.to_pylist()

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        return self._series.__array__(dtype=dtype, copy=copy)

    def to_numpy(self) -> Any:
        return self._series.to_numpy()

    def alias(self, name: str) -> Self:
        return self.__class__(
            self._series,
            name=name,
        )

    @property
    def dtype(self) -> DType:
        return translate_dtype(self._series.type)

    def abs(self) -> Self:
        pc = get_pyarrow_compute()
        return self._from_series(pc.abs(self._series))

    def cum_sum(self) -> Self:
        pc = get_pyarrow_compute()
        return self._from_series(pc.cumulative_sum(self._series))

    def any(self) -> bool:
        pc = get_pyarrow_compute()
        return pc.any(self._series)  # type: ignore[no-any-return]

    def all(self) -> bool:
        pc = get_pyarrow_compute()
        return pc.all(self._series)  # type: ignore[no-any-return]

    def is_empty(self) -> bool:
        return len(self) == 0

    def cast(self, dtype: DType) -> Self:
        pc = get_pyarrow_compute()
        ser = self._series
        dtype = reverse_translate_dtype(dtype)
        return self._from_series(pc.cast(ser, dtype))

    @property
    def shape(self) -> tuple[int]:
        return (len(self._series),)

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
        return self._series._from_series(pc.strftime(self._series._series, format))


class ArrowSeriesCatNamespace:
    def __init__(self, series: ArrowSeries) -> None:
        self._series = series

    def get_categories(self) -> ArrowSeries:
        pa = get_pyarrow()
        ca = self._series._series
        # todo: this looks potentially expensive - is there no better way?
        out = pa.chunked_array(
            [pa.concat_arrays([x.dictionary for x in ca.chunks]).unique()]
        )
        return self._series._from_series(out)
