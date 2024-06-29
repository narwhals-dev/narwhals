from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._pandas_like.series import PandasSeries
from narwhals._pandas_like.series import PandasSeriesDateTimeNamespace
from narwhals.dependencies import get_pyarrow_compute

if TYPE_CHECKING:
    from typing_extensions import Self


class ArrowSeries(PandasSeries):
    def __init__(
        self,
        series: Any,
        *,
        name: str,
        implementation: str,
    ) -> None:
        self._name = name
        self._series = series
        self._implementation = implementation
        self._use_copy_false = False
        self._implementation = implementation

    def _from_series(self, series: Any) -> Self:
        return self.__class__(
            series,
            implementation=self._implementation,
            name=self._name,
        )

    def __getitem__(self, idx: int) -> Any:
        return self._series[idx]

    def to_list(self) -> Any:
        return self._series.to_pylist()

    def to_numpy(self) -> Any:
        return self._series.to_numpy()

    def alias(self, name: str) -> Self:
        return self.__class__(
            self._series,
            implementation=self._implementation,
            name=name,
        )

    @property
    def shape(self) -> tuple[int]:
        return (len(self._series),)

    @property
    def dt(self) -> ArrowSeriesDateTimeNamespace:
        return ArrowSeriesDateTimeNamespace(self)


class ArrowSeriesDateTimeNamespace(PandasSeriesDateTimeNamespace):
    def __init__(self, series: ArrowSeries) -> None:
        self._series: ArrowSeries = series

    def to_string(self, format: str) -> ArrowSeries:  # noqa: A002
        pc = get_pyarrow_compute()
        # PyArrow differs from other libraries in that %S also prints out
        # the fractional part of the second...:'(
        # https://arrow.apache.org/docs/python/generated/pyarrow.compute.strftime.html
        format = format.replace("%S.%f", "%S").replace("%S%.f", "%S")
        return self._series._from_series(pc.strftime(self._series._series, format))
