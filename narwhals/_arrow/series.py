from __future__ import annotations

from typing import Any

from narwhals._arrow.utils import validate_column_comparand
from narwhals._pandas_like.series import PandasSeries
from narwhals.dependencies import get_pyarrow


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
        if self._implementation == "pandas":
            pd = get_pandas()

            if parse_version(pd.__version__) < parse_version("3.0.0"):
                self._use_copy_false = True
            else:  # pragma: no cover
                pass
        else:  # pragma: no cover
            pass

    def _from_series(self, series: Any) -> Self:
        return self.__class__(
            series,
            implementation=self._implementation,
            name=self._name,
        )

    def __add__(self, other: Any) -> PandasSeries:
        ser = self._series
        pc = get_pyarrow().compute
        other = validate_column_comparand(self._series.index, other)
        return self._from_series(pc.add(ser, other))

    def _rename(self, series: Any, name: str) -> Any:
        return self.__class__(
            series,
            implementation=self._implementation,
            name=name,
        )
