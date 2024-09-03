from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

from narwhals._pyspark.utils import translate_pandas_api_dtype
from narwhals.dependencies import get_pyspark_sql
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from pyspark.pandas import Series
    from typing_extensions import Self

    from narwhals.dtypes import DType


class PySparkSeries:
    def __init__(self, native_series: Series, *, name: str) -> None:
        self._name = name
        self._native_series = native_series
        self._implementation = Implementation.PYSPARK

    def __native_namespace__(self) -> Any:
        # TODO maybe not the best namespace to return
        return get_pyspark_sql()

    def __narwhals_series__(self) -> Self:
        return self

    def _from_native_series(self, series: Series) -> Self:
        return self.__class__(series, name=self._name)

    @classmethod
    def _from_iterable(cls: type[Self], data: Iterable[Any], name: str) -> Self:
        from pyspark.pandas import Series  # ignore-banned-import()

        return cls(Series([data]), name=name)

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int]:
        return self._native_series.shape  # type: ignore[no-any-return]

    @property
    def dtype(self) -> DType:
        return translate_pandas_api_dtype(self._native_series)

    def alias(self, name: str) -> Self:
        return self._from_native_series(self._native_series.rename(name))
