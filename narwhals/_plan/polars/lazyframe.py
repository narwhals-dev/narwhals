from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from narwhals._plan.common import todo
from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
from narwhals._utils import Implementation, Version
from narwhals.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.dataframe import DataFrame as NwDataFrame
    from narwhals._translate import ArrowStreamExportable

MAIN = Version.MAIN


class PolarsLazyFrame(CompliantLazyFrame[pl.LazyFrame]):
    __slots__ = ("_native", "_version")
    implementation: ClassVar = Implementation.POLARS
    _native: pl.LazyFrame
    _version: Version

    @classmethod
    def from_arrow(
        cls, frame: pa.Table | ArrowStreamExportable, /, version: Version = MAIN, **_: Any
    ) -> Self:
        return cls.from_polars(pl.DataFrame(frame), version)

    @classmethod
    def from_native(cls, native: pl.LazyFrame, /, version: Version = MAIN) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._version = version
        return obj

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame, /, version: Version = MAIN) -> Self:
        return cls.from_polars(pl.from_pandas(frame), version)

    @classmethod
    def from_polars(cls, frame: pl.DataFrame, /, version: Version = MAIN) -> Self:
        return cls.from_native(frame.lazy(), version)

    @classmethod
    def from_narwhals(
        cls, frame: NwDataFrame[Any, Any] | CompliantDataFrame[Any, Any, Any]
    ) -> Self:
        return cls.from_polars(frame.to_polars(), frame.version)

    def collect_arrow(self) -> pa.Table:
        return self.collect_native().to_arrow()

    def collect_native(self) -> pl.DataFrame:
        return self.native.collect()

    def collect_pandas(self) -> pd.DataFrame:
        return self.collect_native().to_pandas()

    def collect_schema(self) -> Schema:
        return Schema.from_polars(self.native.collect_schema())

    @property
    def columns(self) -> Sequence[str]:
        return self.native.collect_schema().names()

    @property
    def native(self) -> pl.LazyFrame:
        return self._native

    @property
    def version(self) -> Version:
        return self._version

    from_compliant = from_narwhals
    from_arrow_c_stream = from_arrow
    collect_polars = collect_native

    collect_narwhals = todo()
