from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from narwhals._plan._version import into_version
from narwhals._plan.common import todo
from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.dataframe import DataFrame as NwDataFrame
    from narwhals._translate import ArrowStreamExportable
    from narwhals.schema import Schema


MAIN = Version.MAIN


class PolarsLazyFrame(CompliantLazyFrame[pl.LazyFrame]):
    __slots__ = ("_input_schema", "_native", "_version")
    implementation: ClassVar = Implementation.POLARS
    _native: pl.LazyFrame
    _version: Version
    _input_schema: Schema | None

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
        obj._input_schema = None
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

    @property
    def input_schema(self) -> Schema:
        if self._input_schema is None:
            self._input_schema = into_version(self.version).schema.from_polars(
                self.native.collect_schema()
            )
        return self._input_schema

    @property
    def native(self) -> pl.LazyFrame:
        return self._native

    @property
    def version(self) -> Version:
        return self._version

    collect_schema = todo()
    collect_arrow = todo()
    collect_polars = todo()
    collect_pandas = todo()

    from_compliant = from_narwhals
    from_arrow_c_stream = from_arrow
