"""Minimal wrapper for a native lazy query."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from narwhals._plan.compliant.typing import Native
from narwhals._utils import Version

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.dataframe import DataFrame as NwDataFrame
    from narwhals._translate import ArrowStreamExportable
    from narwhals._utils import Implementation
    from narwhals.schema import Schema
    from narwhals.typing import EagerAllowed, IntoBackend


MAIN = Version.MAIN


class CompliantLazyFrame(Protocol[Native]):
    """Clean-slate rework of `CompliantFrame`-based design.

    Focused on features for `LogicalPlan`:
    - storing a native object
    - performing import/export operations
    - exposing the schema
    """

    implementation: ClassVar[Implementation]

    @classmethod
    def from_native(
        cls: type[CompliantLazyFrame[Any]], native: Native, /, version: Version = MAIN
    ) -> CompliantLazyFrame[Native]: ...
    @classmethod
    def from_arrow(cls, frame: pa.Table, /, version: Version = MAIN) -> Self: ...
    @classmethod
    def from_arrow_c_stream(
        cls,
        exportable: ArrowStreamExportable,
        /,
        version: Version = MAIN,
        *,
        requested_schema: object | None = None,
    ) -> Self: ...
    @classmethod
    def from_pandas(cls, frame: pd.DataFrame, /, version: Version = MAIN) -> Self: ...
    @classmethod
    def from_polars(cls, frame: pl.DataFrame, /, version: Version = MAIN) -> Self: ...
    @classmethod
    def from_narwhals(cls, frame: NwDataFrame[Any, Any], /) -> Self: ...
    @classmethod
    def from_compliant(cls, frame: CompliantDataFrame[Any, Any, Any], /) -> Self: ...
    def collect_arrow(self) -> pa.Table: ...
    def collect_pandas(self) -> pd.DataFrame: ...
    def collect_polars(self) -> pl.DataFrame: ...
    def collect_narwhals(
        self, backend: IntoBackend[EagerAllowed]
    ) -> NwDataFrame[Any, Any]: ...
    def collect_schema(self) -> Schema: ...
    @property
    def columns(self) -> Sequence[str]: ...
    @property
    def native(self) -> Native: ...
    @property
    def version(self) -> Version: ...
    def __hash__(self) -> int: ...
