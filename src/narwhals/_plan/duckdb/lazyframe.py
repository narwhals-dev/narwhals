from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from narwhals._duckdb import utils
from narwhals._native import NativeDuckDB
from narwhals._plan._version import into_version
from narwhals._plan.common import todo
from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
from narwhals._plan.plans.visitors import ResolvedToCompliant
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.schema import Schema

Incomplete: TypeAlias = Any


class DuckDBLazyFrame(CompliantLazyFrame[NativeDuckDB]):
    __slots__ = ("_input_schema", "_native")
    implementation = Implementation.DUCKDB
    version = Version.MAIN
    _native: NativeDuckDB
    _input_schema: Schema | None

    collect_arrow = todo()
    collect_pandas = todo()
    collect_polars = todo()
    collect_schema = todo()
    from_arrow = todo()
    from_compliant = todo()
    from_narwhals = todo()

    @classmethod
    def from_native(cls, native: NativeDuckDB, /) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._input_schema = None
        return obj

    from_pandas = todo()
    from_polars = todo()
    scan_csv = todo()
    scan_parquet = todo()
    sink_parquet = todo()

    @property
    def input_schema(self) -> Schema:
        if self._input_schema is None:
            from_native = utils.native_to_narwhals_dtype
            tz = utils.DeferredTimeZone(self.native)
            native = zip(self.native.columns, self.native.types, strict=False)
            version = self.version
            tp_schema = into_version(version).schema
            self._input_schema = tp_schema(
                (name, (from_native(dtype, version, tz))) for name, dtype in native
            )
        return self._input_schema

    @property
    def native(self) -> NativeDuckDB:
        return self._native


DuckDBLazyFrame()


class DuckDBEvaluator(ResolvedToCompliant[NativeDuckDB]):
    implementation = Implementation.DUCKDB
    version = Version.MAIN
    collect = todo()
    concat_horizontal = todo()
    concat_vertical = todo()
    explode = todo()
    filter = todo()
    group_by = todo()
    group_by_names = todo()
    join = todo()
    join_asof = todo()
    map_function = todo()
    rename = todo()
    scan_csv = todo()
    scan_dataframe = todo()
    scan_empty = todo()
    scan_lazyframe = todo()
    scan_parquet = todo()
    select = todo()
    select_names = todo()
    sink_parquet = todo()
    slice = todo()
    sort = todo()
    unique = todo()
    unique_by = todo()
    unnest = todo()
    unpivot = todo()
    with_columns = todo()
    with_row_index = todo()
    with_row_index_by = todo()


DuckDBEvaluator()
