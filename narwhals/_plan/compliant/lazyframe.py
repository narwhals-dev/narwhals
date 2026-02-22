"""Minimal wrapper for a native lazy query."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, get_args

from narwhals._plan._version import into_version
from narwhals._plan.compliant.typing import Native
from narwhals._typing import _LazyFrameCollectImpl
from narwhals._utils import Implementation, Version, can_lazyframe_collect

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.dataframe import DataFrame as NwDataFrame
    from narwhals._translate import ArrowStreamExportable
    from narwhals.schema import Schema
    from narwhals.typing import EagerAllowed, IntoBackend

    CollectMap: TypeAlias = Mapping[
        _LazyFrameCollectImpl, Callable[[], pa.Table | pd.DataFrame | pl.DataFrame]
    ]


MAIN = Version.MAIN


class NarwhalsHash(Protocol):
    __slots__ = ()

    def __narwhals_hash_values__(self) -> Iterator[object]:
        """Yield one or more attributes to seed a hash.

        All backends *could* use a psuedo hash:

            if slots := self.__slots__:
                yield from (id(getattr(self, key)) for key in slots)
            else:
                yield from (id(value) for value in vars(self).values())

        Each of these could work as a pre-computed hash, but be careful

        Polars:

            seed: bytes = pl.LazyFrame(...).serialize()

        DuckDB:

            seed: str = duckdb.DuckDBPyRelation(...).sql_query()


        SQLFrame:

            seed: str = sqlframe.base.dataframe.BaseDataFrame(...).sql(optimize=False, pretty=False)


        <!--TODO @dangotbanned: Find what the docs refer to as "Unlike the standard hash code ..."
        The example gives a hash that ignores aliases (bad)
        -->

        [PySpark]:

            seed: int = pyspark.sql.DataFrame(...).semanticHash()


        [PySpark]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.semanticHash.html#pyspark.sql.DataFrame.semanticHash
        """
        ...

    def __hash__(self) -> int:
        return hash((type(self), *self.__narwhals_hash_values__()))

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        return hash(self) == hash(other)


class CompliantLazyFrame(NarwhalsHash, Protocol[Native]):
    """Clean-slate rework of `CompliantFrame`-based design.

    Focused on features for `LogicalPlan`:
    - storing a native object
    - performing import/export operations
    - exposing the schema
    """

    __slots__ = ()

    implementation: ClassVar[Implementation]

    def __narwhals_hash_values__(self) -> Iterator[object]:
        yield self.version, self.implementation, id(self.native)

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
    ) -> NwDataFrame[Any, Any]:
        mapping: CollectMap = {
            Implementation.PANDAS: self.collect_pandas,
            Implementation.PYARROW: self.collect_arrow,
            Implementation.POLARS: self.collect_polars,
        }
        impl = Implementation.from_backend(backend)
        if can_lazyframe_collect(impl):
            return into_version(self.version).dataframe.from_native(mapping[impl]())
        msg = f"Unsupported `backend` value.\nExpected one of {get_args(_LazyFrameCollectImpl)} or None, got: {impl}."
        raise TypeError(msg)

    def collect_schema(self) -> Schema: ...
    @property
    def columns(self) -> Sequence[str]: ...
    @property
    def native(self) -> Native: ...
    @property
    def version(self) -> Version: ...
