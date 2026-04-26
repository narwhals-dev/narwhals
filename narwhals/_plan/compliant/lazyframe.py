"""Minimal wrapper for a native lazy query."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from narwhals._plan._namespace import collect_implementation
from narwhals._plan.compliant.typing import Native_co
from narwhals._typing_compat import assert_never
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.dataframe import DataFrame as NwDataFrame
    from narwhals._plan.plans import logical as lp
    from narwhals._plan.typing import IncompleteVarianceLie
    from narwhals._translate import ArrowStreamExportable, IntoArrowTable
    from narwhals._typing import _LazyFrameCollectImpl
    from narwhals.schema import Schema
    from narwhals.typing import EagerAllowed, IntoBackend

    CollectMap: TypeAlias = Mapping[
        _LazyFrameCollectImpl, Callable[[], pa.Table | pd.DataFrame | pl.DataFrame]
    ]

Incomplete: TypeAlias = Any


class NarwhalsHash(Protocol):
    __slots__ = ()

    def __narwhals_hash_values__(self) -> Iterator[object]:
        """Yield one or more attributes to seed a hash.

        All backends *could* use a pseudo hash:

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


class CompliantLazyFrame(NarwhalsHash, Protocol[Native_co]):
    """Clean-slate rework of `CompliantFrame`-based design.

    `[Native_co]`.

    Focused on features for `LogicalPlan`:
    - storing a native object
    - performing import/export operations
    - exposing the schema
    """

    __slots__ = ()

    implementation: ClassVar[Implementation]
    version: ClassVar[Version]

    def __narwhals_hash_values__(self) -> Iterator[object]:
        yield self.version, self.implementation, id(self.native)

    @classmethod
    def from_native(
        cls, native: IncompleteVarianceLie, /
    ) -> CompliantLazyFrame[Native_co]: ...
    @classmethod
    def from_arrow(cls, frame: IntoArrowTable, /) -> Self: ...
    @classmethod
    def from_arrow_c_stream(
        cls,
        exportable: ArrowStreamExportable,
        /,
        *,
        requested_schema: object | None = None,
    ) -> Self:
        if requested_schema is not None:
            msg = f"{cls.__name__}.from_arrow_c_stream"
            raise NotImplementedError(msg)
        return cls.from_arrow(exportable)

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame, /) -> Self: ...
    @classmethod
    def from_polars(cls, frame: pl.DataFrame, /) -> Self: ...
    @classmethod
    def from_narwhals(cls, frame: NwDataFrame[Any, Any], /) -> Self: ...
    @classmethod
    def from_compliant(cls, frame: CompliantDataFrame[Any, Any], /) -> Self: ...
    def collect_arrow(self, **kwds: Any) -> pa.Table: ...
    def collect_pandas(self, **kwds: Any) -> pd.DataFrame: ...
    def collect_polars(self, **kwds: Any) -> pl.DataFrame: ...

    # TODO @dangotbanned: Review version/implementation entrypoint
    def collect_compliant(
        self, backend: IntoBackend[EagerAllowed], **kwds: Any
    ) -> CompliantDataFrame[Any, Any]:
        impl = collect_implementation(backend)
        if impl is Implementation.PYARROW:
            from narwhals._plan import arrow

            return arrow.DataFrame.from_native(self.collect_arrow(**kwds))
        if impl is Implementation.POLARS:
            from narwhals._plan import polars

            return polars.DataFrame.from_native(self.collect_polars(**kwds))
        if impl is Implementation.PANDAS:
            raise NotImplementedError(impl)
        assert_never(impl)

    def collect_narwhals(
        self, backend: IntoBackend[EagerAllowed], **kwds: Any
    ) -> NwDataFrame[Any, Any]:
        return self.collect_compliant(backend, **kwds).to_narwhals()

    @property
    def input_schema(self) -> Schema:
        """Schema at the time of construction."""
        ...

    @property
    def input_columns(self) -> Sequence[str]:
        """Column names at the time of construction."""
        return self.input_schema.names()

    @property
    def native(self) -> Native_co: ...

    def to_logical(self) -> lp.ScanLazyFrame[Native_co]:
        from narwhals._plan.plans import LogicalPlan

        return LogicalPlan.from_lf(self)

    # TODO @dangotbanned: Need to think more about these
    # - Separating them from the `input_*`s was a good first step
    # - The result should be `ResolvedPlan.schema`
    #   - But `CompliantLazyFrame` only refers to a `Scan*` step
    # - `Resolver.collect_schema(plan: LogicalPlan)` might be the better place?
    def collect_schema(self, *args: Incomplete, **kwds: Incomplete) -> Schema: ...
    def collect_columns(self, *args: Incomplete, **kwds: Incomplete) -> Sequence[str]:
        return self.collect_schema(*args, **kwds).names()

    def to_native(self) -> Native_co:
        return self.native
