from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from narwhals._plan import plugins
from narwhals._utils import Implementation, Version, normalize_path, unstable

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.lazyframe import LazyFrame
    from narwhals._typing import Arrow, Polars
    from narwhals.schema import Schema
    from narwhals.typing import Backend, EagerAllowed, FileSource, IntoBackend

_MAIN = Version.MAIN


@overload
def read_csv(
    source: FileSource, *, backend: Arrow, **kwds: Any
) -> DataFrame[pa.Table, pa.ChunkedArray[Any]]: ...
@overload
def read_csv(
    source: FileSource, *, backend: Polars, **kwds: Any
) -> DataFrame[pl.DataFrame, pl.Series]: ...
@overload
def read_csv(
    source: FileSource, *, backend: IntoBackend[EagerAllowed], **kwds: Any
) -> DataFrame: ...
def read_csv(
    source: FileSource, *, backend: IntoBackend[EagerAllowed], **kwds: Any
) -> DataFrame[Any, Any]:
    source = normalize_path(source)
    manager = plugins.manager()
    return manager.dataframe(backend, _MAIN).read_csv(source, **kwds).to_narwhals()


@overload
def read_parquet(
    source: FileSource, *, backend: Arrow, **kwds: Any
) -> DataFrame[pa.Table, pa.ChunkedArray[Any]]: ...
@overload
def read_parquet(
    source: FileSource, *, backend: Polars, **kwds: Any
) -> DataFrame[pl.DataFrame, pl.Series]: ...
@overload
def read_parquet(
    source: FileSource, *, backend: IntoBackend[EagerAllowed], **kwds: Any
) -> DataFrame: ...
def read_parquet(
    source: FileSource, *, backend: IntoBackend[EagerAllowed], **kwds: Any
) -> DataFrame[Any, Any]:
    source = normalize_path(source)
    manager = plugins.manager()
    return manager.dataframe(backend, _MAIN).read_parquet(source, **kwds).to_narwhals()


def scan_csv(
    source: FileSource, *, backend: IntoBackend[Backend], **kwds: Any
) -> LazyFrame[Any]:
    return _scan_file(source, backend, kwds, "scan_csv", _MAIN)


def scan_parquet(
    source: FileSource, *, backend: IntoBackend[Backend], **kwds: Any
) -> LazyFrame[Any]:
    return _scan_file(source, backend, kwds, "scan_parquet", _MAIN)


@unstable
def read_csv_schema(
    source: FileSource, *, backend: IntoBackend[Backend], **kwds: Any
) -> Schema:
    """Infer the schema of a Csv file."""
    return _read_schema(source, backend, kwds, "csv")


def read_parquet_schema(
    source: FileSource, *, backend: IntoBackend[Backend], **kwds: Any
) -> Schema:
    """Get the schema of a Parquet file without reading data."""
    return _read_schema(source, backend, kwds, "parquet")


# TODO @dangotbanned: Avoid `read_{csv,parquet}_schema` depending on `CompliantNamespace`
def _read_schema(
    source: FileSource,
    backend: IntoBackend[Backend],
    kwds: dict[str, Any],
    extension: Literal["csv", "parquet"],
) -> Schema:
    impl = Implementation.from_backend(backend)
    if impl is Implementation.POLARS:
        from narwhals._plan import polars as _polars

        ns = _polars.Namespace()
    elif impl is Implementation.PYARROW:
        from narwhals._plan import arrow as _arrow

        ns = _arrow.Namespace()
    else:
        msg = f"Not yet supported in `narwhals._plan`, got: {impl!r}"
        raise NotImplementedError(msg)
    method = ns.read_parquet_schema if extension == "parquet" else ns.read_csv_schema
    return method(normalize_path(source), **kwds)


# TODO @dangotbanned: Coordinate overloads with `ScanFile.to_narwhals`?
def _scan_file(
    source: FileSource,
    backend: IntoBackend[Backend],
    kwds: dict[str, Any],
    method: Literal["scan_csv", "scan_parquet"],
    version: Version,
) -> LazyFrame[Any]:
    from narwhals._plan.plans import LogicalPlan

    if kwds:
        msg = f"Passing arbitrary keywords arguments to `{method}()` is not yet implemented, got: {kwds!r}"
        raise NotImplementedError(msg)
    plan = LogicalPlan.scan_csv if method == "scan_csv" else LogicalPlan.scan_parquet
    return plan(source).to_narwhals(backend, version)
