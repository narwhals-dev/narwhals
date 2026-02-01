from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal

import narwhals as nw
from narwhals.exceptions import NarwhalsError
from tpch.constants import (
    DATABASE_TABLE_NAMES,
    LOGGER_NAME,
    QUERIES_PACKAGE,
    QUERY_IDS,
    SCALE_FACTOR_DEFAULT,
    DBTableName,
    _scale_factor_dir,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import polars as pl
    from typing_extensions import Self

    from narwhals._typing import _EagerAllowedImpl, _LazyAllowedImpl
    from narwhals.typing import FileSource
    from tpch.typing_ import QueryID, QueryModule, ScaleFactor, TPCHBackend


class Backend:
    name: TPCHBackend
    implementation: Literal[_EagerAllowedImpl, _LazyAllowedImpl]
    kwds: dict[str, Any]

    def __init__(self, name: TPCHBackend, /, **kwds: Any) -> None:
        backend = name.partition("[")[0]
        impl = nw.Implementation.from_backend(backend)
        assert impl is not nw.Implementation.UNKNOWN, f"{name!r} -> {backend!r}"  # noqa: S101
        self.name = name
        self.implementation = impl
        self.kwds = kwds

    def __repr__(self) -> str:
        return self.name

    def scan(self, source: FileSource) -> nw.LazyFrame[Any]:
        return nw.scan_parquet(source, backend=self.implementation, **self.kwds)


class Query:
    id: QueryID
    table_names: tuple[DBTableName, ...]
    scale_factor: ScaleFactor

    def __init__(self, query_id: QueryID, table_names: tuple[DBTableName, ...]) -> None:
        self.id = query_id
        self.table_names = table_names
        self.scale_factor = SCALE_FACTOR_DEFAULT

    def __repr__(self) -> str:
        return self.id

    def inputs(self, backend: Backend) -> tuple[nw.LazyFrame[Any], ...]:
        """Get the frame inputs for this query at the given scale factor."""
        sf_dir = _scale_factor_dir(self.scale_factor)
        return tuple(
            backend.scan((sf_dir / f"{name}.parquet").as_posix())
            for name in self.table_names
        )

    def expected(self) -> pl.DataFrame:
        import polars as pl

        sf_dir = _scale_factor_dir(self.scale_factor)
        return pl.read_parquet(sf_dir / f"result_{self}.parquet")

    def execute(self, backend: Backend) -> None:
        from polars.testing import assert_frame_equal

        data = self.inputs(backend)
        query = self._import_module().query

        try:
            result = query(*data).lazy().collect("polars").to_polars()
        except NarwhalsError as exc:
            msg = f"Query [{self}-{backend}] ({self.scale_factor=}) failed with the following error in Narwhals:\n{exc}"
            raise RuntimeError(msg) from exc
        expected = self.expected()
        try:
            assert_frame_equal(expected, result, check_dtypes=False)
        except AssertionError as exc:
            msg = f"Query [{self}-{backend}] ({self.scale_factor=}) resulted in wrong answer:\n{exc}"
            raise AssertionError(msg) from exc

    def with_scale_factor(self, scale_factor: ScaleFactor, /) -> Query:
        self.scale_factor = scale_factor
        return self

    def _import_module(self) -> QueryModule:
        result: Any = import_module(f"{QUERIES_PACKAGE}.{self}")
        return result


logger = logging.getLogger(LOGGER_NAME)


class TableLogger:
    """A logger that streams table rows with box-drawing characters."""

    # Size column: 4 leading digits + 1 dot + 2 decimals + 1 space + 2 unit chars = 10 chars
    SIZE_WIDTH = 10

    def __init__(self, file_names: Iterable[str]) -> None:
        self._file_width = max(len(name) for name in file_names)

    @staticmethod
    def answers() -> TableLogger:
        return TableLogger(f"result_{qid}.parquet" for qid in QUERY_IDS)

    @staticmethod
    def database() -> TableLogger:
        return TableLogger(f"{t}.parquet" for t in DATABASE_TABLE_NAMES)

    def __enter__(self) -> Self:
        # header
        fw, sw = self._file_width, self.SIZE_WIDTH
        logger.info("┌─%s─┬─%s─┐", "─" * fw, "─" * sw)
        logger.info("│ %s ┆ %s │", "File".rjust(fw), "Size".rjust(sw))
        logger.info("╞═%s═╪═%s═╡", "═" * fw, "═" * sw)
        return self

    def log_row(self, path: Path) -> None:
        size = self.format_size(path.stat().st_size)
        logger.info("│ %s ┆ %s │", path.name.rjust(self._file_width), size)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        # footer
        fw, sw = self._file_width, self.SIZE_WIDTH
        logger.info("└─%s─┴─%s─┘", "─" * fw, "─" * sw)

    @staticmethod
    def format_size(n_bytes: float) -> str:
        """Return the best human-readable size and unit for the given byte count."""
        size = float(n_bytes)
        units = iter(("b", "kb", "mb", "gb", "tb"))
        unit = next(units)
        while size >= 1024 and unit != "tb":
            size /= 1024
            unit = next(units, "tb")
        return f"{size:>7.2f} {unit:>2}"
