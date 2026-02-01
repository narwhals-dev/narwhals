from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.exceptions import NarwhalsError
from tpch.constants import (
    DATABASE_TABLE_NAMES,
    LOGGER_NAME,
    QUERIES_PACKAGE,
    QUERY_IDS,
    SCALE_FACTOR_DEFAULT,
    _scale_factor_dir,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import polars as pl
    import pytest
    from typing_extensions import Self

    from narwhals.typing import FileSource
    from tpch.typing_ import (
        KnownImpl,
        Predicate,
        QueryID,
        QueryModule,
        TPCHBackend,
        XFailRaises,
    )


class Backend:
    name: TPCHBackend
    implementation: KnownImpl
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
    table_names: tuple[str, ...]
    scale_factor: float
    _into_xfails: tuple[tuple[Predicate, str, XFailRaises], ...]
    _into_skips: tuple[tuple[Predicate, str], ...]

    def __init__(self, query_id: QueryID, table_names: tuple[str, ...]) -> None:
        self.id = query_id
        self.table_names = table_names
        self._into_xfails = ()
        self._into_skips = ()
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

    def execute(self, backend: Backend, request: pytest.FixtureRequest) -> None:
        from polars.testing import assert_frame_equal

        self._apply_skips(backend)
        data = self.inputs(backend)
        query = self._import_module().query

        try:
            result = query(*data).lazy().collect("polars").to_polars()
        except NarwhalsError as exc:
            msg = f"Query [{self}-{backend}] ({self.scale_factor=}) failed with the following error in Narwhals:\n{exc}"
            raise RuntimeError(msg) from exc

        self._apply_xfails(backend, request)
        expected = self.expected()
        try:
            assert_frame_equal(expected, result, check_dtypes=False)
        except AssertionError as exc:
            msg = f"Query [{self}-{backend}] ({self.scale_factor=}) resulted in wrong answer:\n{exc}"
            raise AssertionError(msg) from exc

    def with_scale_factor(self, scale_factor: float, /) -> Query:
        self.scale_factor = scale_factor
        return self

    def with_skip(self, predicate: Predicate, reason: str) -> Query:
        self._into_skips = (*self._into_skips, (predicate, reason))
        return self

    def with_xfail(
        self, predicate: Predicate, reason: str, *, raises: XFailRaises = AssertionError
    ) -> Query:
        self._into_xfails = (*self._into_xfails, (predicate, reason, raises))
        return self

    def _apply_skips(self, backend: Backend) -> None:
        import pytest

        for predicate, reason in self._into_skips:
            if predicate(backend, self.scale_factor):
                pytest.skip(reason)

    def _apply_xfails(self, backend: Backend, request: pytest.FixtureRequest) -> None:
        import pytest

        for predicate, reason, raises in self._into_xfails:
            condition = predicate(backend, self.scale_factor)
            mark = pytest.mark.xfail(condition, reason=reason, raises=raises)
            request.applymarker(mark)

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
