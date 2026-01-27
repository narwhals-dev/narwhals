from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import pytest
from polars.testing import assert_frame_equal as pl_assert_frame_equal

import narwhals as nw
from narwhals.exceptions import NarwhalsError
from tpch.constants import DATA_DIR

if TYPE_CHECKING:
    from pathlib import Path

    from narwhals._typing import IntoBackendAny
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

    def __init__(
        self, name: TPCHBackend, into_backend: IntoBackendAny, /, **kwds: Any
    ) -> None:
        self.name = name
        impl = nw.Implementation.from_backend(into_backend)
        assert impl is not nw.Implementation.UNKNOWN  # noqa: S101
        self.implementation = impl
        self.kwds = kwds

    def __repr__(self) -> str:
        return self.name

    def scan(self, source: FileSource) -> nw.LazyFrame[Any]:
        return nw.scan_parquet(source, backend=self.implementation, **self.kwds)


class Query:
    id: QueryID
    paths: tuple[Path, ...]
    _into_xfails: tuple[tuple[Predicate, str, XFailRaises], ...]
    _into_skips: tuple[tuple[Predicate, str], ...]

    PACKAGE_PREFIX: ClassVar = "tpch.queries"

    def __init__(self, query_id: QueryID, paths: tuple[Path, ...]) -> None:
        self.id = query_id
        self.paths = paths
        self._into_xfails = ()
        self._into_skips = ()

    def __repr__(self) -> str:
        return self.id

    def _import_module(self) -> QueryModule:
        result: Any = import_module(f"{self.PACKAGE_PREFIX}.{self}")
        return result

    def expected(self) -> pl.DataFrame:
        return pl.read_parquet(DATA_DIR / f"result_{self}.parquet")

    def run(self, backend: Backend) -> pl.DataFrame:
        data = tuple(backend.scan(fp.as_posix()) for fp in self.paths)
        return self._import_module().query(*data).lazy().collect("polars").to_polars()

    def try_run(self, backend: Backend, scale_factor: float) -> pl.DataFrame:
        self._apply_skips(backend, scale_factor)
        try:
            result = self.run(backend)
        except NarwhalsError as exc:
            msg = f"Query [{self}-{backend}] ({scale_factor=}) failed with the following error in Narwhals:\n{exc}"
            raise RuntimeError(msg) from exc
        return result

    def with_skip(self, predicate: Predicate, reason: str) -> Query:
        self._into_skips = (*self._into_skips, (predicate, reason))
        return self

    def with_xfail(
        self, predicate: Predicate, reason: str, *, raises: XFailRaises = AssertionError
    ) -> Query:
        self._into_xfails = (*self._into_xfails, (predicate, reason, raises))
        return self

    def _apply_skips(self, backend: Backend, scale_factor: float) -> None:
        for predicate, reason in self._into_skips:
            if predicate(backend, scale_factor):
                pytest.skip(reason)

    def _apply_xfails(
        self, backend: Backend, scale_factor: float, request: pytest.FixtureRequest
    ) -> None:
        for predicate, reason, raises in self._into_xfails:
            condition = predicate(backend, scale_factor)
            request.applymarker(
                pytest.mark.xfail(condition, reason=reason, raises=raises)
            )

    def assert_expected(
        self,
        result: pl.DataFrame,
        backend: Backend,
        scale_factor: float,
        request: pytest.FixtureRequest,
    ) -> None:
        self._apply_xfails(backend, scale_factor, request)
        try:
            pl_assert_frame_equal(self.expected(), result, check_dtypes=False)
        except AssertionError as exc:
            msg = f"Query [{self}-{backend}] ({scale_factor=}) resulted in wrong answer:\n{exc}"
            raise AssertionError(msg) from exc
