from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from narwhals._plan._version import into_version
from narwhals._plan.common import todo
from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
from narwhals._plan.plans.visitors import ResolvedToCompliant
from narwhals._plan.polars.frame import PolarsFrame
from narwhals._utils import Version

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.compliant.typing import DataFrameAny
    from narwhals._plan.plans import resolved as rp
    from narwhals._plan.polars.dataframe import PolarsDataFrame
    from narwhals.schema import Schema


MAIN = Version.MAIN


class PolarsLazyFrame(PolarsFrame, CompliantLazyFrame[pl.LazyFrame]):
    __slots__ = ("_input_schema", "_native", "_version")
    _native: pl.LazyFrame
    _version: Version
    _input_schema: Schema | None

    @classmethod
    def from_native(cls, native: pl.LazyFrame, /, version: Version = MAIN) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._version = version
        obj._input_schema = None
        return obj

    @classmethod
    def from_polars(cls, frame: pl.DataFrame, /, version: Version = MAIN) -> Self:
        return cls.from_native(frame.lazy(), version)

    def collect_polars(self, **kwds: Any) -> pl.DataFrame:
        if "background" in kwds:
            msg = "TODO @dangotbanned: handle `LazyFrame.collect(background=True) -> InProcessQuery`"
            raise NotImplementedError(msg)
        return self.native.collect(**kwds, background=False)

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
    collect_pandas = todo()


def wrap_df(df: pl.DataFrame, /, version: Version = MAIN) -> PolarsDataFrame:
    from narwhals._plan.polars.dataframe import PolarsDataFrame

    return PolarsDataFrame.from_native(df, version)


class PolarsWhatever(ResolvedToCompliant[pl.LazyFrame]):
    """No idea what to call this yet.

    *Somewhat* of a mix between `*Namespace` and `*LazyFrame`
    """

    def collect(self, plan: rp.Collect, /) -> DataFrameAny:
        return wrap_df(plan.input.evaluate(self).collect_polars(**plan.kwds()))

    def scan_csv(self, plan: rp.ScanCsv) -> PolarsLazyFrame:
        return PolarsLazyFrame.from_native(pl.scan_csv(plan.source))

    def scan_dataframe(self, plan: rp.ScanDataFrame, /) -> PolarsLazyFrame:
        return PolarsLazyFrame.from_narwhals(plan.frame)

    def scan_lazyframe(self, plan: rp.ScanLazyFrame[Any], /) -> PolarsLazyFrame:
        if plan.frame.implementation.is_polars() and isinstance(
            plan.frame, PolarsLazyFrame
        ):
            return plan.frame
        raise NotImplementedError(plan.frame.implementation, type(plan.frame))

    def scan_parquet(self, plan: rp.ScanParquet) -> PolarsLazyFrame:
        return PolarsLazyFrame.from_native(pl.scan_parquet(plan.source))

    def sink_parquet(self, plan: rp.SinkParquet) -> None:
        plan.input.evaluate(self).native.sink_parquet(plan.target)
