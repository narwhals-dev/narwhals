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
    from narwhals._plan.typing import Seq
    from narwhals.schema import Schema
    from narwhals.typing import EagerAllowed


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


class PolarsWhatever(ResolvedToCompliant[pl.LazyFrame]):
    """No idea what to call this yet.

    *Somewhat* of a mix between `*Namespace` and `*LazyFrame`
    """

    __slots__ = ("_version",)
    _version: Version

    @property
    def version(self) -> Version:
        return self._version

    def __init__(self, version: Version = Version.MAIN) -> None:
        self._version = version

    @classmethod
    def collect(
        cls,
        plan: rp.Collect,
        /,
        backend: EagerAllowed | None = None,
        version: Version = Version.MAIN,
    ) -> DataFrameAny:
        kwds = plan.kwds()
        return plan.input.evaluate(cls(version)).collect_compliant(
            backend or "polars", **kwds
        )

    @classmethod
    def sink_parquet(
        cls, plan: rp.SinkParquet, /, version: Version = Version.MAIN
    ) -> None:
        plan.input.evaluate(cls(version)).native.sink_parquet(plan.target)

    def _into_compliant(self, native: pl.LazyFrame, /) -> PolarsLazyFrame:
        return PolarsLazyFrame.from_native(native, self.version)

    def concat_horizontal(self, plan: rp.HConcat) -> PolarsLazyFrame:
        inputs = (input.evaluate(self).native for input in plan.inputs)
        return self._into_compliant(pl.concat(inputs, how="horizontal"))

    def concat_vertical(self, plan: rp.VConcat) -> PolarsLazyFrame:
        inputs = (input.evaluate(self).native for input in plan.inputs)
        if not plan.maintain_order:
            # TODO @dangotbanned: low priority, since we don't have `nw.union`
            msg = (
                f"TODO @dangotbanned: Add version branching for {plan.maintain_order=}\n"
                "Should dispatch to `pl.union` or raise if not available"
            )
            raise NotImplementedError(msg)
        return self._into_compliant(pl.concat(inputs, how="vertical"))

    def join(self, plan: rp.Join) -> PolarsLazyFrame:
        left, right = (input.evaluate(self).native for input in plan.inputs)
        return self._into_compliant(
            left.join(
                right,
                plan.options.how,
                left_on=plan.left_on,
                right_on=plan.right_on,
                suffix=plan.options.suffix,
            )
        )

    def join_asof(self, plan: rp.JoinAsof) -> PolarsLazyFrame:
        left, right = (input.evaluate(self).native for input in plan.inputs)
        by_left: Seq[str] | None = None
        by_right: Seq[str] | None = None
        if by := plan.options.by:
            by_left, by_right = by.left_by, by.right_by
        return self._into_compliant(
            left.join_asof(
                right,
                left_on=plan.left_on,
                right_on=plan.right_on,
                by_left=by_left,
                by_right=by_right,
                strategy=plan.options.strategy,
                suffix=plan.options.suffix,
            )
        )

    def scan_csv(self, plan: rp.ScanCsv) -> PolarsLazyFrame:
        return self._into_compliant(pl.scan_csv(plan.source))

    def scan_dataframe(self, plan: rp.ScanDataFrame, /) -> PolarsLazyFrame:
        return PolarsLazyFrame.from_narwhals(plan.frame)

    def scan_lazyframe(self, plan: rp.ScanLazyFrame[Any], /) -> PolarsLazyFrame:
        if plan.frame.implementation.is_polars() and isinstance(
            plan.frame, PolarsLazyFrame
        ):
            return plan.frame
        raise NotImplementedError(plan.frame.implementation, type(plan.frame))

    def scan_parquet(self, plan: rp.ScanParquet) -> PolarsLazyFrame:
        return self._into_compliant(pl.scan_parquet(plan.source))

    def select_names(self, plan: rp.SelectNames) -> PolarsLazyFrame:
        return self._into_compliant(plan.input.evaluate(self).native.select(plan.names))

    def slice(self, plan: rp.Slice) -> PolarsLazyFrame:
        # NOTE: This is just `head`/`tail`
        # `duckdb` & `ibis` support this as `limit`
        # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-lazy/src/frame/mod.rs#L1825-L1838
        return self._into_compliant(
            plan.input.evaluate(self).native.slice(plan.offset, plan.length)
        )

    def sort(self, plan: rp.Sort, /) -> PolarsLazyFrame:
        opts = plan.options
        return self._into_compliant(
            plan.input.evaluate(self).native.sort(
                plan.by, descending=opts.descending, nulls_last=opts.nulls_last
            )
        )

    def unique(self, plan: rp.Unique) -> PolarsLazyFrame:
        opts = plan.options
        return self._into_compliant(
            plan.input.evaluate(self).native.unique(
                plan.subset, keep=opts.keep, maintain_order=opts.maintain_order
            )
        )

    # TODO @dangotbanned: Add version branching for False in `Explode.options`
    # Default is backwards compatible
    def explode(self, plan: rp.MapFunction[rp.Explode]) -> PolarsLazyFrame:
        f = plan.function
        opts = f.options
        if not (opts.empty_as_null or opts.keep_nulls):
            msg = f"TODO @dangotbanned: Add version branching for False in `Explode.options`, got: {opts}"
            raise NotImplementedError(msg)
        return self._into_compliant(plan.input.evaluate(self).native.explode(f.columns))

    def unnest(self, plan: rp.MapFunction[rp.Unnest]) -> PolarsLazyFrame:
        f = plan.function
        return self._into_compliant(plan.input.evaluate(self).native.unnest(f.columns))

    def unpivot(self, plan: rp.MapFunction[rp.Unpivot]) -> PolarsLazyFrame:
        f = plan.function
        return self._into_compliant(
            plan.input.evaluate(self).native.unpivot(
                f.on,
                index=f.index,
                variable_name=f.options.variable_name,
                value_name=f.options.value_name,
            )
        )

    def with_row_index(self, plan: rp.MapFunction[rp.RowIndex]) -> PolarsLazyFrame:
        return self._into_compliant(
            plan.input.evaluate(self).native.with_row_index(plan.function.name)
        )

    filter = todo()
    group_by = todo()
    group_by_names = todo()
    select = todo()
    with_columns = todo()
