from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from narwhals._plan._version import into_version
from narwhals._plan.common import temp, todo
from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
from narwhals._plan.plans.visitors import ResolvedToCompliant
from narwhals._plan.polars.frame import PolarsFrame
from narwhals._plan.polars.namespace import PolarsNamespace as Namespace, explode_todo
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    from io import BytesIO
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._plan.compliant.typing import DataFrameAny
    from narwhals._plan.plans import resolved as rp
    from narwhals._plan.typing import Seq
    from narwhals.schema import Schema
    from narwhals.typing import EagerAllowed


class PolarsLazyFrame(PolarsFrame, CompliantLazyFrame[pl.LazyFrame]):
    __slots__ = ("_input_schema", "_native")
    _native: pl.LazyFrame
    _input_schema: Schema | None

    @classmethod
    def from_native(cls, native: pl.LazyFrame, /) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._input_schema = None
        return obj

    @classmethod
    def from_polars(cls, frame: pl.DataFrame, /) -> Self:
        return cls.from_native(frame.lazy())

    def collect_polars(self, **kwds: Any) -> pl.DataFrame:
        if "background" in kwds:
            msg = "TODO @dangotbanned: handle `LazyFrame.collect(background=True) -> InProcessQuery`"
            raise NotImplementedError(msg)
        return self.native.collect(**kwds, background=False)

    def sink_parquet(self, target: str | BytesIO | Path, /, **kwds: Any) -> None:
        self.native.sink_parquet(target, **kwds)

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

    collect_schema = todo()

    def collect_arrow(self, **kwds: Any) -> pa.Table:
        return self.collect_polars(**kwds).to_arrow()

    def collect_pandas(self, **kwds: Any) -> pd.DataFrame:
        return self.collect_polars(**kwds).to_pandas()


class PolarsEvaluator(ResolvedToCompliant[pl.LazyFrame]):
    """*Somewhat* of a mix between `*Namespace` and `*LazyFrame`."""

    __slots__ = ()
    version: ClassVar[Version] = Version.MAIN
    implementation: ClassVar = Implementation.POLARS

    def __narwhals_namespace__(self) -> Namespace:
        return Namespace()

    # NOTE: These get special treatment as they're hot paths
    _lazyframe: ClassVar = PolarsLazyFrame

    to_lazy = _lazyframe.from_native

    def _into_compliant(self, native: pl.LazyFrame, /) -> PolarsLazyFrame:
        return self._lazyframe.from_native(native)

    @classmethod
    def collect(
        cls, plan: rp.Collect, /, backend: EagerAllowed | None = None
    ) -> DataFrameAny:
        kwds = plan.kwds()
        return plan.input.evaluate(cls()).collect_compliant(backend or "polars", **kwds)

    @classmethod
    def sink_parquet(cls, plan: rp.SinkParquet, /) -> None:
        plan.input.evaluate(cls()).native.sink_parquet(plan.target)

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

    def rename(self, plan: rp.MapFunction[rp.Rename]) -> PolarsLazyFrame:
        return self._into_compliant(
            plan.input.evaluate(self).native.rename(plan.function.mapping)
        )

    def scan_csv(self, plan: rp.ScanCsv) -> PolarsLazyFrame:
        return self.__narwhals_namespace__().scan_csv(plan.source)

    def scan_dataframe(self, plan: rp.ScanDataFrame, /) -> PolarsLazyFrame:
        return self._lazyframe.from_narwhals(plan.frame)

    def scan_lazyframe(self, plan: rp.ScanLazyFrame[Any], /) -> PolarsLazyFrame:
        if plan.frame.implementation.is_polars() and isinstance(
            plan.frame, PolarsLazyFrame
        ):
            return plan.frame
        raise NotImplementedError(plan.frame.implementation, type(plan.frame))

    def scan_parquet(self, plan: rp.ScanParquet) -> PolarsLazyFrame:
        return self.__narwhals_namespace__().scan_parquet(plan.source)

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
        by = plan.by
        return self._into_compliant(
            plan.input.evaluate(self).native.sort(by, **plan.options.to_polars(by))
        )

    def unique(self, plan: rp.Unique) -> PolarsLazyFrame:
        opts = plan.options
        return self._into_compliant(
            plan.input.evaluate(self).native.unique(
                plan.subset, keep=opts.keep, maintain_order=opts.maintain_order
            )
        )

    # Adapted from (https://github.com/narwhals-dev/narwhals/blob/dd929a36839c4ab7c63a5e8e799f773d81e553a4/narwhals/_polars/dataframe.py#L189-L197)
    # TODO @dangotbanned: Check if the newer `pyarrow` version was simpler than this
    # 2x `sort` + `with_row_index` seems pretty expensive to add to `unique`
    def unique_by(self, plan: rp.UniqueBy) -> PolarsLazyFrame:
        native = plan.input.evaluate(self).native
        opts = plan.options
        if not opts.maintain_order:
            result = native.sort(plan.order_by).unique(plan.subset, keep=opts.keep)
            return self._into_compliant(result)
        names = plan.schema.names
        idx = temp.column_name(names)
        return self._into_compliant(
            native.with_row_index(idx)
            .sort(plan.order_by)
            .unique(plan.subset, keep=opts.keep)
            .sort(idx)
            .select(names)
        )

    def explode(self, plan: rp.MapFunction[rp.Explode]) -> PolarsLazyFrame:
        f = plan.function
        opts = f.options
        explode_todo(empty_as_null=opts.empty_as_null, keep_nulls=opts.keep_nulls)
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

    def with_row_index_by(self, plan: rp.MapFunction[rp.RowIndexBy]) -> PolarsLazyFrame:
        f = plan.function
        native = plan.input.evaluate(self).native
        int_range = pl.int_range(pl.len()).over(order_by=f.order_by).alias(f.name)
        return self._into_compliant(native.select(int_range, pl.all()))

    # TODO @dangotbanned: All require adding an `Expr` layer
    # Revisit after getting coverage for everything else
    filter = todo()
    group_by = todo()
    group_by_names = todo()
    select = todo()
    with_columns = todo()
