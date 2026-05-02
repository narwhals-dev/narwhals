from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, overload

from narwhals._plan._version import into_version
from narwhals._plan.compliant import io
from narwhals._plan.compliant.group_by import Grouped
from narwhals._plan.typing import (
    IncompleteCyclic,
    IncompleteVarianceLie,
    IntoExpr,
    NativeDataFrameT_co,
    NativeFrameT_co,
    NativeSeriesT_co,
    NonCrossJoinStrategy,
    OneOrIterable,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from io import BytesIO

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._plan import expressions as ir
    from narwhals._plan.compliant.group_by import (
        CompliantGroupBy,
        DataFrameGroupBy,
        EagerDataFrameGroupBy,
        GroupByResolver,
    )
    from narwhals._plan.compliant.namespace import CompliantNamespace
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.compliant.typing import LazyFrameAny
    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.expressions import NamedIR
    from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
    from narwhals._plan.typing import Seq
    from narwhals._translate import ArrowStreamExportable, IntoArrowTable
    from narwhals._typing import _LazyAllowedImpl
    from narwhals._utils import Implementation, Version
    from narwhals.dtypes import DType
    from narwhals.typing import AsofJoinStrategy, IntoSchema, PivotAgg, UniqueKeepStrategy


class CompliantFrame(Protocol[NativeFrameT_co]):
    """`[NativeFrameT_co]`."""

    implementation: ClassVar[Implementation]
    version: ClassVar[Version]

    def __narwhals_namespace__(self) -> IncompleteCyclic: ...
    @property
    def _group_by(self) -> type[CompliantGroupBy[Self]]: ...
    @classmethod
    def from_native(cls, native: IncompleteVarianceLie, /) -> Self: ...
    @property
    def native(self) -> NativeFrameT_co: ...
    @property
    def columns(self) -> list[str]: ...
    def drop(self, columns: Sequence[str]) -> Self: ...
    def drop_nulls(self, subset: Sequence[str] | None) -> Self: ...
    def explode(self, columns: Sequence[str], options: ExplodeOptions) -> Self: ...
    # Shouldn't *need* to be `NamedIR`, but current impl depends on a name being passed around
    def filter(self, predicate: NamedIR, /) -> CompliantFrame[NativeFrameT_co]: ...
    def join(
        self,
        other: Self,
        *,
        how: NonCrossJoinStrategy,
        left_on: Sequence[str],
        right_on: Sequence[str],
        suffix: str = "_right",
    ) -> Self: ...
    def join_cross(self, other: Self, *, suffix: str = "_right") -> Self: ...
    def join_asof(
        self,
        other: Self,
        *,
        left_on: str,
        right_on: str,
        left_by: Sequence[str] = (),  # https://github.com/pola-rs/polars/issues/18496
        right_by: Sequence[str] = (),
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
    ) -> Self: ...
    def rename(self, mapping: Mapping[str, str]) -> Self: ...
    @property
    def schema(self) -> Mapping[str, DType]: ...
    def select(self, irs: Seq[NamedIR]) -> Self: ...
    def select_names(self, *column_names: str) -> Self: ...
    def sort(self, by: Sequence[str], options: SortMultipleOptions) -> Self: ...
    def unique(
        self, subset: Sequence[str] | None = None, *, keep: UniqueKeepStrategy = "any"
    ) -> Self: ...
    def unique_by(
        self,
        subset: Sequence[str] | None = None,
        *,
        order_by: Sequence[str],
        keep: UniqueKeepStrategy = "any",
    ) -> Self: ...
    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        *,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> Self: ...
    def unnest(self, columns: Sequence[str]) -> Self: ...
    def with_columns(self, irs: Seq[NamedIR]) -> Self: ...
    def with_row_index_by(
        self, name: str, order_by: Sequence[str], *, nulls_last: bool = False
    ) -> Self: ...


class CompliantDataFrame(
    io.EagerOutput,
    CompliantFrame[NativeDataFrameT_co],
    Protocol[NativeDataFrameT_co, NativeSeriesT_co],
):
    """`[NativeDataFrameT_co, NativeSeriesT_co]`."""

    def __narwhals_dataframe__(self) -> Self:  # pragma: no cover
        return self

    def __narwhals_namespace__(
        self,
    ) -> CompliantNamespace[IncompleteVarianceLie, Any, Any]: ...

    @property
    def _grouper(self) -> type[Grouped]:
        return Grouped

    @property
    def width(self) -> int:
        return self.shape[-1]

    @classmethod
    def from_arrow_c_stream(
        cls,
        exportable: ArrowStreamExportable,
        /,
        *,
        requested_schema: object | None = None,
    ) -> Self:  # pragma: no cover
        if requested_schema is not None:
            msg = f"{cls.__name__}.from_arrow_c_stream"
            raise NotImplementedError(msg)
        return cls.from_arrow(exportable)

    def group_by_agg(
        self, by: OneOrIterable[IntoExpr], aggs: OneOrIterable[IntoExpr], /
    ) -> Self:  # pragma: no cover
        """Compliant-level `group_by(by).agg(agg)`, allows `Expr`."""
        return self._grouper.by(by).agg(aggs).resolve(self).evaluate(self)

    def group_by_agg_irs(
        self, by: OneOrIterable[ir.ExprIR], aggs: OneOrIterable[ir.ExprIR], /
    ) -> Self:
        """Compliant-level `group_by(by).agg(agg)`, allows `ExprIR`.

        Useful for rewriting `over(*partition_by)`.
        """
        by = (by,) if not isinstance(by, Iterable) else by
        aggs = (aggs,) if not isinstance(aggs, Iterable) else aggs
        return self._grouper.by_irs(*by).agg_irs(*aggs).resolve(self).evaluate(self)

    def group_by_names(self, names: Seq[str], /) -> DataFrameGroupBy[Self]:
        """Compliant-level `group_by`, allowing only `str` keys."""
        return self._group_by.by_names(self, names)  # type: ignore[return-value]

    def group_by_resolver(
        self, resolver: GroupByResolver, /
    ) -> DataFrameGroupBy[Self]:  # pragma: no cover
        """Narwhals-level resolved `group_by`.

        `keys`, `aggs` are already parsed and projections planned.
        """
        return self._group_by.from_resolver(self, resolver)  # type: ignore[return-value]

    @overload
    def to_dict(
        self, *, as_series: Literal[True]
    ) -> Mapping[str, CompliantSeries[NativeSeriesT_co]]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(
        self, *, as_series: bool
    ) -> Mapping[str, CompliantSeries[NativeSeriesT_co]] | dict[str, list[Any]]: ...
    def to_dict(
        self, *, as_series: bool
    ) -> Mapping[str, CompliantSeries[NativeSeriesT_co]] | dict[str, list[Any]]:
        it = self.iter_columns()
        if as_series:
            return {ser.name: ser for ser in it}  # pragma: no cover
        return {ser.name: ser.to_list() for ser in it}

    def to_narwhals(self) -> DataFrame[NativeDataFrameT_co, NativeSeriesT_co]:
        return into_version(self.version).dataframe(self)

    def sample_frac(
        self, fraction: float, *, with_replacement: bool = False, seed: int | None = None
    ) -> Self:
        n = int(len(self) * fraction)
        return self.sample_n(n, with_replacement=with_replacement, seed=seed)

    def lazy(self, backend: _LazyAllowedImpl | None, **kwds: Any) -> LazyFrameAny: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    def __len__(self) -> int: ...

    # NOTE: `pyright` includes `Self` in `_group_by` to calculate variance (`mypy` doesn't)
    @property
    def _group_by(self) -> type[DataFrameGroupBy[Self]]: ...
    @classmethod
    def from_arrow(cls, frame: IntoArrowTable, /) -> Self: ...
    @classmethod
    def from_pandas(cls, frame: pd.DataFrame, /) -> Self: ...
    @classmethod
    def from_polars(cls, frame: pl.DataFrame, /) -> Self: ...
    @classmethod
    def from_narwhals(cls, frame: DataFrame[Any, Any], /) -> Self: ...
    @classmethod
    def from_compliant(cls, frame: CompliantDataFrame[Any, Any], /) -> Self: ...
    def clone(self) -> Self: ...
    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], /, *, schema: IntoSchema | None = None
    ) -> Self: ...
    def gather_every(self, n: int, offset: int = 0) -> Self: ...
    def get_column(self, name: str) -> CompliantSeries[NativeSeriesT_co]: ...
    def filter(self, predicate: NamedIR, /) -> Self: ...
    def iter_columns(self) -> Iterator[CompliantSeries[NativeSeriesT_co]]: ...
    def partition_by(
        self, by: Sequence[str], *, include_key: bool = True
    ) -> list[Self]: ...
    def pivot(
        self,
        on: Sequence[str],
        on_columns: Self,
        *,
        index: Sequence[str],
        values: Sequence[str],
        aggregate_function: PivotAgg | None = None,
        separator: str = "_",
        sort_columns: bool = False,
    ) -> Self:
        """Create a spreadsheet-style pivot table as a DataFrame.

        Note:
            `sort_columns` is passed down for backwards compatibility with [`polars<1.36.0`],
            where [`sort_columns` moved] from being handled in rust to python.
            All other backends **should ignore `sort_columns`**, as the narwhals-level
            sorts *into* `on_columns` when needed.

        [`polars<1.36.0`]: https://github.com/pola-rs/polars/pull/25016
        [`sort_columns` moved]: https://github.com/pola-rs/polars/pull/25016/changes#diff-d2e79c12d0d5ed35f2015c678f5be62199581d902d25069b9635817c673ca6ebR9450-R9459
        """
        ...

    def row(self, index: int) -> tuple[Any, ...]: ...
    def to_series(self, index: int = 0) -> CompliantSeries[NativeSeriesT_co]: ...
    def to_struct(self, name: str = "") -> CompliantSeries[NativeSeriesT_co]: ...
    def to_arrow(self) -> pa.Table: ...
    def to_pandas(self) -> pd.DataFrame: ...
    def to_polars(self) -> pl.DataFrame: ...
    def unique(
        self,
        subset: Sequence[str] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
    ) -> Self: ...
    def unique_by(
        self,
        subset: Sequence[str] | None = None,
        *,
        order_by: Sequence[str],
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
    ) -> Self: ...
    def with_row_index(self, name: str) -> Self: ...
    def slice(self, offset: int, length: int | None = None) -> Self: ...
    def sample_n(
        self, n: int = 1, *, with_replacement: bool = False, seed: int | None = None
    ) -> Self: ...


class EagerDataFrame(
    io.LazyOutput,
    CompliantDataFrame[NativeDataFrameT_co, NativeSeriesT_co],
    Protocol[NativeDataFrameT_co, NativeSeriesT_co],
):
    """`[NativeDataFrameT_co, NativeSeriesT_co]`."""

    @property
    def _group_by(self) -> type[EagerDataFrameGroupBy[Self]]: ...
    def _evaluate_irs(
        self, nodes: Iterable[NamedIR], /, *, length: int | None = None
    ) -> Iterator[CompliantSeries[NativeSeriesT_co]]: ...

    def group_by_resolver(
        self, resolver: GroupByResolver, /
    ) -> EagerDataFrameGroupBy[Self]:  # pragma: no cover
        return self._group_by.from_resolver(self, resolver)  # type: ignore[return-value]

    def sink_parquet(self, target: str | BytesIO, /) -> None:  # pragma: no cover
        self.write_parquet(target)
