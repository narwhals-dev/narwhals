from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, overload

from narwhals._plan.compliant import io
from narwhals._plan.compliant.group_by import Grouped
from narwhals._plan.compliant.typing import (
    ColumnT_co,
    DataFrameAny,
    HasVersion,
    LazyFrameAny,
    SeriesT,
)
from narwhals._plan.typing import (
    IncompleteCyclic,
    IntoExpr,
    NativeDataFrameT,
    NativeFrameT_co,
    NativeLazyFrameT,
    NativeSeriesT,
    NonCrossJoinStrategy,
    OneOrIterable,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from io import BytesIO

    import polars as pl
    from typing_extensions import Self, TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.compliant.group_by import (
        CompliantGroupBy,
        DataFrameGroupBy,
        EagerDataFrameGroupBy,
        GroupByResolver,
    )
    from narwhals._plan.compliant.namespace import EagerNamespace
    from narwhals._plan.dataframe import BaseFrame, DataFrame
    from narwhals._plan.expressions import NamedIR
    from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
    from narwhals._plan.typing import Seq
    from narwhals._typing import _EagerAllowedImpl, _LazyAllowedImpl
    from narwhals._utils import Implementation, Version
    from narwhals.dtypes import DType
    from narwhals.typing import IntoSchema, PivotAgg, UniqueKeepStrategy

Incomplete: TypeAlias = Any


class CompliantFrame(HasVersion, Protocol[ColumnT_co, NativeFrameT_co]):
    implementation: ClassVar[Implementation]

    def __narwhals_namespace__(self) -> IncompleteCyclic: ...
    def _evaluate_irs(
        self, nodes: Iterable[NamedIR[ir.ExprIR]], /
    ) -> Iterator[ColumnT_co]: ...
    @property
    def _group_by(self) -> type[CompliantGroupBy[Self]]: ...
    def _with_native(self, native: Incomplete) -> Self: ...
    @classmethod
    def from_native(cls, native: Incomplete, /, version: Version) -> Self: ...
    @property
    def native(self) -> NativeFrameT_co: ...
    def to_narwhals(self) -> BaseFrame[NativeFrameT_co]: ...
    @property
    def columns(self) -> list[str]: ...
    def drop(self, columns: Sequence[str]) -> Self: ...
    def drop_nulls(self, subset: Sequence[str] | None) -> Self: ...
    def explode(self, subset: Sequence[str], options: ExplodeOptions) -> Self: ...
    # Shouldn't *need* to be `NamedIR`, but current impl depends on a name being passed around
    def filter(self, predicate: NamedIR, /) -> Self: ...
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
    def with_columns(self, irs: Seq[NamedIR]) -> Self: ...
    def with_row_index_by(
        self, name: str, order_by: Sequence[str], *, nulls_last: bool = False
    ) -> Self: ...


class CompliantLazyFrame(
    io.LazyOutput,
    CompliantFrame[ColumnT_co, NativeLazyFrameT],
    Protocol[ColumnT_co, NativeLazyFrameT],
):
    """Very incomplete!

    Using mostly as a placeholder for typing lazy I/O.
    """

    _native: NativeLazyFrameT

    def __narwhals_lazyframe__(self) -> Self:
        return self

    def _with_native(self, native: NativeLazyFrameT) -> Self:
        return self.from_native(native, self.version)

    @classmethod
    def from_native(cls, native: NativeLazyFrameT, /, version: Version) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._version = version
        return obj

    def to_narwhals(self) -> Incomplete:
        msg = f"{type(self).__name__}.to_narwhals"
        raise NotImplementedError(msg)

    @property
    def native(self) -> NativeLazyFrameT:
        return self._native

    def collect(self, backend: _EagerAllowedImpl | None, **kwds: Any) -> DataFrameAny: ...


class CompliantDataFrame(
    io.EagerOutput,
    CompliantFrame[SeriesT, NativeDataFrameT],
    Protocol[SeriesT, NativeDataFrameT, NativeSeriesT],
):
    implementation: ClassVar[_EagerAllowedImpl]
    _native: NativeDataFrameT

    def __narwhals_dataframe__(self) -> Self:
        return self

    def lazy(self, backend: _LazyAllowedImpl | None, **kwds: Any) -> LazyFrameAny: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    def __len__(self) -> int: ...
    @property
    def _group_by(self) -> type[DataFrameGroupBy[Self]]: ...
    @property
    def _grouper(self) -> type[Grouped]:
        return Grouped

    def _with_native(self, native: NativeDataFrameT) -> Self:
        return self.from_native(native, self.version)

    @classmethod
    def from_native(cls, native: NativeDataFrameT, /, version: Version) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._version = version
        return obj

    @property
    def native(self) -> NativeDataFrameT:
        return self._native

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], /, *, schema: IntoSchema | None = None
    ) -> Self: ...
    def gather_every(self, n: int, offset: int = 0) -> Self: ...
    def get_column(self, name: str) -> SeriesT: ...
    def group_by_agg(
        self, by: OneOrIterable[IntoExpr], aggs: OneOrIterable[IntoExpr], /
    ) -> Self:
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
        return self._group_by.by_names(self, names)

    def group_by_resolver(self, resolver: GroupByResolver, /) -> DataFrameGroupBy[Self]:
        """Narwhals-level resolved `group_by`.

        `keys`, `aggs` are already parsed and projections planned.
        """
        return self._group_by.from_resolver(self, resolver)

    def filter(self, predicate: NamedIR, /) -> Self: ...
    def iter_columns(self) -> Iterator[SeriesT]: ...
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
    def partition_by(
        self, by: Sequence[str], *, include_key: bool = True
    ) -> list[Self]: ...
    def pivot(
        self,
        on: Sequence[str],
        on_columns: Sequence[str] | Self,
        *,
        index: Sequence[str],
        values: Sequence[str],
        separator: str = "_",
    ) -> Self: ...
    def pivot_agg(
        self,
        on: Sequence[str],
        on_columns: Sequence[str] | Self,
        *,
        index: Sequence[str],
        values: Sequence[str],
        aggregate_function: PivotAgg,  # not sure if possible for pyarrow yet
        separator: str = "_",
    ) -> Self: ...
    def row(self, index: int) -> tuple[Any, ...]: ...
    @overload
    def to_dict(self, *, as_series: Literal[True]) -> dict[str, SeriesT]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, SeriesT] | dict[str, list[Any]]: ...
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, SeriesT] | dict[str, list[Any]]: ...
    def to_narwhals(self) -> DataFrame[NativeDataFrameT, NativeSeriesT]:
        from narwhals._plan.dataframe import DataFrame

        return DataFrame[NativeDataFrameT, NativeSeriesT](self)

    def to_series(self, index: int = 0) -> SeriesT: ...
    def to_struct(self, name: str = "") -> SeriesT: ...
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
    def sample_frac(
        self, fraction: float, *, with_replacement: bool = False, seed: int | None = None
    ) -> Self:
        n = int(len(self) * fraction)
        return self.sample_n(n, with_replacement=with_replacement, seed=seed)

    def sample_n(
        self, n: int = 1, *, with_replacement: bool = False, seed: int | None = None
    ) -> Self: ...


class EagerDataFrame(
    io.LazyOutput,
    CompliantDataFrame[SeriesT, NativeDataFrameT, NativeSeriesT],
    Protocol[SeriesT, NativeDataFrameT, NativeSeriesT],
):
    def __narwhals_namespace__(self) -> EagerNamespace[Self, SeriesT, Any, Any]: ...
    @property
    def _group_by(self) -> type[EagerDataFrameGroupBy[Self]]: ...
    def _evaluate_irs(
        self, nodes: Iterable[NamedIR[ir.ExprIR]], /, *, length: int | None = None
    ) -> Iterator[SeriesT]: ...

    def group_by_resolver(
        self, resolver: GroupByResolver, /
    ) -> EagerDataFrameGroupBy[Self]:
        return self._group_by.from_resolver(self, resolver)

    def select(self, irs: Seq[NamedIR]) -> Self:
        return self.__narwhals_namespace__()._concat_horizontal(self._evaluate_irs(irs))

    def with_columns(self, irs: Seq[NamedIR]) -> Self:
        return self.__narwhals_namespace__()._concat_horizontal(
            self._evaluate_irs(irs, length=len(self))
        )

    def to_series(self, index: int = 0) -> SeriesT:
        return self.get_column(self.columns[index])

    def sink_parquet(self, target: str | BytesIO, /) -> None:
        self.write_parquet(target)
