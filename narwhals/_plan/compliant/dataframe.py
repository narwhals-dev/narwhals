from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from narwhals._plan.compliant.group_by import Grouped
from narwhals._plan.compliant.typing import ColumnT_co, SeriesT, StoresVersion
from narwhals._plan.typing import (
    IntoExpr,
    NativeDataFrameT,
    NativeFrameT,
    NativeSeriesT,
    OneOrIterable,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from typing_extensions import Self

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
    from narwhals._plan.options import SortMultipleOptions
    from narwhals._plan.typing import Seq
    from narwhals._utils import Version
    from narwhals.dtypes import DType
    from narwhals.typing import IntoSchema


class CompliantFrame(StoresVersion, Protocol[ColumnT_co, NativeFrameT]):
    _native: NativeFrameT

    def __narwhals_namespace__(self) -> Any: ...
    @property
    def _group_by(self) -> type[CompliantGroupBy[Self]]: ...
    @property
    def native(self) -> NativeFrameT:
        return self._native

    @property
    def columns(self) -> list[str]: ...
    def to_narwhals(self) -> BaseFrame[NativeFrameT]: ...
    @classmethod
    def from_native(cls, native: NativeFrameT, /, version: Version) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        obj._version = version
        return obj

    def _with_native(self, native: NativeFrameT) -> Self:
        return self.from_native(native, self.version)

    @property
    def schema(self) -> Mapping[str, DType]: ...
    def _evaluate_irs(
        self, nodes: Iterable[NamedIR[ir.ExprIR]], /
    ) -> Iterator[ColumnT_co]: ...
    def select(self, irs: Seq[NamedIR]) -> Self: ...
    def select_names(self, *column_names: str) -> Self: ...
    def with_columns(self, irs: Seq[NamedIR]) -> Self: ...
    def sort(self, by: Seq[NamedIR], options: SortMultipleOptions) -> Self: ...
    def drop(self, columns: Sequence[str], *, strict: bool = True) -> Self: ...
    def drop_nulls(self, subset: Sequence[str] | None) -> Self: ...


class CompliantDataFrame(
    CompliantFrame[SeriesT, NativeDataFrameT],
    Protocol[SeriesT, NativeDataFrameT, NativeSeriesT],
):
    @property
    def _group_by(self) -> type[DataFrameGroupBy[Self]]: ...
    @property
    def _grouper(self) -> type[Grouped]:
        return Grouped

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], /, *, schema: IntoSchema | None = None
    ) -> Self: ...
    def group_by_agg(
        self, by: OneOrIterable[IntoExpr], aggs: OneOrIterable[IntoExpr], /
    ) -> Self:
        """Compliant-level `group_by(by).agg(agg)`, allows `Expr`."""
        return self._grouper.by(by).agg(aggs).resolve(self).evaluate(self)

    def group_by_names(self, names: Seq[str], /) -> DataFrameGroupBy[Self]:
        """Compliant-level `group_by`, allowing only `str` keys."""
        return self._group_by.by_names(self, names)

    def group_by_resolver(self, resolver: GroupByResolver, /) -> DataFrameGroupBy[Self]:
        """Narwhals-level resolved `group_by`.

        `keys`, `aggs` are already parsed and projections planned.
        """
        return self._group_by.from_resolver(self, resolver)

    def to_narwhals(self) -> DataFrame[NativeDataFrameT, NativeSeriesT]: ...
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
    def __len__(self) -> int: ...
    def with_row_index(self, name: str) -> Self: ...
    def row(self, index: int) -> tuple[Any, ...]: ...


class EagerDataFrame(
    CompliantDataFrame[SeriesT, NativeDataFrameT, NativeSeriesT],
    Protocol[SeriesT, NativeDataFrameT, NativeSeriesT],
):
    @property
    def _group_by(self) -> type[EagerDataFrameGroupBy[Self]]: ...
    def __narwhals_namespace__(self) -> EagerNamespace[Self, SeriesT, Any, Any]: ...
    def select(self, irs: Seq[NamedIR]) -> Self:
        return self.__narwhals_namespace__()._concat_horizontal(self._evaluate_irs(irs))

    def with_columns(self, irs: Seq[NamedIR]) -> Self:
        return self.__narwhals_namespace__()._concat_horizontal(self._evaluate_irs(irs))
