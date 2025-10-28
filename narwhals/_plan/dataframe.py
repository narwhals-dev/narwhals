from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, get_args, overload

from narwhals._plan import _parse
from narwhals._plan._expansion import (
    expand_selector_irs_names,
    prepare_projection,
    prepare_projection_s,
)
from narwhals._plan.common import ensure_seq_str, temp
from narwhals._plan.exceptions import group_by_no_keys_error
from narwhals._plan.group_by import GroupBy, Grouped
from narwhals._plan.options import SortMultipleOptions
from narwhals._plan.series import Series
from narwhals._plan.typing import (
    ColumnNameOrSelector,
    IntoExpr,
    IntoExprColumn,
    NativeDataFrameT,
    NativeDataFrameT_co,
    NativeFrameT_co,
    NativeSeriesT,
    NonCrossJoinStrategy,
    OneOrIterable,
    PartialSeries,
    Seq,
)
from narwhals._utils import Implementation, Version, generate_repr
from narwhals.dependencies import is_pyarrow_table
from narwhals.schema import Schema
from narwhals.typing import IntoDType, JoinStrategy

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import pyarrow as pa
    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._plan.arrow.typing import NativeArrowDataFrame
    from narwhals._plan.compliant.dataframe import CompliantDataFrame, CompliantFrame
    from narwhals._typing import _EagerAllowedImpl


Incomplete: TypeAlias = Any


class BaseFrame(Generic[NativeFrameT_co]):
    _compliant: CompliantFrame[Any, NativeFrameT_co]
    _version: ClassVar[Version] = Version.MAIN

    @property
    def version(self) -> Version:
        return self._version

    @property
    def implementation(self) -> Implementation:
        return self._compliant.implementation

    @property
    def schema(self) -> Schema:
        return Schema(self._compliant.schema.items())

    @property
    def columns(self) -> list[str]:
        return self._compliant.columns

    def __repr__(self) -> str:  # pragma: no cover
        return generate_repr(f"nw.{type(self).__name__}", self.to_native().__repr__())

    def __init__(self, compliant: CompliantFrame[Any, NativeFrameT_co], /) -> None:
        self._compliant = compliant

    def _with_compliant(self, compliant: CompliantFrame[Any, Incomplete], /) -> Self:
        return type(self)(compliant)

    def to_native(self) -> NativeFrameT_co:
        return self._compliant.native

    def filter(
        self, *predicates: OneOrIterable[IntoExprColumn], **constraints: Any
    ) -> Self:
        e = _parse.parse_predicates_constraints_into_expr_ir(*predicates, **constraints)
        named_irs, _ = prepare_projection_s((e,), schema=self)
        if len(named_irs) != 1:
            # Should be unreachable, but I guess we will see
            msg = f"Expected a single predicate after expansion, but got {len(named_irs)!r}\n\n{named_irs!r}"
            raise ValueError(msg)
        return self._with_compliant(self._compliant.filter(named_irs[0]))

    def select(self, *exprs: OneOrIterable[IntoExpr], **named_exprs: Any) -> Self:
        named_irs, schema = prepare_projection(
            _parse.parse_into_seq_of_expr_ir(*exprs, **named_exprs), schema=self
        )
        return self._with_compliant(self._compliant.select(schema.select_irs(named_irs)))

    def with_columns(self, *exprs: OneOrIterable[IntoExpr], **named_exprs: Any) -> Self:
        named_irs, schema = prepare_projection_s(
            _parse.parse_into_seq_of_expr_ir(*exprs, **named_exprs), schema=self
        )
        return self._with_compliant(
            self._compliant.with_columns(schema.with_columns_irs(named_irs))
        )

    def sort(
        self,
        by: OneOrIterable[ColumnNameOrSelector],
        *more_by: ColumnNameOrSelector,
        descending: OneOrIterable[bool] = False,
        nulls_last: OneOrIterable[bool] = False,
    ) -> Self:
        sort = _parse.parse_sort_by_into_seq_of_expr_ir(by, *more_by)
        opts = SortMultipleOptions.parse(descending=descending, nulls_last=nulls_last)
        named_irs, _ = prepare_projection_s(sort, schema=self)
        return self._with_compliant(self._compliant.sort(named_irs, opts))

    def drop(self, *columns: str, strict: bool = True) -> Self:
        return self._with_compliant(self._compliant.drop(columns, strict=strict))

    def drop_nulls(self, subset: str | Sequence[str] | None = None) -> Self:
        subset = [subset] if isinstance(subset, str) else subset
        return self._with_compliant(self._compliant.drop_nulls(subset))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        return self._with_compliant(self._compliant.rename(mapping))


class DataFrame(
    BaseFrame[NativeDataFrameT_co], Generic[NativeDataFrameT_co, NativeSeriesT]
):
    _compliant: CompliantDataFrame[Any, NativeDataFrameT_co, NativeSeriesT]

    @property
    def implementation(self) -> _EagerAllowedImpl:
        return self._compliant.implementation

    def __len__(self) -> int:
        return len(self._compliant)

    @property
    def _series(self) -> type[Series[NativeSeriesT]]:
        return Series[NativeSeriesT]

    def _partial_series(
        self, *, dtype: IntoDType | None = None
    ) -> PartialSeries[NativeSeriesT]:
        it_names = temp.column_names(self.columns)
        backend = self.implementation
        series = self._series.from_iterable

        def fn(values: Iterable[Any], /) -> Series[NativeSeriesT]:
            return series(values, name=next(it_names), dtype=dtype, backend=backend)

        return fn

    @overload
    @classmethod
    def from_native(
        cls: type[DataFrame[Any, Any]], native: NativeArrowDataFrame, /
    ) -> DataFrame[pa.Table, pa.ChunkedArray[Any]]: ...
    @overload
    @classmethod
    def from_native(
        cls: type[DataFrame[Any, Any]], native: NativeDataFrameT, /
    ) -> DataFrame[NativeDataFrameT]: ...
    @classmethod
    def from_native(
        cls: type[DataFrame[Any, Any]], native: NativeDataFrameT, /
    ) -> DataFrame[Any, Any]:
        if is_pyarrow_table(native):
            from narwhals._plan.arrow.dataframe import ArrowDataFrame

            return cls(ArrowDataFrame.from_native(native, cls._version))

        raise NotImplementedError(type(native))

    @overload
    def to_dict(
        self, *, as_series: Literal[True] = ...
    ) -> dict[str, Series[NativeSeriesT]]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, Series[NativeSeriesT]] | dict[str, list[Any]]: ...
    def to_dict(
        self, *, as_series: bool = True
    ) -> dict[str, Series[NativeSeriesT]] | dict[str, list[Any]]:
        if as_series:
            return {
                key: self._series(value)
                for key, value in self._compliant.to_dict(as_series=as_series).items()
            }
        return self._compliant.to_dict(as_series=as_series)

    def to_series(self, index: int = 0) -> Series[NativeSeriesT]:
        return self._series(self._compliant.to_series(index))

    def get_column(self, name: str) -> Series[NativeSeriesT]:
        return self._series(self._compliant.get_column(name))

    @overload
    def group_by(
        self,
        *by: OneOrIterable[IntoExpr],
        drop_null_keys: Literal[False] = ...,
        **named_by: IntoExpr,
    ) -> GroupBy[Self]: ...

    @overload
    def group_by(
        self, *by: OneOrIterable[str], drop_null_keys: Literal[True]
    ) -> GroupBy[Self]: ...

    def group_by(
        self,
        *by: OneOrIterable[IntoExpr],
        drop_null_keys: bool = False,
        **named_by: IntoExpr,
    ) -> GroupBy[Self]:
        return Grouped.by(*by, drop_null_keys=drop_null_keys, **named_by).to_group_by(
            self
        )

    def row(self, index: int) -> tuple[Any, ...]:
        return self._compliant.row(index)

    def join(
        self,
        other: Self,
        on: str | Sequence[str] | None = None,
        how: JoinStrategy = "inner",
        *,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        suffix: str = "_right",
    ) -> Self:
        left, right = self._compliant, other._compliant
        how = _validate_join_strategy(how)
        if how == "cross":
            if left_on is not None or right_on is not None or on is not None:
                msg = "Can not pass `left_on`, `right_on` or `on` keys for cross join"
                raise ValueError(msg)
            return self._with_compliant(left.join_cross(right, suffix=suffix))
        left_on, right_on = normalize_join_on(on, how, left_on, right_on)
        return self._with_compliant(
            left.join(right, how=how, left_on=left_on, right_on=right_on, suffix=suffix)
        )

    def filter(
        self, *predicates: OneOrIterable[IntoExprColumn] | list[bool], **constraints: Any
    ) -> Self:
        e = _parse.parse_predicates_constraints_into_expr_ir(
            *predicates,
            _list_as_series=self._partial_series(dtype=self.version.dtypes.Boolean()),
            **constraints,
        )
        named_irs, _ = prepare_projection_s((e,), schema=self)
        if len(named_irs) != 1:
            # Should be unreachable, but I guess we will see
            msg = f"Expected a single predicate after expansion, but got {len(named_irs)!r}\n\n{named_irs!r}"
            raise ValueError(msg)
        return self._with_compliant(self._compliant.filter(named_irs[0]))

    def partition_by(
        self,
        by: OneOrIterable[ColumnNameOrSelector],
        *more_by: ColumnNameOrSelector,
        include_key: bool = True,
    ) -> list[Self]:
        by_selectors = _parse.parse_into_seq_of_selector_ir(by, *more_by)
        names = expand_selector_irs_names(by_selectors, schema=self)
        if not names:
            raise group_by_no_keys_error()
        partitions = self._compliant.partition_by(names, include_key=include_key)
        return [self._with_compliant(p) for p in partitions]


def _is_join_strategy(obj: Any) -> TypeIs[JoinStrategy]:
    return obj in {"inner", "left", "full", "cross", "anti", "semi"}


def _validate_join_strategy(how: str, /) -> JoinStrategy:
    if _is_join_strategy(how):
        return how
    msg = f"Only the following join strategies are supported: {get_args(JoinStrategy)}; found '{how}'."
    raise NotImplementedError(msg)


def normalize_join_on(
    on: OneOrIterable[str] | None,
    how: NonCrossJoinStrategy,
    left_on: OneOrIterable[str] | None,
    right_on: OneOrIterable[str] | None,
    /,
) -> tuple[Seq[str], Seq[str]]:
    """Reduce the 3 potential key (`on*`) arguments to 2.

    Ensures the keys spelling is compatible with the join strategy.
    """
    if on is None:
        if left_on is None or right_on is None:
            msg = f"Either (`left_on` and `right_on`) or `on` keys should be specified for {how}."
            raise ValueError(msg)
        left_on = ensure_seq_str(left_on)
        right_on = ensure_seq_str(right_on)
        if len(left_on) != len(right_on):
            msg = "`left_on` and `right_on` must have the same length."
            raise ValueError(msg)
        return left_on, right_on
    if left_on is not None or right_on is not None:
        msg = f"If `on` is specified, `left_on` and `right_on` should be None for {how}."
        raise ValueError(msg)
    on = ensure_seq_str(on)
    return on, on
