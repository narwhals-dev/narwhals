from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, overload

from narwhals._plan import _parse
from narwhals._plan._expansion import prepare_projection
from narwhals._plan.expr import _parse_sort_by
from narwhals._plan.group_by import GroupBy, Grouped
from narwhals._plan.series import Series
from narwhals._plan.typing import (
    IntoExpr,
    NativeDataFrameT,
    NativeDataFrameT_co,
    NativeFrameT_co,
    NativeSeriesT,
    OneOrIterable,
)
from narwhals._utils import Version, generate_repr
from narwhals.dependencies import is_pyarrow_table
from narwhals.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.arrow.typing import NativeArrowDataFrame
    from narwhals._plan.compliant.dataframe import CompliantDataFrame, CompliantFrame
    from narwhals.typing import JoinStrategy

Incomplete: TypeAlias = Any


class BaseFrame(Generic[NativeFrameT_co]):
    _compliant: CompliantFrame[Any, NativeFrameT_co]
    _version: ClassVar[Version] = Version.MAIN

    @property
    def version(self) -> Version:
        return self._version

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

    def select(self, *exprs: OneOrIterable[IntoExpr], **named_exprs: Any) -> Self:
        named_irs, schema = prepare_projection(
            _parse.parse_into_seq_of_expr_ir(*exprs, **named_exprs), schema=self
        )
        return self._with_compliant(self._compliant.select(schema.select_irs(named_irs)))

    def with_columns(self, *exprs: OneOrIterable[IntoExpr], **named_exprs: Any) -> Self:
        named_irs, schema = prepare_projection(
            _parse.parse_into_seq_of_expr_ir(*exprs, **named_exprs), schema=self
        )
        return self._with_compliant(
            self._compliant.with_columns(schema.with_columns_irs(named_irs))
        )

    def sort(
        self,
        by: OneOrIterable[str],
        *more_by: str,
        descending: OneOrIterable[bool] = False,
        nulls_last: OneOrIterable[bool] = False,
    ) -> Self:
        sort, opts = _parse_sort_by(
            by, *more_by, descending=descending, nulls_last=nulls_last
        )
        named_irs, _ = prepare_projection(sort, schema=self)
        return self._with_compliant(self._compliant.sort(named_irs, opts))

    def drop(self, *columns: str, strict: bool = True) -> Self:
        return self._with_compliant(self._compliant.drop(columns, strict=strict))

    def drop_nulls(self, subset: str | Sequence[str] | None = None) -> Self:
        subset = [subset] if isinstance(subset, str) else subset
        return self._with_compliant(self._compliant.drop_nulls(subset))


class DataFrame(
    BaseFrame[NativeDataFrameT_co], Generic[NativeDataFrameT_co, NativeSeriesT]
):
    _compliant: CompliantDataFrame[Any, NativeDataFrameT_co, NativeSeriesT]

    def __len__(self) -> int:
        return len(self._compliant)

    @property
    def _series(self) -> type[Series[NativeSeriesT]]:
        return Series[NativeSeriesT]

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
        if on is not None:
            on = [on] if isinstance(on, str) else list(on)
        if left_on is not None:
            left_on = [left_on] if isinstance(left_on, str) else list(left_on)
        if right_on is not None:
            right_on = [right_on] if isinstance(right_on, str) else list(right_on)
        result = self._compliant.join(
            other._compliant, how=how, left_on=left_on, right_on=right_on, suffix=suffix
        )
        return self._with_compliant(result)
