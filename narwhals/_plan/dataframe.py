from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, get_args, overload

from narwhals._plan import _parse
from narwhals._plan._expansion import expand_selector_irs_names, prepare_projection
from narwhals._plan._guards import is_series
from narwhals._plan.common import ensure_seq_str, normalize_target_file, temp, todo
from narwhals._plan.compliant.dataframe import EagerDataFrame
from narwhals._plan.compliant.namespace import EagerNamespace
from narwhals._plan.group_by import GroupBy, Grouped, LazyGroupBy
from narwhals._plan.logical_plan import LogicalPlan
from narwhals._plan.options import (
    ExplodeOptions,
    JoinOptions,
    SortMultipleOptions,
    UniqueOptions,
    UnpivotOptions,
)
from narwhals._plan.series import Series
from narwhals._plan.typing import (
    ColumnNameOrSelector,
    IncompleteCyclic,
    IntoExpr,
    IntoExprColumn,
    NativeDataFrameT,
    NativeDataFrameT_co,
    NativeFrameT_co,
    NativeLazyFrameT_co,
    NativeSeriesT,
    NativeSeriesT2,
    NonCrossJoinStrategy,
    OneOrIterable,
    PartialSeries,
    Seq,
)
from narwhals._utils import (
    Implementation,
    Version,
    check_column_names_are_unique as raise_duplicate_error,
    generate_repr,
    not_implemented,
    qualified_type_name,
)
from narwhals.dependencies import is_pyarrow_table
from narwhals.exceptions import InvalidOperationError, ShapeError
from narwhals.schema import Schema
from narwhals.typing import (
    AsofJoinStrategy,
    EagerAllowed,
    FileSource,
    IntoBackend,
    IntoDType,
    IntoSchema,
    JoinStrategy,
    LazyAllowed,
    PivotAgg,
    UniqueKeepStrategy,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from io import BytesIO

    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._native import NativeSeries
    from narwhals._plan.arrow.typing import NativeArrowDataFrame
    from narwhals._plan.compliant.dataframe import (
        CompliantFrame,
        CompliantLazyFrame,
        EagerDataFrame,
    )
    from narwhals._plan.compliant.namespace import EagerNamespace
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._typing import Arrow, _EagerAllowedImpl

    EagerNs: TypeAlias = EagerNamespace[
        EagerDataFrame[Any, NativeDataFrameT, Any],
        CompliantSeries[NativeSeriesT],
        Any,
        Any,
    ]


Incomplete: TypeAlias = Any


class BaseFrame(Generic[NativeFrameT_co]):
    _compliant: CompliantFrame[Any, NativeFrameT_co]
    _version: ClassVar[Version] = Version.MAIN

    @property
    def version(self) -> Version:
        return self._version

    @property
    def implementation(self) -> Implementation:  # pragma: no cover
        return self._compliant.implementation

    @property
    def schema(self) -> Schema:
        return Schema(self._compliant.schema.items())

    @property
    def columns(self) -> list[str]:
        return self._compliant.columns

    def __repr__(self) -> str:
        return generate_repr(f"nw.{type(self).__name__}", self.to_native().__repr__())

    def __init__(self, compliant: CompliantFrame[Any, NativeFrameT_co], /) -> None:
        self._compliant = compliant

    def _unwrap_compliant(self, other: Self | Any, /) -> Incomplete:
        """Return the `CompliantFrame` that backs `other` if it matches self.

        - Rejects (`DataFrame`, `LazyFrame`) and (`LazyFrame`, `DataFrame`)
        - Rejects mixed backends like (`DataFrame[pa.Table]`, `DataFrame[pd.DataFrame]`)
        """
        if isinstance(other, type(self)):
            compliant = other._compliant
            if isinstance(compliant, type(self._compliant)):
                return compliant
            msg = f"Expected {qualified_type_name(self._compliant)!r}, got {qualified_type_name(compliant)!r}"
            raise NotImplementedError(msg)
        msg = f"Expected `other` to be a {qualified_type_name(self)!r}, got: {qualified_type_name(other)!r}"  # pragma: no cover
        raise TypeError(msg)  # pragma: no cover

    def _with_compliant(self, compliant: CompliantFrame[Any, Incomplete], /) -> Self:
        return type(self)(compliant)

    def to_native(self) -> NativeFrameT_co:
        return self._compliant.native

    def filter(
        self, *predicates: OneOrIterable[IntoExprColumn], **constraints: Any
    ) -> Self:  # pragma: no cover
        e = _parse.parse_predicates_constraints_into_expr_ir(*predicates, **constraints)
        named_irs, _ = prepare_projection((e,), schema=self)
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
        named_irs, schema = prepare_projection(
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
        s_irs = _parse.parse_into_seq_of_selector_ir(by, *more_by)
        names = expand_selector_irs_names(s_irs, schema=self, require_any=True)
        opts = SortMultipleOptions.parse(descending=descending, nulls_last=nulls_last)
        return self._with_compliant(self._compliant.sort(names, opts))

    def drop(
        self, *columns: OneOrIterable[ColumnNameOrSelector], strict: bool = True
    ) -> Self:
        s_ir = _parse.parse_into_combined_selector_ir(*columns, require_all=strict)
        if names := expand_selector_irs_names((s_ir,), schema=self):
            compliant = self._compliant.drop(names)
        else:
            compliant = self._compliant._with_native(self.to_native())
        return self._with_compliant(compliant)

    def drop_nulls(
        self, subset: OneOrIterable[ColumnNameOrSelector] | None = None
    ) -> Self:
        if subset is not None:
            s_irs = _parse.parse_into_seq_of_selector_ir(subset)
            subset = expand_selector_irs_names(s_irs, schema=self) or None
        return self._with_compliant(self._compliant.drop_nulls(subset))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        return self._with_compliant(self._compliant.rename(mapping))

    def collect_schema(self) -> Schema:
        return self.schema

    def unpivot(
        self,
        on: OneOrIterable[ColumnNameOrSelector] | None = None,
        *,
        index: OneOrIterable[ColumnNameOrSelector] | None = None,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> Self:
        on_: Seq[str] | None = None
        index_: Seq[str] | None = None
        schema = self.schema
        if on is not None:
            s_irs = _parse.parse_into_seq_of_selector_ir(on)
            on_ = expand_selector_irs_names(s_irs, schema=schema, require_any=True)
        if index is not None:
            s_irs = _parse.parse_into_seq_of_selector_ir(index)
            index_ = expand_selector_irs_names(s_irs, schema=schema, require_any=True)
        return self._with_compliant(
            self._compliant.unpivot(
                on_, index_, variable_name=variable_name, value_name=value_name
            )
        )

    def with_row_index(
        self, name: str = "index", *, order_by: OneOrIterable[ColumnNameOrSelector]
    ) -> Self:
        by_selectors = _parse.parse_into_seq_of_selector_ir(order_by)
        by_names = expand_selector_irs_names(by_selectors, schema=self, require_any=True)
        return self._with_compliant(self._compliant.with_row_index_by(name, by_names))

    def join(
        self,
        other: Incomplete,
        on: str | Sequence[str] | None = None,
        how: JoinStrategy = "inner",
        *,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        suffix: str = "_right",
    ) -> Self:
        left = self._compliant
        right: CompliantFrame[Any, NativeFrameT_co] = self._unwrap_compliant(other)
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

    def join_asof(
        self,
        other: Incomplete,
        *,
        left_on: str | None = None,
        right_on: str | None = None,
        on: str | None = None,
        by_left: str | Sequence[str] | None = None,
        by_right: str | Sequence[str] | None = None,
        by: str | Sequence[str] | None = None,
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
    ) -> Self:
        left = self._compliant
        right: CompliantFrame[Any, NativeFrameT_co] = self._unwrap_compliant(other)
        strategy = _validate_join_asof_strategy(strategy)
        left_on_, right_on_ = normalize_join_asof_on(left_on, right_on, on)
        if by_left or by_right or by:
            left_by, right_by = normalize_join_asof_by(by_left, by_right, by)
            result = left.join_asof(
                right,
                left_on=left_on_,
                right_on=right_on_,
                left_by=left_by,
                right_by=right_by,
                strategy=strategy,
                suffix=suffix,
            )
        else:
            result = left.join_asof(
                right,
                left_on=left_on_,
                right_on=right_on_,
                strategy=strategy,
                suffix=suffix,
            )
        return self._with_compliant(result)

    def explode(
        self,
        columns: OneOrIterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
        empty_as_null: bool = True,
        keep_nulls: bool = True,
    ) -> Self:
        s_ir = _parse.parse_into_combined_selector_ir(columns, *more_columns)
        schema = self.collect_schema()
        subset = expand_selector_irs_names((s_ir,), schema=schema, require_any=True)
        tp_list = self.version.dtypes.List
        for col_to_explode in subset:
            dtype = schema[col_to_explode]
            if not isinstance(dtype, tp_list):
                msg = f"`explode` operation is not supported for dtype `{dtype}`, expected List type"
                raise InvalidOperationError(msg)
        options = ExplodeOptions(empty_as_null=empty_as_null, keep_nulls=keep_nulls)
        return self._with_compliant(self._compliant.explode(subset, options))

    def unnest(
        self,
        columns: OneOrIterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> Self:
        s_ir = _parse.parse_into_combined_selector_ir(columns, *more_columns)
        schema = self.collect_schema()
        subset = expand_selector_irs_names((s_ir,), schema=schema, require_any=True)
        tp_struct = self.version.dtypes.Struct
        existing_names = schema.keys() - subset
        for col_to_unnest in subset:
            dtype = schema[col_to_unnest]
            if not isinstance(dtype, tp_struct):
                msg = f"`unnest` operation is not supported for dtype `{dtype}`, expected Struct type"
                raise InvalidOperationError(msg)
            field_names = {fld.name for fld in dtype.fields}
            if existing_names.isdisjoint(field_names):
                existing_names |= field_names
            else:
                raise_duplicate_error([*existing_names, *field_names])
        return self._with_compliant(self._compliant.unnest(subset))


def _dataframe_from_dict(
    data: Mapping[str, Any],
    schema: IntoSchema | None,
    ns: EagerNs[NativeDataFrameT, NativeSeriesT],
    /,
) -> DataFrame[NativeDataFrameT, NativeSeriesT]:
    return ns._dataframe.from_dict(data, schema=schema).to_narwhals()


class DataFrame(
    BaseFrame[NativeDataFrameT_co], Generic[NativeDataFrameT_co, NativeSeriesT]
):
    _compliant: EagerDataFrame[IncompleteCyclic, NativeDataFrameT_co, NativeSeriesT]

    def __narwhals_namespace__(
        self,
    ) -> EagerNamespace[
        EagerDataFrame[Any, NativeDataFrameT_co, NativeSeriesT],
        CompliantSeries[NativeSeriesT],
        Any,
        Any,
    ]:
        return self._compliant.__narwhals_namespace__()

    @property
    def implementation(self) -> _EagerAllowedImpl:
        return self._compliant.implementation

    @property
    def shape(self) -> tuple[int, int]:
        return self._compliant.shape

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

    def _parse_into_compliant_series(
        self, other: Series[Any] | Iterable[Any], /, name: str = ""
    ) -> CompliantSeries[NativeSeriesT]:
        if columns := self.columns:
            compliant = self.get_column(columns[0])._parse_into_compliant(other)
            return compliant if not name or compliant.name else compliant.alias(name)
        else:  # pragma: no cover # noqa: RET505
            tp_series = self.__narwhals_namespace__()._series
            if not is_series(other):
                return tp_series.from_iterable(other, version=self.version, name=name)
            s = other._compliant
            if isinstance(s, tp_series):
                return s
            msg = f"Expected {qualified_type_name(tp_series)!r}, got {qualified_type_name(s)!r}"
            raise NotImplementedError(msg)

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
            from narwhals._plan import arrow as _arrow

            return cls(_arrow.DataFrame.from_native(native, cls._version))

        raise NotImplementedError(type(native))

    @overload
    @classmethod
    def from_dict(
        cls: type[DataFrame[Any, Any]],
        data: Mapping[str, Any],
        schema: IntoSchema | None = ...,
        *,
        backend: Arrow,
    ) -> DataFrame[pa.Table, pa.ChunkedArray[Any]]: ...
    @overload
    @classmethod
    def from_dict(
        cls: type[DataFrame[Any, Any]],
        data: Mapping[str, Any],
        schema: IntoSchema | None = ...,
        *,
        backend: IntoBackend[EagerAllowed],
    ) -> DataFrame[Any, Any]: ...
    @overload
    @classmethod
    def from_dict(
        cls: type[DataFrame[Any, Any]],
        data: Mapping[str, Series[NativeSeriesT2]],
        schema: IntoSchema | None = ...,
    ) -> DataFrame[Any, NativeSeriesT2]: ...
    @classmethod
    def from_dict(
        cls: type[DataFrame[Any, Any]],
        data: Mapping[str, Any],
        schema: IntoSchema | None = None,
        *,
        backend: IntoBackend[EagerAllowed] | None = None,
    ) -> DataFrame[Any, Any]:
        from narwhals._plan import functions as F

        if backend is None:
            unwrapped: dict[str, NativeSeries | Any] = {}
            impl: _EagerAllowedImpl | None = backend
            for k, v in data.items():
                if is_series(v):
                    current = v.implementation
                    if impl is None:
                        impl = current
                    elif current is not impl:
                        msg = f"All `Series` must share the same backend, but got:\n  -{impl!r}\n  -{current!r}"
                        raise NotImplementedError(msg)
                    unwrapped[k] = v.to_native()
                else:
                    unwrapped[k] = v
            if impl is None:
                msg = "Calling `from_dict` without `backend` is only supported if all input values are already Narwhals Series"
                raise TypeError(msg)
            return _dataframe_from_dict(unwrapped, schema, F._eager_namespace(impl))

        ns = F._eager_namespace(backend)
        return _dataframe_from_dict(data, schema, ns)

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
        if as_series:  # pragma: no cover
            return {
                key: self._series(value)
                for key, value in self._compliant.to_dict(as_series=as_series).items()
            }
        return self._compliant.to_dict(as_series=as_series)

    def to_series(self, index: int = 0) -> Series[NativeSeriesT]:
        return self._series(self._compliant.to_series(index))

    def to_struct(self, name: str = "") -> Series[NativeSeriesT]:
        return self._series(self._compliant.to_struct(name))

    def to_polars(self) -> pl.DataFrame:
        return self._compliant.to_polars()

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self._with_compliant(self._compliant.gather_every(n, offset))

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

    def iter_columns(self) -> Iterator[Series[NativeSeriesT]]:
        for series in self._compliant.iter_columns():
            yield self._series(series)

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
        return super().join(
            other, how=how, left_on=left_on, right_on=right_on, on=on, suffix=suffix
        )

    def join_asof(
        self,
        other: Self,
        *,
        left_on: str | None = None,
        right_on: str | None = None,
        on: str | None = None,
        by_left: str | Sequence[str] | None = None,
        by_right: str | Sequence[str] | None = None,
        by: str | Sequence[str] | None = None,
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
    ) -> Self:
        return super().join_asof(
            other,
            left_on=left_on,
            right_on=right_on,
            on=on,
            by_left=by_left,
            by_right=by_right,
            by=by,
            strategy=strategy,
            suffix=suffix,
        )

    def filter(
        self, *predicates: OneOrIterable[IntoExprColumn] | list[bool], **constraints: Any
    ) -> Self:
        e = _parse.parse_predicates_constraints_into_expr_ir(
            *predicates,
            _list_as_series=self._partial_series(dtype=self.version.dtypes.Boolean()),
            **constraints,
        )
        named_irs, _ = prepare_projection((e,), schema=self)
        if len(named_irs) != 1:  # pragma: no cover
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
        names = expand_selector_irs_names(by_selectors, schema=self, require_any=True)
        partitions = self._compliant.partition_by(names, include_key=include_key)
        return [self._with_compliant(p) for p in partitions]

    # TODO @dangotbanned: (Follow-up) Accept selectors in `on`, `index`, `values`
    def pivot(
        self,
        on: OneOrIterable[str],
        on_columns: Sequence[str] | Series | Self | None = None,
        *,
        index: OneOrIterable[str] | None = None,
        values: OneOrIterable[str] | None = None,
        aggregate_function: PivotAgg | None = None,
        sort_columns: bool = False,
        separator: str = "_",
    ) -> Self:
        from narwhals._plan import functions as F

        on_, index_, values_ = normalize_pivot_args(
            on, index=index, values=values, frame_columns=self.columns
        )
        dtype_str = self.version.dtypes.String()
        on_cols: EagerDataFrame[IncompleteCyclic, NativeDataFrameT_co, NativeSeriesT]

        if on_columns is None:
            nw_on_cols = self.select(F.col(name).cast(dtype_str) for name in on_).unique(
                on_, maintain_order=True
            )
            if sort_columns:
                nw_on_cols = nw_on_cols.sort(on_)
            on_cols = nw_on_cols._compliant
        elif isinstance(on_columns, DataFrame):
            on_cols = on_columns._compliant
        else:
            on_cols = (
                self._parse_into_compliant_series(on_columns, on_[0])
                .cast(dtype_str)
                .to_frame()
            )

        if len(on_) != on_cols.width:
            msg = "`pivot` expected `on` and `on_columns` to have the same amount of columns."
            raise InvalidOperationError(msg)
        if on_ != tuple(on_cols.columns):
            msg = "`pivot` has mismatching column names between `on` and `on_columns`."
            raise InvalidOperationError(msg)

        return self._with_compliant(
            self._compliant.pivot(
                on_,
                on_cols,
                index=index_,
                values=values_,
                aggregate_function=aggregate_function,
                separator=separator,
            )
        )

    def sort(
        self,
        by: OneOrIterable[ColumnNameOrSelector],
        *more_by: ColumnNameOrSelector,
        descending: OneOrIterable[bool] = False,
        nulls_last: OneOrIterable[bool] = False,
    ) -> Self:
        if (
            not more_by
            and _is_sort_by_one(by, self.columns)
            and isinstance(descending, bool)
            and isinstance(nulls_last, bool)
        ):
            return self._with_compliant(
                self.to_series()
                ._compliant.sort(descending=descending, nulls_last=nulls_last)
                .to_frame()
            )
        return super().sort(by, *more_by, descending=descending, nulls_last=nulls_last)

    def unique(
        self,
        subset: OneOrIterable[ColumnNameOrSelector] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
        order_by: OneOrIterable[ColumnNameOrSelector] | None = None,
    ) -> Self:
        keep = _validate_unique_keep_strategy(keep)
        schema = self.schema
        subset_names: Sequence[str] | None = None
        if subset is not None:
            s_irs = _parse.parse_into_seq_of_selector_ir(subset)
            subset_names = expand_selector_irs_names(
                s_irs, schema=schema, require_any=True
            )
        if order_by is None:
            if len(schema) == 1 and keep in {"any", "first"}:
                # NOTE: Fastpath for single-column frame
                result = (
                    self.to_series()
                    ._compliant.unique(maintain_order=maintain_order)
                    .to_frame()
                )
            else:
                result = self._compliant.unique(
                    subset_names, keep=keep, maintain_order=maintain_order
                )
            return self._with_compliant(result)
        s_irs = _parse.parse_into_seq_of_selector_ir(order_by)
        by_names = expand_selector_irs_names(s_irs, schema=schema, require_any=True)
        return self._with_compliant(
            self._compliant.unique_by(
                subset_names, keep=keep, maintain_order=maintain_order, order_by=by_names
            )
        )

    def with_row_index(
        self,
        name: str = "index",
        *,
        order_by: OneOrIterable[ColumnNameOrSelector] | None = None,
    ) -> Self:
        if order_by is None:
            return self._with_compliant(self._compliant.with_row_index(name))
        return super().with_row_index(name, order_by=order_by)

    @overload
    def write_csv(self, file: None = None) -> str: ...
    @overload
    def write_csv(self, file: FileSource | BytesIO) -> None: ...
    def write_csv(self, file: FileSource | BytesIO | None = None) -> str | None:
        return self._compliant.write_csv(normalize_target_file(file))

    def write_parquet(self, file: FileSource | BytesIO) -> None:
        return self._compliant.write_parquet(normalize_target_file(file))

    def slice(self, offset: int, length: int | None = None) -> Self:
        return type(self)(self._compliant.slice(offset=offset, length=length))

    def sample(
        self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> Self:
        if n is not None and fraction is not None:
            msg = "cannot specify both `n` and `fraction`"
            raise ValueError(msg)
        df = self._compliant
        if fraction is not None:
            result = df.sample_frac(
                fraction, with_replacement=with_replacement, seed=seed
            )
        elif n is None:
            result = df.sample_n(with_replacement=with_replacement, seed=seed)
        elif not with_replacement and n > len(self):
            msg = "cannot take a larger sample than the total population when `with_replacement=false`"
            raise ShapeError(msg)
        else:
            result = df.sample_n(n, with_replacement=with_replacement, seed=seed)
        return type(self)(result)

    def clone(self) -> Self:
        """Create a copy of this DataFrame."""
        return type(self)(self._compliant.clone())

    # NOTE: Using for testing
    # Actually integrating will mean rewriting the current `DataFrame`, `BaseFrame` impls
    # Everything is one-way, you can build a `LogicalPlan` but nothing useful beyond that
    def _to_lp(self) -> LogicalPlan:  # pragma: no cover
        return LogicalPlan.from_df(self)

    def _lazy(
        self, backend: IntoBackend[LazyAllowed] | None = None
    ) -> LazyFrame[Any]:  # pragma: no cover
        if backend is not None:
            msg = f"Lazy backends are not yet supported in `narwhals._plan`, got: {backend!r}"
            raise NotImplementedError(msg)
        return LazyFrame._from_lp(self._to_lp())


class LazyFrame(
    BaseFrame[NativeLazyFrameT_co], Generic[NativeLazyFrameT_co]
):  # pragma: no cover
    """WIP: need to change a lot before something useful can happen.

    ## Notes
    `CompliantLazyFrame` seems like the wrong abstraction,
    but preserving the native type is non-negotiable.

    ### Different operations
    I'd like to acknowledge

    (1) Fake lazy (needs a reference to eager data):

        DataFrame(<native-df>).lazy(None)
        # becomes
        LazyFrame._from_lp(ScanDataFrame.from_narwhals(<narwhals-df>))

    (2) Native lazy (needs a reference to lazy query):

        LazyFrame.from_native(<native_lf>)
        # becomes
        LazyFrame._from_lp(<some-new-node>)

    (3) Lazy from file (needs a reference to `Implementation`):

        scan_parquet(source, backend=backend)
        # becomes
        LazyFrame._from_lp(ScanFile.from_source(source, Implementation.from_backend(backend)))

    (4) Eager -> lazy conversion (needs a reference to lazy query, [maybe `Implementation`]):

        DataFrame(<native-df>).lazy(<backend-conversion>)
        # do the conversion ...
        LazyFrame.from_native(<converted-to-native-lf>)
        LazyFrame._from_lp(<some-new-node>)

    [maybe `Implementation`]: https://github.com/narwhals-dev/narwhals/issues/3210
    """

    _compliant: CompliantLazyFrame[Incomplete, NativeLazyFrameT_co]
    _plan: LogicalPlan

    # NOTE: Need to override most of `BaseFrame` for now
    @classmethod
    def _from_lp(cls: type[LazyFrame[Any]], plan: LogicalPlan, /) -> LazyFrame[Any]:
        obj = cls.__new__(cls)
        obj._plan = plan
        return obj

    def _with_lp(self, plan: LogicalPlan, /) -> Self:
        tp = type(self)
        obj = tp.__new__(tp)
        obj._plan = plan
        return obj

    implementation = todo()  # type: ignore[assignment]
    to_native = todo()
    _unwrap_compliant = todo()
    _with_compliant = todo()

    columns = todo()  # type: ignore[assignment]
    schema = todo()  # type: ignore[assignment]
    collect_schema = todo()
    collect = not_implemented()  # depends on resolving everything else

    join_asof = not_implemented()  # not in `LogicalPlan`

    def _unwrap_plan(self, other: Self | Any, /) -> LogicalPlan:
        """Equivalent* to `BaseFrame._unwrap_compliant`, used for `join(other)`."""
        if isinstance(other, type(self)):
            # TODO @dangotbanned: Handle `Implementation` matching of `_plan`
            # Requires introducing the concept to `LogicalPlan` first
            return other._plan
        msg = f"Expected `other` to be a {qualified_type_name(self)!r}, got: {qualified_type_name(other)!r}"  # pragma: no cover
        raise TypeError(msg)  # pragma: no cover

    def __repr__(self) -> str:
        return "<LazyFrame todo>"

    def drop(
        self, *columns: OneOrIterable[ColumnNameOrSelector], strict: bool = True
    ) -> Self:
        s_ir = _parse.parse_into_combined_selector_ir(*columns, require_all=strict)
        return self._with_lp(self._plan.drop(s_ir))

    def drop_nulls(
        self, subset: OneOrIterable[ColumnNameOrSelector] | None = None
    ) -> Self:
        s_ir = None if subset is None else _parse.parse_into_combined_selector_ir(subset)
        return self._with_lp(self._plan.drop_nulls(s_ir))

    def explode(
        self,
        columns: OneOrIterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
        empty_as_null: bool = True,
        keep_nulls: bool = True,
    ) -> Self:
        s_ir = _parse.parse_into_combined_selector_ir(columns, *more_columns)
        options = ExplodeOptions(empty_as_null=empty_as_null, keep_nulls=keep_nulls)
        return self._with_lp(self._plan.explode(s_ir, options))

    def filter(
        self, *predicates: OneOrIterable[IntoExprColumn], **constraints: Any
    ) -> Self:  # pragma: no cover
        p = _parse.parse_predicates_constraints_into_expr_ir(*predicates, **constraints)
        return self._with_lp(self._plan.filter(p))

    # TODO @dangotbanned: LazyFrame.group_by(drop_null_keys=True)
    def group_by(
        self,
        *by: OneOrIterable[IntoExpr],
        drop_null_keys: bool = False,
        **named_by: IntoExpr,
    ) -> LazyGroupBy[Self]:
        return LazyGroupBy(self, *by, drop_null_keys=drop_null_keys, **named_by)

    def head(self, n: int = 5) -> Self:
        return self._with_lp(self._plan.head(n))

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
        left, right = self._plan, self._unwrap_plan(other)
        if how == "cross":
            if left_on is not None or right_on is not None or on is not None:
                msg = "Can not pass `left_on`, `right_on` or `on` keys for cross join"
                raise ValueError(msg)
            return self._with_lp(left.join_cross(right, suffix=suffix))
        opts = JoinOptions(how=_validate_join_strategy(how), suffix=suffix)
        left_on, right_on = normalize_join_on(on, how, left_on, right_on)
        return self._with_lp(left.join(right, left_on, right_on, opts))

    # TODO @dangotbanned: Figure out how `on_columns: ...` should work for lazy (besides polars)
    # See https://github.com/narwhals-dev/narwhals/issues/1901#issuecomment-3697700426
    def pivot(
        self,
        on: OneOrIterable[ColumnNameOrSelector],
        on_columns: Sequence[str] | Series | DataFrame,
        *,
        index: OneOrIterable[ColumnNameOrSelector] | None = None,
        values: OneOrIterable[ColumnNameOrSelector] | None = None,
        aggregate_function: PivotAgg | None = None,
        separator: str = "_",
    ) -> Self:
        from narwhals._plan import selectors as cs

        on_ = _parse.parse_into_combined_selector_ir(on)
        if index is None:
            if values is None:
                msg = "`pivot` needs either `index or `values` needs to be specified"
                raise InvalidOperationError(msg)
            values_ = _parse.parse_into_combined_selector_ir(values)
            index_ = (cs.all() - on_.to_narwhals() - values_.to_narwhals())._ir
        else:
            index_ = _parse.parse_into_combined_selector_ir(index)
            if values is not None:
                values_ = _parse.parse_into_combined_selector_ir(values)
            else:
                values_ = (cs.all() - on_.to_narwhals() - index_.to_narwhals())._ir

        return self._with_lp(
            self._plan.pivot(
                on_,
                on_columns=on_columns,  # type: ignore[arg-type]
                index=index_,
                values=values_,
                agg=aggregate_function,
                separator=separator,
            )
        )

    def rename(self, mapping: Mapping[str, str]) -> Self:
        return self._with_lp(self._plan.rename(mapping))

    def select(self, *exprs: OneOrIterable[IntoExpr], **named_exprs: Any) -> Self:
        e_irs = _parse.parse_into_seq_of_expr_ir(*exprs, **named_exprs)
        return self._with_lp(self._plan.select(e_irs))

    # TODO @dangotbanned: Figure out `LazyFrame.sink_parquet -> None`
    def sink_parquet(self, file: FileSource | BytesIO) -> None:
        _ = self._plan.sink_parquet(file)
        msg = "TODO: LazyFrame.sink_parquet"
        raise NotImplementedError(msg)

    def sort(
        self,
        by: OneOrIterable[ColumnNameOrSelector],
        *more_by: ColumnNameOrSelector,
        descending: OneOrIterable[bool] = False,
        nulls_last: OneOrIterable[bool] = False,
    ) -> Self:
        s_irs = _parse.parse_into_seq_of_selector_ir(by, *more_by)
        opts = SortMultipleOptions.parse(descending=descending, nulls_last=nulls_last)
        return self._with_lp(self._plan.sort(s_irs, opts))

    def tail(self, n: int = 5) -> Self:
        return self._with_lp(self._plan.tail(n))

    def unique(
        self,
        subset: OneOrIterable[ColumnNameOrSelector] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        order_by: OneOrIterable[ColumnNameOrSelector] | None = None,
    ) -> Self:
        opts = UniqueOptions.lazy(_validate_unique_keep_strategy(keep))
        parse = _parse.parse_into_seq_of_selector_ir
        s_subset = None if subset is None else parse(subset)
        if order_by is not None:
            return self._with_lp(self._plan.unique_by(s_subset, opts, parse(order_by)))
        if keep in {"first", "last"}:
            msg = "'first' and 'last' are only supported if `order_by` is passed."
            raise InvalidOperationError(msg)
        return self._with_lp(self._plan.unique(s_subset, opts))

    def unnest(
        self,
        columns: OneOrIterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> Self:
        s_ir = _parse.parse_into_combined_selector_ir(columns, *more_columns)
        return self._with_lp(self._plan.unnest(s_ir))

    def unpivot(
        self,
        on: OneOrIterable[ColumnNameOrSelector] | None = None,
        *,
        index: OneOrIterable[ColumnNameOrSelector] | None = None,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> Self:
        s_on = on if on is None else _parse.parse_into_combined_selector_ir(on)
        s_index = None if index is None else _parse.parse_into_combined_selector_ir(index)
        options = UnpivotOptions(variable_name=variable_name, value_name=value_name)
        return self._with_lp(self._plan.unpivot(s_on, index=s_index, options=options))

    def with_columns(self, *exprs: OneOrIterable[IntoExpr], **named_exprs: Any) -> Self:
        e_irs = _parse.parse_into_seq_of_expr_ir(*exprs, **named_exprs)
        return self._with_lp(self._plan.with_columns(e_irs))

    def with_row_index(
        self, name: str = "index", *, order_by: OneOrIterable[ColumnNameOrSelector]
    ) -> Self:
        by = _parse.parse_into_seq_of_selector_ir(order_by)
        return self._with_lp(self._plan.with_row_index_by(name, order_by=by))


def _is_sort_by_one(
    by: OneOrIterable[ColumnNameOrSelector], frame_columns: list[str]
) -> bool:
    """Return True if requested to sort a single-column DataFrame - without consuming iterators."""
    columns = frame_columns
    if len(columns) != 1:
        return False
    return (isinstance(by, str) and by in columns) or (
        isinstance(by, Sequence) and len(by) == 1 and by[0] in columns
    )


def _is_join_strategy(obj: Any) -> TypeIs[JoinStrategy]:
    return obj in {"inner", "left", "full", "cross", "anti", "semi"}


def _is_unique_keep_strategy(obj: Any) -> TypeIs[UniqueKeepStrategy]:
    return obj in {"any", "first", "last", "none"}


def _is_join_asof_strategy(obj: Any) -> TypeIs[AsofJoinStrategy]:
    return obj in {"backward", "forward", "nearest"}


def _validate_join_strategy(how: str, /) -> JoinStrategy:
    if _is_join_strategy(how):
        return how
    msg = f"Only the following join strategies are supported: {get_args(JoinStrategy)}; found '{how}'."
    raise NotImplementedError(msg)


def _validate_join_asof_strategy(strategy: str, /) -> AsofJoinStrategy:
    if _is_join_asof_strategy(strategy):
        return strategy
    msg = f"Only the following join strategies are supported: {get_args(AsofJoinStrategy)}; found '{strategy}'."
    raise NotImplementedError(msg)


def _validate_unique_keep_strategy(keep: str, /) -> UniqueKeepStrategy:
    if _is_unique_keep_strategy(keep):
        return keep
    msg = f"Only the following keep strategies are supported: {get_args(UniqueKeepStrategy)}; found '{keep}'."
    raise NotImplementedError(msg)


def normalize_join_on(
    on: OneOrIterable[str] | None,
    how: NonCrossJoinStrategy,
    left_on: OneOrIterable[str] | None,
    right_on: OneOrIterable[str] | None,
    /,
) -> tuple[Seq[str], Seq[str]]:
    """Reduce the 3 potential key (`*on`) arguments to 2.

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


def normalize_join_asof_on(
    left_on: str | None, right_on: str | None, on: str | None
) -> tuple[str, str]:
    """Reduce the 3 potential `join_asof` (`*on`) arguments to 2."""
    if on is None:
        if left_on is None or right_on is None:
            msg = "Either (`left_on` and `right_on`) or `on` keys should be specified."
            raise ValueError(msg)
        return left_on, right_on
    if left_on is not None or right_on is not None:
        msg = "If `on` is specified, `left_on` and `right_on` should be None."
        raise ValueError(msg)
    return on, on


def normalize_join_asof_by(
    by_left: str | Sequence[str] | None,
    by_right: str | Sequence[str] | None,
    by: str | Sequence[str] | None,
) -> tuple[Seq[str], Seq[str]]:
    """Reduce the 3 potential `join_asof` (`by*`) arguments to 2."""
    if by is None:
        if by_left and by_right:
            left_by = ensure_seq_str(by_left)
            right_by = ensure_seq_str(by_right)
            if len(left_by) != len(right_by):
                msg = "`by_left` and `by_right` must have the same length."
                raise ValueError(msg)
            return left_by, right_by
        msg = "Can not specify only `by_left` or `by_right`, you need to specify both."
        raise ValueError(msg)
    if by_left or by_right:
        msg = "If `by` is specified, `by_left` and `by_right` should be None."
        raise ValueError(msg)
    by_ = ensure_seq_str(by)  # pragma: no cover
    return by_, by_  # pragma: no cover


def normalize_pivot_args(
    on: OneOrIterable[str],
    *,
    index: OneOrIterable[str] | None,
    values: OneOrIterable[str] | None,
    frame_columns: list[str],
) -> tuple[Seq[str], Seq[str], Seq[str]]:
    """Derive a pivot specification from optional arguments.

    Returns in the order:

        (on, index, values)
    """
    columns = frame_columns
    on_ = ensure_seq_str(on)
    if not on_:
        msg = "`pivot` called without `on` columns."
        raise InvalidOperationError(msg)
    if index is None:
        if values is None:
            msg = "At least one of `values` and `index` must be passed"
            raise ValueError(msg)
        values_ = ensure_seq_str(values)
        index_ = tuple(
            nm for nm in columns if nm in set(columns).difference(on_, values_)
        )
    elif values is None:
        index_ = ensure_seq_str(index)
        values_ = tuple(
            nm for nm in columns if nm in set(columns).difference(on_, index_)
        )
    else:
        index_ = ensure_seq_str(index)
        values_ = ensure_seq_str(values)
    return on_, index_, values_
