from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, get_args, overload

from narwhals._plan import _parse, translate
from narwhals._plan._expansion import (
    expand_selector_irs_names,
    parse_expand_selectors,
    prepare_projection,
)
from narwhals._plan._guards import is_series
from narwhals._plan._namespace import eager_implementation, namespace_from_backend
from narwhals._plan.common import ensure_seq_str, normalize_target_file, temp
from narwhals._plan.compliant.translate import can_from_dict
from narwhals._plan.exceptions import unsupported_backend_operation_error
from narwhals._plan.group_by import GroupBy, Grouped
from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
from narwhals._plan.plans import LogicalPlan
from narwhals._plan.series import Series
from narwhals._plan.typing import (
    ColumnNameOrSelector,
    IntoExpr,
    IntoExprColumn,
    NativeDataFrameT,
    NativeDataFrameT_co,
    NativeFrameT_co,
    NativeSeriesT,
    NativeSeriesT_co,
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
    qualified_type_name,
)
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

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._native import NativeSeries
    from narwhals._plan.arrow.typing import NativeArrowDataFrame
    from narwhals._plan.compliant.dataframe import CompliantDataFrame, CompliantFrame
    from narwhals._plan.compliant.namespace import CompliantNamespace
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.lazyframe import LazyFrame
    from narwhals._plan.polars.typing import NativePolarsDataFrame
    from narwhals._typing import Arrow, Polars, _EagerAllowedImpl


Incomplete: TypeAlias = Any


class BaseFrame(Generic[NativeFrameT_co]):
    _compliant: CompliantFrame[NativeFrameT_co]
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

    def __init__(self, compliant: CompliantFrame[NativeFrameT_co], /) -> None:
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

    def _with_compliant(self, compliant: CompliantFrame[Incomplete], /) -> Self:
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
        named_irs, _ = prepare_projection(
            _parse.parse_into_seq_of_expr_ir(*exprs, **named_exprs), schema=self
        )
        return self._with_compliant(self._compliant.select(named_irs))

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
        names = parse_expand_selectors(by, more_by, schema=self)
        opts = SortMultipleOptions.parse(descending=descending, nulls_last=nulls_last)
        return self._with_compliant(self._compliant.sort(names, opts))

    def drop(
        self, *columns: OneOrIterable[ColumnNameOrSelector], strict: bool = True
    ) -> Self:
        s_ir = _parse.parse_into_combined_selector_ir(*columns, require_all=strict)
        if names := expand_selector_irs_names((s_ir,), schema=self, require_any=False):
            compliant = self._compliant.drop(names)
        else:
            compliant = self._compliant._with_native(self.to_native())
        return self._with_compliant(compliant)

    def drop_nulls(
        self, subset: OneOrIterable[ColumnNameOrSelector] | None = None
    ) -> Self:
        if subset is not None:
            subset = (
                parse_expand_selectors(subset, schema=self, require_any=False) or None
            )
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
        schema = self.collect_schema()
        if on is not None:
            s_irs = _parse.parse_into_seq_of_selector_ir(on)
            on_ = expand_selector_irs_names(s_irs, schema=schema)
        if index is not None:
            s_irs = _parse.parse_into_seq_of_selector_ir(index)
            index_ = expand_selector_irs_names(s_irs, schema=schema)
        return self._with_compliant(
            self._compliant.unpivot(
                on_, index_, variable_name=variable_name, value_name=value_name
            )
        )

    def with_row_index(
        self, name: str = "index", *, order_by: OneOrIterable[ColumnNameOrSelector]
    ) -> Self:
        by_names = parse_expand_selectors(order_by, schema=self)
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
        right: CompliantFrame[NativeFrameT_co] = self._unwrap_compliant(other)
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
        right: CompliantFrame[NativeFrameT_co] = self._unwrap_compliant(other)
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
        schema = self.collect_schema()
        subset = parse_expand_selectors(columns, more_columns, schema=schema)
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
        schema = self.collect_schema()
        subset = parse_expand_selectors(columns, more_columns, schema=schema)
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


class DataFrame(
    BaseFrame[NativeDataFrameT_co], Generic[NativeDataFrameT_co, NativeSeriesT_co]
):
    _compliant: CompliantDataFrame[NativeDataFrameT_co, NativeSeriesT_co]

    def __narwhals_namespace__(
        self,
    ) -> CompliantNamespace[
        CompliantDataFrame[NativeDataFrameT_co, NativeSeriesT_co], Any, Any
    ]:
        return self._compliant.__narwhals_namespace__()

    @property
    def implementation(self) -> _EagerAllowedImpl:
        return eager_implementation(self._compliant.implementation)

    @property
    def shape(self) -> tuple[int, int]:
        return self._compliant.shape

    def __len__(self) -> int:
        return len(self._compliant)

    @property
    def _series(self) -> type[Series[NativeSeriesT_co]]:
        return Series[NativeSeriesT_co]

    def _partial_series(
        self, *, dtype: IntoDType | None = None
    ) -> PartialSeries[NativeSeriesT_co]:
        it_names = temp.column_names(self.columns)
        backend = self.implementation
        series = self._series.from_iterable

        def fn(values: Iterable[Any], /) -> Series[NativeSeriesT_co]:
            return series(values, name=next(it_names), dtype=dtype, backend=backend)

        return fn

    def _parse_into_compliant_series(
        self, other: Series[Any] | Iterable[Any], /, name: str = ""
    ) -> CompliantSeries[NativeSeriesT_co]:
        if columns := self.columns:
            compliant = self.get_column(columns[0])._parse_into_compliant(other)
            return compliant if not name or compliant.name else compliant.alias(name)
        else:  # pragma: no cover # noqa: RET505
            backend = self.implementation
            series = self._series.from_iterable
            if not is_series(other):
                return series(other, name=name, backend=backend)._compliant
            s: CompliantSeries[Any] = other._compliant
            if s.implementation is backend:
                return s
            msg = f"Expected {backend!r}, got {s.implementation!r}"
            raise NotImplementedError(msg)

    @overload
    @classmethod
    def from_native(
        cls: type[DataFrame[Any, Any]], native: NativeArrowDataFrame, /
    ) -> DataFrame[pa.Table, pa.ChunkedArray[Any]]: ...
    @overload
    @classmethod
    def from_native(
        cls: type[DataFrame[Any, Any]], native: NativePolarsDataFrame, /
    ) -> DataFrame[pl.DataFrame, pl.Series]: ...
    @overload
    @classmethod
    def from_native(
        cls: type[DataFrame[Any, Any]], native: NativeDataFrameT, /
    ) -> DataFrame[NativeDataFrameT, Any]: ...
    @classmethod
    def from_native(
        cls: type[DataFrame[Any, Any]], native: NativeDataFrameT, /
    ) -> DataFrame[Any, Any]:
        return cls(translate.from_native_dataframe(native))

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
        backend: Polars,
    ) -> DataFrame[pl.DataFrame, pl.Series]: ...
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
        data: Mapping[str, Series[NativeSeriesT]],
        schema: IntoSchema | None = ...,
    ) -> DataFrame[Any, NativeSeriesT]: ...
    @classmethod
    def from_dict(
        cls: type[DataFrame[Any, Any]],
        data: Mapping[str, Any],
        schema: IntoSchema | None = None,
        *,
        backend: IntoBackend[EagerAllowed] | None = None,
    ) -> DataFrame[Any, Any]:
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
            backend = impl
            data = unwrapped
        ns = namespace_from_backend(backend)
        if can_from_dict(ns):
            return ns.from_dict(data, schema=schema, version=cls._version).to_narwhals()
        raise unsupported_backend_operation_error(
            backend, "from_dict"
        )  # pragma: no cover

    @overload
    def to_dict(
        self, *, as_series: Literal[True] = ...
    ) -> dict[str, Series[NativeSeriesT_co]]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, Series[NativeSeriesT_co]] | dict[str, list[Any]]: ...
    def to_dict(
        self, *, as_series: bool = True
    ) -> dict[str, Series[NativeSeriesT_co]] | dict[str, list[Any]]:
        if as_series:  # pragma: no cover
            return {
                key: self._series(value)
                for key, value in self._compliant.to_dict(as_series=as_series).items()
            }
        return self._compliant.to_dict(as_series=as_series)

    def to_series(self, index: int = 0) -> Series[NativeSeriesT_co]:
        return self._series(self._compliant.to_series(index))

    def to_struct(self, name: str = "") -> Series[NativeSeriesT_co]:
        return self._series(self._compliant.to_struct(name))

    def to_arrow(self) -> pa.Table:  # pragma: no cover
        return self._compliant.to_arrow()

    def to_pandas(self) -> pd.DataFrame:  # pragma: no cover
        return self._compliant.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        return self._compliant.to_polars()

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self._with_compliant(self._compliant.gather_every(n, offset))

    def get_column(self, name: str) -> Series[NativeSeriesT_co]:
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

    def iter_columns(self) -> Iterator[Series[NativeSeriesT_co]]:
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
        names = parse_expand_selectors(by, more_by, schema=self)
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
        on_, index_, values_ = normalize_pivot_args(
            on, index=index, values=values, frame_columns=self.columns
        )
        on_cols: CompliantDataFrame[NativeDataFrameT_co, NativeSeriesT_co]

        if on_columns is None:
            nw_on_cols = self.select(on_).unique(on_, maintain_order=True)
            if sort_columns:
                nw_on_cols = nw_on_cols.sort(on_)
            on_cols = nw_on_cols._compliant
        elif isinstance(on_columns, DataFrame):
            on_cols = on_columns._compliant
        else:
            on_cols = self._parse_into_compliant_series(on_columns, on_[0]).to_frame()

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
        schema = self.collect_schema()
        subset_names: Sequence[str] | None = None
        if subset is not None:
            subset_names = parse_expand_selectors(subset, schema=schema)
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
        by_names = parse_expand_selectors(order_by, schema=schema)
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

    # TODO @dangotbanned: Review this `LogicalPlan` entry point again
    # - Last `LogicalPlan` import left in `dataframe.py`
    # - `ScanDataFrame.to_narwhals` is still pretty messy and closed for extension
    def lazy(self, backend: IntoBackend[LazyAllowed] | None = None) -> LazyFrame[Any]:
        return LogicalPlan.from_df(self).to_narwhals(backend, self.version)


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


# TODO @dangotbanned: Remove after migrating `DataFrame.join_asof` to use options parsing
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
