from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, get_args, overload

from narwhals._plan import _parse
from narwhals._plan._expansion import expand_selector_irs_names, prepare_projection
from narwhals._plan._guards import is_series
from narwhals._plan.common import ensure_seq_str, temp
from narwhals._plan.group_by import GroupBy, Grouped
from narwhals._plan.logical_plan import LpBuilder
from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
from narwhals._plan.series import Series
from narwhals._plan.typing import (
    ColumnNameOrSelector,
    IncompleteCyclic,
    IntoExpr,
    IntoExprColumn,
    NativeDataFrameT,
    NativeDataFrameT_co,
    NativeFrameT_co,
    NativeSeriesT,
    NativeSeriesT2,
    NonCrossJoinStrategy,
    OneOrIterable,
    PartialSeries,
    Seq,
)
from narwhals._utils import Implementation, Version, generate_repr
from narwhals.dependencies import is_pyarrow_table
from narwhals.exceptions import InvalidOperationError, ShapeError
from narwhals.schema import Schema
from narwhals.typing import EagerAllowed, IntoBackend, IntoDType, IntoSchema, JoinStrategy

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._native import NativeSeries
    from narwhals._plan.arrow.typing import NativeArrowDataFrame
    from narwhals._plan.compliant.dataframe import (
        CompliantDataFrame,
        CompliantFrame,
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

    def with_row_index(
        self, name: str = "index", *, order_by: OneOrIterable[ColumnNameOrSelector]
    ) -> Self:
        by_selectors = _parse.parse_into_seq_of_selector_ir(order_by)
        by_names = expand_selector_irs_names(by_selectors, schema=self, require_any=True)
        return self._with_compliant(self._compliant.with_row_index_by(name, by_names))

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
        dtypes = self.version.dtypes
        tp_list = dtypes.List
        for col_to_explode in subset:
            dtype = schema[col_to_explode]
            if dtype != tp_list:
                msg = f"`explode` operation is not supported for dtype `{dtype}`, expected List type"
                raise InvalidOperationError(msg)
        options = ExplodeOptions(empty_as_null=empty_as_null, keep_nulls=keep_nulls)
        return self._with_compliant(self._compliant.explode(subset, options))


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
    _compliant: CompliantDataFrame[IncompleteCyclic, NativeDataFrameT_co, NativeSeriesT]

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

    def with_row_index(
        self,
        name: str = "index",
        *,
        order_by: OneOrIterable[ColumnNameOrSelector] | None = None,
    ) -> Self:
        if order_by is None:
            return self._with_compliant(self._compliant.with_row_index(name))
        return super().with_row_index(name, order_by=order_by)

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
    def _to_lp_builder(self) -> LpBuilder:
        return LpBuilder.from_df(self)


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
