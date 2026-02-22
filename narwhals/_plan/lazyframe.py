from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from narwhals._plan import _parse
from narwhals._plan.common import todo
from narwhals._plan.compliant.typing import Native
from narwhals._plan.group_by import LazyGroupBy
from narwhals._plan.options import (
    ExplodeOptions,
    JoinAsofOptions,
    JoinOptions,
    SortMultipleOptions,
    UniqueOptions,
    UnpivotOptions,
)
from narwhals._utils import Implementation, not_implemented, qualified_type_name
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from io import BytesIO

    from typing_extensions import Self

    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.plans import LogicalPlan, logical as lp
    from narwhals._plan.series import Series
    from narwhals._plan.typing import (
        ColumnNameOrSelector,
        IntoExpr,
        IntoExprColumn,
        OneOrIterable,
    )
    from narwhals.typing import (
        AsofJoinStrategy,
        FileSource,
        JoinStrategy,
        PivotAgg,
        UniqueKeepStrategy,
    )


# TODO @dangotbanned: Figure out `from_native` + remove `self._compliant`
class LazyFrame(Generic[Native]):  # pragma: no cover
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

    _compliant: CompliantLazyFrame[Native]
    _plan: LogicalPlan
    _implementation: Implementation

    @property
    def implementation(self) -> Implementation:
        """The implementation that will execute the plan."""
        return self._implementation

    # TODO @dangotbanned: Propagate typing from `ScanLazyFrame`
    # TODO @dangotbanned: `@overload` matching for the other typing
    @classmethod
    def _from_lp_scan(
        cls, plan: lp.Scan, implementation: Implementation
    ) -> LazyFrame[Any]:
        obj = cls.__new__(cls)
        obj._plan = plan
        obj._implementation = implementation
        return obj

    def _with_lp(self, plan: lp.SingleInput | lp.MultipleInputs[Any], /) -> Self:
        tp = type(self)
        obj = tp.__new__(tp)
        obj._plan = plan
        obj._implementation = self._implementation
        return obj

    to_native = todo()
    columns = todo()
    schema = todo()
    collect_schema = todo()
    collect = not_implemented()  # depends on resolving everything else

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

    def explain(self) -> str:
        """Create a string representation of the query plan."""
        return self._plan.explain()

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
        opts = JoinOptions.parse(how, suffix)
        left_on, right_on = opts.normalize_on(on, left_on, right_on)
        return self._with_lp(left.join(right, left_on, right_on, opts))

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
        opts = JoinAsofOptions.parse(by_left, by_right, by, strategy, suffix)
        left_on_, right_on_ = opts.normalize_on(left_on, right_on, on)
        right = self._unwrap_plan(other)
        return self._with_lp(self._plan.join_asof(right, left_on_, right_on_, opts))

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
                on_columns,  # type: ignore[arg-type]
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
        opts = UniqueOptions.parse(keep, maintain_order=False)
        parse = _parse.parse_into_seq_of_selector_ir
        s_subset = None if subset is None else parse(subset)
        if order_by is not None:
            return self._with_lp(self._plan.unique_by(s_subset, parse(order_by), opts))
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
