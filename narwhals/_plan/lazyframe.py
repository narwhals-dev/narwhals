from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic

from narwhals._exceptions import issue_warning
from narwhals._plan import _parse, translate
from narwhals._plan._namespace import (
    eager_implementation,
    evaluator,
    known_implementation,
)
from narwhals._plan.common import closed_kwds
from narwhals._plan.compliant.typing import FromNative, Native
from narwhals._plan.group_by import LazyGroupBy
from narwhals._plan.options import (
    ExplodeOptions,
    JoinAsofOptions,
    JoinOptions,
    SortMultipleOptions,
    UniqueOptions,
    UnpivotOptions,
)
from narwhals._plan.plans.conversion import Resolver
from narwhals._utils import Implementation, Version, not_implemented, qualified_type_name
from narwhals.exceptions import InvalidOperationError, PerformanceWarning

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from io import BytesIO

    from typing_extensions import Self, TypeAlias

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
    from narwhals._typing import Arrow, Pandas, Polars
    from narwhals.schema import Schema
    from narwhals.typing import (
        AsofJoinStrategy,
        FileSource,
        IntoBackend,
        JoinStrategy,
        PivotAgg,
        UniqueKeepStrategy,
    )

Incomplete: TypeAlias = Any


# TODO @dangotbanned: Figure out `from_native` + remove `self._compliant`
class LazyFrame(Generic[Native]):
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

    _plan: LogicalPlan
    _implementation: Implementation
    _version: ClassVar[Version] = Version.MAIN

    _compliant: CompliantLazyFrame[Native]
    to_native = not_implemented()  # look into this *after* `collect`

    @property
    def version(self) -> Version:  # pragma: no cover
        return self._version

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

    @classmethod
    def from_native(
        cls: type[LazyFrame[Any]], native: FromNative, /
    ) -> LazyFrame[FromNative]:
        return (
            translate.from_native_lazyframe(native)
            .to_logical()
            .to_narwhals(version=cls._version)
        )

    def _unwrap_plan(self, other: Self | Any, /) -> LogicalPlan:  # pragma: no cover
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
        first, more = (columns[0], columns[1:]) if columns else ((), ())
        s_ir = _parse.into_selector_ir(first, more, require_all=strict)
        return self._with_lp(self._plan.drop(s_ir))

    def drop_nulls(
        self, subset: OneOrIterable[ColumnNameOrSelector] | None = None
    ) -> Self:  # pragma: no cover
        s_ir = None if subset is None else _parse.into_selector_ir(subset)
        return self._with_lp(self._plan.drop_nulls(s_ir))

    def explain(self) -> str:  # pragma: no cover
        """Create a string representation of the query plan."""
        return self._plan.explain()

    def explode(
        self,
        columns: OneOrIterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
        empty_as_null: bool = True,
        keep_nulls: bool = True,
    ) -> Self:  # pragma: no cover
        s_ir = _parse.into_selector_ir(columns, more_columns)
        options = ExplodeOptions(empty_as_null=empty_as_null, keep_nulls=keep_nulls)
        return self._with_lp(self._plan.explode(s_ir, options))

    def filter(
        self, *predicates: OneOrIterable[IntoExprColumn], **constraints: Any
    ) -> Self:  # pragma: no cover
        p = _parse.predicates_constraints_into_expr_ir(*predicates, **constraints)
        return self._with_lp(self._plan.filter(p))

    # TODO @dangotbanned: LazyFrame.group_by(drop_null_keys=True)
    def group_by(
        self,
        *by: OneOrIterable[IntoExpr],
        drop_null_keys: bool = False,
        **named_by: IntoExpr,
    ) -> LazyGroupBy[Self]:  # pragma: no cover
        return LazyGroupBy(self, *by, drop_null_keys=drop_null_keys, **named_by)

    def head(self, n: int = 5) -> Self:  # pragma: no cover
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
    ) -> Self:  # pragma: no cover
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
    ) -> Self:  # pragma: no cover
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
    ) -> Self:  # pragma: no cover
        from narwhals._plan import selectors as cs

        on_ = _parse.into_selector_ir(on)
        if index is None:
            if values is None:
                msg = "`pivot` needs either `index or `values` needs to be specified"
                raise InvalidOperationError(msg)
            values_ = _parse.into_selector_ir(values)
            index_ = (cs.all() - on_.to_narwhals() - values_.to_narwhals())._ir
        else:
            index_ = _parse.into_selector_ir(index)
            if values is not None:
                values_ = _parse.into_selector_ir(values)
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
        e_irs = tuple(_parse.into_iter_expr_ir(*exprs, **named_exprs))
        return self._with_lp(self._plan.select(e_irs))

    # TODO @dangotbanned: Open an issue to find out why we don't have this on main?
    @property
    def columns(self) -> list[str]:  # pragma: no cover
        if self._version is not Version.V1:
            issue_warning(
                "Determining the column names of a LazyFrame requires resolving its schema,"
                " which is a potentially expensive operation. Use `LazyFrame.collect_schema().names()`"
                " to get the column names without this warning.",
                category=PerformanceWarning,
            )
        return self.collect_schema().names()

    @property
    def schema(self) -> Schema:  # pragma: no cover
        if self._version is not Version.V1:
            msg = (
                "Resolving the schema of a LazyFrame is a potentially expensive operation. "
                "Use `LazyFrame.collect_schema()` to get the schema without this warning."
            )
            issue_warning(msg, PerformanceWarning)
        return self.collect_schema()

    def collect_schema(self) -> Schema:
        """Resolve the schema of this LazyFrame."""
        return (
            Resolver.from_backend(self.implementation)
            .collect_schema(self._plan)
            .to_narwhals(self._version)
        )

    def collect(
        self, backend: IntoBackend[Polars | Pandas | Arrow] | None = None, **kwds: Any
    ) -> DataFrame[Any]:  # pragma: no cover
        """Materialize this LazyFrame into a DataFrame."""
        lazy = known_implementation(self.implementation)
        eager = eager_implementation(backend) if backend else None
        logical = self._plan.collect(closed_kwds(**kwds))
        resolved = Resolver.from_backend(lazy).collect(logical)
        return evaluator(lazy).collect(resolved, eager, self.version).to_narwhals()

    def sink_parquet(self, file: FileSource | BytesIO) -> None:
        lazy = known_implementation(self.implementation)
        logical = self._plan.sink_parquet(file)
        resolved = Resolver.from_backend(lazy).sink_parquet(logical)
        evaluator(lazy).sink_parquet(resolved, self.version)

    def sort(
        self,
        by: OneOrIterable[ColumnNameOrSelector],
        *more_by: ColumnNameOrSelector,
        descending: OneOrIterable[bool] = False,
        nulls_last: OneOrIterable[bool] = False,
    ) -> Self:
        s_irs = tuple(_parse.into_iter_selector_ir(by, more_by))
        opts = SortMultipleOptions.parse(descending=descending, nulls_last=nulls_last)
        return self._with_lp(self._plan.sort(s_irs, opts))

    def tail(self, n: int = 5) -> Self:  # pragma: no cover
        return self._with_lp(self._plan.tail(n))

    def unique(
        self,
        subset: OneOrIterable[ColumnNameOrSelector] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        order_by: OneOrIterable[ColumnNameOrSelector] | None = None,
    ) -> Self:  # pragma: no cover
        opts = UniqueOptions.parse(keep, maintain_order=False)
        s_subset = None if subset is None else tuple(_parse.into_iter_selector_ir(subset))
        if order_by is not None:
            by = tuple(_parse.into_iter_selector_ir(order_by))
            return self._with_lp(self._plan.unique_by(s_subset, by, opts))
        if keep in {"first", "last"}:
            msg = "'first' and 'last' are only supported if `order_by` is passed."
            raise InvalidOperationError(msg)
        return self._with_lp(self._plan.unique(s_subset, opts))

    def unnest(
        self,
        columns: OneOrIterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> Self:  # pragma: no cover
        s_ir = _parse.into_selector_ir(columns, more_columns)
        return self._with_lp(self._plan.unnest(s_ir))

    def unpivot(
        self,
        on: OneOrIterable[ColumnNameOrSelector] | None = None,
        *,
        index: OneOrIterable[ColumnNameOrSelector] | None = None,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> Self:
        s_on = on if on is None else _parse.into_selector_ir(on)
        s_index = None if index is None else _parse.into_selector_ir(index)
        options = UnpivotOptions(variable_name=variable_name, value_name=value_name)
        return self._with_lp(self._plan.unpivot(s_on, index=s_index, options=options))

    def with_columns(self, *exprs: OneOrIterable[IntoExpr], **named_exprs: Any) -> Self:
        e_irs = tuple(_parse.into_iter_expr_ir(*exprs, **named_exprs))
        return self._with_lp(self._plan.with_columns(e_irs))

    def with_row_index(
        self, name: str = "index", *, order_by: OneOrIterable[ColumnNameOrSelector]
    ) -> Self:  # pragma: no cover
        by = tuple(_parse.into_iter_selector_ir(order_by))
        return self._with_lp(self._plan.with_row_index_by(name, order_by=by))
