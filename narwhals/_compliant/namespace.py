from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, cast, overload

from narwhals._compliant.typing import (
    CompliantExprT,
    CompliantFrameT,
    CompliantLazyFrameT,
    DepthTrackingExprT,
    EagerDataFrameT,
    EagerExprT,
    EagerSeriesT_co,
    LazyExprT,
    NativeFrameT,
    NativeFrameT_co,
    NativeSeriesT,
)
from narwhals._expression_parsing import is_expr
from narwhals._utils import (
    exclude_column_names,
    get_column_names,
    is_compliant_expr,
    passthrough_column_names,
)
from narwhals.dependencies import is_numpy_array_2d

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from typing_extensions import TypeAlias

    from narwhals._compliant.selectors import CompliantSelectorNamespace
    from narwhals._utils import Implementation, Version
    from narwhals.expr import Expr
    from narwhals.typing import (
        ConcatMethod,
        Into1DArray,
        IntoDType,
        IntoSchema,
        NonNestedLiteral,
        _2DArray,
    )

    Incomplete: TypeAlias = Any

__all__ = [
    "CompliantNamespace",
    "DepthTrackingNamespace",
    "EagerNamespace",
    "LazyNamespace",
]


class CompliantNamespace(Protocol[CompliantFrameT, CompliantExprT]):
    # NOTE: `narwhals`
    _implementation: Implementation
    _version: Version

    @property
    def _expr(self) -> type[CompliantExprT]: ...
    def evaluate_expr(self, data: Expr | NonNestedLiteral, /) -> CompliantExprT:
        if is_expr(data):
            ret = data(self)
        else:
            from narwhals.functions import lit

            ret = lit(data)(self)
        return cast("CompliantExprT", ret)

    # NOTE: `polars`
    def all(self) -> CompliantExprT:
        return self._expr.from_column_names(get_column_names, context=self)

    def col(self, *names: str) -> CompliantExprT:
        return self._expr.from_column_names(passthrough_column_names(names), context=self)

    def exclude(self, *names: str) -> CompliantExprT:
        return self._expr.from_column_names(
            partial(exclude_column_names, names=names), context=self
        )

    def nth(self, indices: Sequence[int]) -> CompliantExprT:
        return self._expr.from_column_indices(*indices, context=self)

    def len(self) -> CompliantExprT: ...
    def lit(self, value: NonNestedLiteral, dtype: IntoDType | None) -> CompliantExprT: ...
    def all_horizontal(
        self, *exprs: CompliantExprT, ignore_nulls: bool
    ) -> CompliantExprT: ...
    def any_horizontal(
        self, *exprs: CompliantExprT, ignore_nulls: bool
    ) -> CompliantExprT: ...
    def sum_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def mean_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def min_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def max_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def concat(
        self, items: Iterable[CompliantFrameT], *, how: ConcatMethod
    ) -> CompliantFrameT: ...
    def concat_str(
        self, *exprs: CompliantExprT, separator: str, ignore_nulls: bool
    ) -> CompliantExprT: ...
    @property
    def selectors(self) -> CompliantSelectorNamespace[Any, Any]: ...
    def coalesce(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    # NOTE: typing this accurately requires 2x more `TypeVar`s
    def from_native(self, data: Any, /) -> Any: ...


class DepthTrackingNamespace(
    CompliantNamespace[CompliantFrameT, DepthTrackingExprT],
    Protocol[CompliantFrameT, DepthTrackingExprT],
):
    def all(self) -> DepthTrackingExprT:
        return self._expr.from_column_names(get_column_names, context=self)

    def col(self, *names: str) -> DepthTrackingExprT:
        return self._expr.from_column_names(passthrough_column_names(names), context=self)

    def exclude(self, *names: str) -> DepthTrackingExprT:
        return self._expr.from_column_names(
            partial(exclude_column_names, names=names), context=self
        )


class LazyNamespace(
    CompliantNamespace[CompliantLazyFrameT, LazyExprT],
    Protocol[CompliantLazyFrameT, LazyExprT, NativeFrameT_co],
):
    @property
    def _backend_version(self) -> tuple[int, ...]:
        return self._implementation._backend_version()

    @property
    def _lazyframe(self) -> type[CompliantLazyFrameT]: ...

    def from_native(self, data: NativeFrameT_co | Any, /) -> CompliantLazyFrameT:
        if self._lazyframe._is_native(data):
            return self._lazyframe.from_native(data, context=self)
        msg = f"Unsupported type: {type(data).__name__!r}"  # pragma: no cover
        raise TypeError(msg)


class EagerNamespace(
    DepthTrackingNamespace[EagerDataFrameT, EagerExprT],
    Protocol[EagerDataFrameT, EagerSeriesT_co, EagerExprT, NativeFrameT, NativeSeriesT],
):
    @property
    def _backend_version(self) -> tuple[int, ...]:
        return self._implementation._backend_version()

    @property
    def _dataframe(self) -> type[EagerDataFrameT]: ...
    @property
    def _series(self) -> type[EagerSeriesT_co]: ...
    def _if_then_else(
        self,
        when: NativeSeriesT,
        then: NativeSeriesT,
        otherwise: NativeSeriesT | None = None,
    ) -> NativeSeriesT: ...
    def when_then(
        self,
        predicate: EagerExprT,
        then: EagerExprT | NonNestedLiteral,
        otherwise: EagerExprT | NonNestedLiteral | None = None,
    ) -> EagerExprT:
        def func(df: EagerDataFrameT) -> Sequence[EagerSeriesT_co]:
            predicate_s = df._evaluate_expr(predicate)
            align = predicate_s._align_full_broadcast

            if is_compliant_expr(then):
                then_s = df._evaluate_expr(then)
            else:
                then_s = predicate_s._from_scalar(then).alias("literal")
                then_s._broadcast = True
            if otherwise is None:
                predicate_s, then_s = align(predicate_s, then_s)
                result = self._if_then_else(predicate_s.native, then_s.native)

            if is_compliant_expr(otherwise):
                otherwise_s = df._evaluate_expr(otherwise)
            elif otherwise is not None:
                otherwise_s = predicate_s._from_scalar(otherwise).alias("literal")
                otherwise_s._broadcast = True

            if otherwise is None:
                predicate_s, then_s = align(predicate_s, then_s)
                result = self._if_then_else(predicate_s.native, then_s.native)
            else:
                predicate_s, then_s, otherwise_s = align(predicate_s, then_s, otherwise_s)
                result = self._if_then_else(
                    predicate_s.native, then_s.native, otherwise_s.native
                )
            return [then_s._with_native(result)]

        return self._expr._from_callable(
            func=func,
            evaluate_output_names=getattr(
                then, "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(then, "_alias_output_names", None),
            context=predicate,
        )

    @overload
    def from_native(self, data: NativeFrameT, /) -> EagerDataFrameT: ...
    @overload
    def from_native(self, data: NativeSeriesT, /) -> EagerSeriesT_co: ...
    def from_native(
        self, data: NativeFrameT | NativeSeriesT | Any, /
    ) -> EagerDataFrameT | EagerSeriesT_co:
        if self._dataframe._is_native(data):
            return self._dataframe.from_native(data, context=self)
        if self._series._is_native(data):
            return self._series.from_native(data, context=self)
        msg = f"Unsupported type: {type(data).__name__!r}"
        raise TypeError(msg)

    @overload
    def from_numpy(self, data: Into1DArray, /, schema: None = ...) -> EagerSeriesT_co: ...

    @overload
    def from_numpy(
        self, data: _2DArray, /, schema: IntoSchema | Sequence[str] | None
    ) -> EagerDataFrameT: ...

    def from_numpy(
        self,
        data: Into1DArray | _2DArray,
        /,
        schema: IntoSchema | Sequence[str] | None = None,
    ) -> EagerDataFrameT | EagerSeriesT_co:
        if is_numpy_array_2d(data):
            return self._dataframe.from_numpy(data, schema=schema, context=self)
        return self._series.from_numpy(data, context=self)

    def _concat_diagonal(self, dfs: Sequence[NativeFrameT], /) -> NativeFrameT: ...
    def _concat_horizontal(
        self, dfs: Sequence[NativeFrameT | Any], /
    ) -> NativeFrameT: ...
    def _concat_vertical(self, dfs: Sequence[NativeFrameT], /) -> NativeFrameT: ...
    def concat(
        self, items: Iterable[EagerDataFrameT], *, how: ConcatMethod
    ) -> EagerDataFrameT:
        dfs = [item.native for item in items]
        if how == "horizontal":
            native = self._concat_horizontal(dfs)
        elif how == "vertical":
            native = self._concat_vertical(dfs)
        elif how == "diagonal":
            native = self._concat_diagonal(dfs)
        else:  # pragma: no cover
            raise NotImplementedError
        return self._dataframe.from_native(native, context=self)
