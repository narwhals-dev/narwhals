from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

from narwhals._compliant.expr import CompliantExpr
from narwhals._compliant.typing import (
    CompliantExprAny,
    CompliantFrameAny,
    CompliantLazyFrameT,
    CompliantSeriesOrNativeExprAny,
    EagerDataFrameT,
    EagerExprT,
    EagerSeriesT,
    LazyExprAny,
    NativeExprT,
    NativeSeriesT,
    WindowFunction,
)
from narwhals._typing_compat import Protocol38

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self, TypeAlias

    from narwhals._compliant.typing import EvalSeries, ScalarKwargs
    from narwhals._compliant.window import WindowInputs
    from narwhals._utils import Implementation, Version, _LimitedContext
    from narwhals.typing import NonNestedLiteral


__all__ = ["CompliantThen", "CompliantWhen", "EagerWhen", "LazyThen", "LazyWhen"]

ExprT = TypeVar("ExprT", bound=CompliantExprAny)
LazyExprT = TypeVar("LazyExprT", bound=LazyExprAny)
SeriesT = TypeVar("SeriesT", bound=CompliantSeriesOrNativeExprAny)
FrameT = TypeVar("FrameT", bound=CompliantFrameAny)

Scalar: TypeAlias = Any
"""A native literal value."""

IntoExpr: TypeAlias = "SeriesT | ExprT | NonNestedLiteral | Scalar"
"""Anything that is convertible into a `CompliantExpr`."""


class CompliantWhen(Protocol38[FrameT, SeriesT, ExprT]):
    _condition: ExprT
    _then_value: IntoExpr[SeriesT, ExprT]
    _otherwise_value: IntoExpr[SeriesT, ExprT] | None
    _implementation: Implementation
    _version: Version

    @property
    def _then(self) -> type[CompliantThen[FrameT, SeriesT, ExprT]]: ...
    def __call__(self, compliant_frame: FrameT, /) -> Sequence[SeriesT]: ...
    def _window_function(
        self, compliant_frame: FrameT, window_inputs: WindowInputs[Any]
    ) -> Sequence[SeriesT]: ...

    def then(
        self, value: IntoExpr[SeriesT, ExprT], /
    ) -> CompliantThen[FrameT, SeriesT, ExprT]:
        return self._then.from_when(self, value)

    @classmethod
    def from_expr(cls, condition: ExprT, /, *, context: _LimitedContext) -> Self:
        obj = cls.__new__(cls)
        obj._condition = condition
        obj._then_value = None
        obj._otherwise_value = None
        obj._implementation = context._implementation
        obj._version = context._version
        return obj


class CompliantThen(CompliantExpr[FrameT, SeriesT], Protocol38[FrameT, SeriesT, ExprT]):
    _call: EvalSeries[FrameT, SeriesT]
    _when_value: CompliantWhen[FrameT, SeriesT, ExprT]
    _function_name: str
    _depth: int
    _implementation: Implementation
    _version: Version
    _scalar_kwargs: ScalarKwargs

    @classmethod
    def from_when(
        cls,
        when: CompliantWhen[FrameT, SeriesT, ExprT],
        then: IntoExpr[SeriesT, ExprT],
        /,
    ) -> Self:
        when._then_value = then
        obj = cls.__new__(cls)
        obj._call = when
        obj._when_value = when
        obj._depth = 0
        obj._function_name = "whenthen"
        obj._evaluate_output_names = getattr(
            then, "_evaluate_output_names", lambda _df: ["literal"]
        )
        obj._alias_output_names = getattr(then, "_alias_output_names", None)
        obj._implementation = when._implementation
        obj._version = when._version
        obj._scalar_kwargs = {}
        return obj

    def otherwise(self, otherwise: IntoExpr[SeriesT, ExprT], /) -> ExprT:
        self._when_value._otherwise_value = otherwise
        self._function_name = "whenotherwise"
        return cast("ExprT", self)


class LazyThen(
    CompliantThen[CompliantLazyFrameT, NativeExprT, LazyExprT],
    Protocol38[CompliantLazyFrameT, NativeExprT, LazyExprT],
):
    _window_function: WindowFunction[CompliantLazyFrameT, NativeExprT] | None

    @classmethod
    def from_when(
        cls,
        when: CompliantWhen[CompliantLazyFrameT, NativeExprT, LazyExprT],
        then: IntoExpr[NativeExprT, LazyExprT],
        /,
    ) -> Self:
        when._then_value = then
        obj = cls.__new__(cls)
        obj._call = when
        obj._window_function = when._window_function
        obj._when_value = when
        obj._depth = 0
        obj._function_name = "whenthen"
        obj._evaluate_output_names = getattr(
            then, "_evaluate_output_names", lambda _df: ["literal"]
        )
        obj._alias_output_names = getattr(then, "_alias_output_names", None)
        obj._implementation = when._implementation
        obj._version = when._version
        obj._scalar_kwargs = {}
        return obj


class EagerWhen(
    CompliantWhen[EagerDataFrameT, EagerSeriesT, EagerExprT],
    Protocol38[EagerDataFrameT, EagerSeriesT, EagerExprT, NativeSeriesT],
):
    def _if_then_else(
        self,
        when: NativeSeriesT,
        then: NativeSeriesT,
        otherwise: NativeSeriesT | NonNestedLiteral | Scalar,
        /,
    ) -> NativeSeriesT: ...

    def __call__(self, df: EagerDataFrameT, /) -> Sequence[EagerSeriesT]:
        is_expr = self._condition._is_expr
        when: EagerSeriesT = self._condition(df)[0]
        then: EagerSeriesT
        align = when._align_full_broadcast

        if is_expr(self._then_value):
            then = self._then_value(df)[0]
        else:
            then = when.alias("literal")._from_scalar(self._then_value)
            then._broadcast = True

        if is_expr(self._otherwise_value):
            otherwise = self._otherwise_value(df)[0]
            when, then, otherwise = align(when, then, otherwise)
            result = self._if_then_else(when.native, then.native, otherwise.native)
        else:
            when, then = align(when, then)
            result = self._if_then_else(when.native, then.native, self._otherwise_value)
        return [then._with_native(result)]


class LazyWhen(
    CompliantWhen[CompliantLazyFrameT, NativeExprT, LazyExprT],
    Protocol38[CompliantLazyFrameT, NativeExprT, LazyExprT],
):
    when: Callable[..., NativeExprT]
    lit: Callable[..., NativeExprT]

    def __call__(self, df: CompliantLazyFrameT) -> Sequence[NativeExprT]:
        is_expr = self._condition._is_expr
        when = self.when
        lit = self.lit
        condition = df._evaluate_expr(self._condition)
        then_ = self._then_value
        then = df._evaluate_expr(then_) if is_expr(then_) else lit(then_)
        other_ = self._otherwise_value
        if other_ is None:
            result = when(condition, then)
        else:
            otherwise = df._evaluate_expr(other_) if is_expr(other_) else lit(other_)
            result = when(condition, then).otherwise(otherwise)  # type: ignore  # noqa: PGH003
        return [result]

    @classmethod
    def from_expr(cls, condition: LazyExprT, /, *, context: _LimitedContext) -> Self:
        obj = cls.__new__(cls)
        obj._condition = condition
        obj._then_value = None
        obj._otherwise_value = None
        obj._implementation = context._implementation
        obj._version = context._version
        return obj

    def _window_function(
        self, df: CompliantLazyFrameT, window_inputs: WindowInputs[NativeExprT]
    ) -> Sequence[NativeExprT]:
        is_expr = self._condition._is_expr
        condition = self._condition.window_function(df, window_inputs)[0]
        then_ = self._then_value
        then = (
            then_.window_function(df, window_inputs)[0]
            if is_expr(then_)
            else self.lit(then_)
        )

        other_ = self._otherwise_value
        if other_ is None:
            result = self.when(condition, then)
        else:
            other = (
                other_.window_function(df, window_inputs)[0]
                if is_expr(other_)
                else self.lit(other_)
            )
            result = self.when(condition, then).otherwise(other)  # type: ignore  # noqa: PGH003
        return [result]
