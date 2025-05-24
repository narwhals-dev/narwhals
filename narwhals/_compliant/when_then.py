from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, cast

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
)
from narwhals._typing_compat import Protocol38

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._compliant.typing import EvalSeries, ScalarKwargs
    from narwhals.typing import NonNestedLiteral
    from narwhals.utils import Implementation, Version, _FullContext


__all__ = ["CompliantThen", "CompliantWhen", "EagerWhen", "LazyWhen"]

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
    _otherwise_value: IntoExpr[SeriesT, ExprT]
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    @property
    def _then(self) -> type[CompliantThen[FrameT, SeriesT, ExprT]]: ...
    def __call__(self, compliant_frame: FrameT, /) -> Sequence[SeriesT]: ...

    def then(
        self, value: IntoExpr[SeriesT, ExprT], /
    ) -> CompliantThen[FrameT, SeriesT, ExprT]:
        return self._then.from_when(self, value)

    @classmethod
    def from_expr(cls, condition: ExprT, /, *, context: _FullContext) -> Self:
        obj = cls.__new__(cls)
        obj._condition = condition
        obj._then_value = None
        obj._otherwise_value = None
        obj._implementation = context._implementation
        obj._backend_version = context._backend_version
        obj._version = context._version
        return obj


class CompliantThen(CompliantExpr[FrameT, SeriesT], Protocol38[FrameT, SeriesT, ExprT]):
    _call: EvalSeries[FrameT, SeriesT]
    _when_value: CompliantWhen[FrameT, SeriesT, ExprT]
    _function_name: str
    _depth: int
    _implementation: Implementation
    _backend_version: tuple[int, ...]
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
        obj._backend_version = when._backend_version
        obj._version = when._version
        obj._scalar_kwargs = {}
        return obj

    def otherwise(self, otherwise: IntoExpr[SeriesT, ExprT], /) -> ExprT:
        self._when_value._otherwise_value = otherwise
        self._function_name = "whenotherwise"
        return cast("ExprT", self)


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
        if is_expr(self._then_value):
            then = self._then_value(df)[0]
        else:
            then = when.alias("literal")._from_scalar(self._then_value)
            then._broadcast = True
        if is_expr(self._otherwise_value):
            otherwise = df._extract_comparand(self._otherwise_value(df)[0])
        else:
            otherwise = self._otherwise_value
        result = self._if_then_else(when.native, df._extract_comparand(then), otherwise)
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
