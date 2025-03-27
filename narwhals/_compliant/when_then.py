from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence
from typing import TypeVar
from typing import cast

from narwhals._compliant.expr import CompliantExpr
from narwhals._compliant.typing import CompliantExprAny
from narwhals._compliant.typing import CompliantFrameAny
from narwhals._compliant.typing import CompliantLazyFrameT
from narwhals._compliant.typing import CompliantSeriesOrNativeExprAny
from narwhals._compliant.typing import EagerDataFrameT
from narwhals._compliant.typing import EagerExprT
from narwhals._compliant.typing import EagerSeriesT
from narwhals._compliant.typing import LazyExprAny
from narwhals._compliant.typing import NativeExprT
from narwhals._compliant.typing import NativeSeriesT

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing_extensions import TypeAlias

    from narwhals.utils import Implementation
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

if not TYPE_CHECKING:  # pragma: no cover
    if sys.version_info >= (3, 9):
        from typing import Protocol as Protocol38
    else:
        from typing import Generic as Protocol38
else:  # pragma: no cover
    # TODO @dangotbanned: Remove after dropping `3.8` (#2084)
    # - https://github.com/narwhals-dev/narwhals/pull/2064#discussion_r1965921386
    from typing import Protocol as Protocol38

__all__ = ["CompliantThen", "CompliantWhen", "EagerWhen", "LazyWhen"]

ExprT = TypeVar("ExprT", bound=CompliantExprAny)
LazyExprT = TypeVar("LazyExprT", bound=LazyExprAny)
SeriesT = TypeVar("SeriesT", bound=CompliantSeriesOrNativeExprAny)
FrameT = TypeVar("FrameT", bound=CompliantFrameAny)

Scalar: TypeAlias = Any
"""A native or python literal value."""

IntoExpr: TypeAlias = "SeriesT | ExprT | Scalar"
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
    _call: Callable[[FrameT], Sequence[SeriesT]]
    _when_value: CompliantWhen[FrameT, SeriesT, ExprT]
    _function_name: str
    _depth: int
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version
    _call_kwargs: dict[str, Any]

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
        obj._call_kwargs = {}
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
        otherwise: NativeSeriesT | Scalar | None,
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
        return [then._from_native_series(result)]


class LazyWhen(
    CompliantWhen[CompliantLazyFrameT, NativeExprT, LazyExprT],
    Protocol38[CompliantLazyFrameT, NativeExprT, LazyExprT],
):
    when: Callable[..., NativeExprT]
    lit: Callable[..., NativeExprT]

    def __call__(self: Self, df: CompliantLazyFrameT) -> Sequence[NativeExprT]:
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
