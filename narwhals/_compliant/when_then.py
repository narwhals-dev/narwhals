from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence
from typing import cast

from narwhals._compliant.expr import CompliantExpr
from narwhals._compliant.typing import CompliantExprT
from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import CompliantSeriesOrNativeExprT

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

_Scalar: TypeAlias = Any

__all__ = ["CompliantThen", "CompliantWhen"]


class CompliantWhen(
    Protocol38[CompliantFrameT, CompliantSeriesOrNativeExprT, CompliantExprT]
):
    _condition: CompliantExprT
    _then_value: CompliantExprT | CompliantSeriesOrNativeExprT | _Scalar
    _otherwise_value: CompliantExprT | CompliantSeriesOrNativeExprT | _Scalar
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    @property
    def _then(
        self,
    ) -> type[
        CompliantThen[CompliantFrameT, CompliantSeriesOrNativeExprT, CompliantExprT]
    ]: ...

    def __call__(
        self, compliant_frame: CompliantFrameT, /
    ) -> Sequence[CompliantSeriesOrNativeExprT]: ...

    def then(
        self, value: CompliantExprT | CompliantSeriesOrNativeExprT | _Scalar, /
    ) -> CompliantThen[CompliantFrameT, CompliantSeriesOrNativeExprT, CompliantExprT]:
        return self._then.from_when(self, value)

    def __init__(self, condition: CompliantExprT, /, *, context: _FullContext) -> None:
        self._condition = condition
        self._then_value = None
        self._otherwise_value = None
        self._implementation = context._implementation
        self._backend_version = context._backend_version
        self._version = context._version


class CompliantThen(
    CompliantExpr[CompliantFrameT, CompliantSeriesOrNativeExprT],
    Protocol38[CompliantFrameT, CompliantSeriesOrNativeExprT, CompliantExprT],
):
    _call: Callable[[CompliantFrameT], Sequence[CompliantSeriesOrNativeExprT]]
    _when: CompliantWhen[CompliantFrameT, CompliantSeriesOrNativeExprT, CompliantExprT]
    _function_name: str
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version
    _call_kwargs: dict[str, Any]

    @classmethod
    def from_when(
        cls,
        when: CompliantWhen[
            CompliantFrameT, CompliantSeriesOrNativeExprT, CompliantExprT
        ],
        then_value: CompliantExprT | CompliantSeriesOrNativeExprT | _Scalar,
        /,
    ) -> Self:
        when._then_value = then_value
        obj = cls.__new__(cls)
        obj._call = when
        obj._when = when
        obj._depth = 0
        obj._function_name = "whenthen"
        obj._evaluate_output_names = getattr(
            then_value, "_evaluate_output_names", lambda _df: ["literal"]
        )
        obj._alias_output_names = getattr(then_value, "_alias_output_names", None)
        obj._implementation = when._implementation
        obj._backend_version = when._backend_version
        obj._version = when._version
        obj._call_kwargs = {}
        return obj

    def otherwise(
        self, value: CompliantExprT | CompliantSeriesOrNativeExprT | _Scalar, /
    ) -> CompliantExprT:
        self._when._otherwise_value = value
        self._function_name = "whenotherwise"
        return cast("CompliantExprT", self)
