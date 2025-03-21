from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence
from typing import cast

from narwhals._compliant.expr import CompliantExpr
from narwhals._compliant.typing import CompliantExprT
from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import CompliantSeriesOrNativeExprT_co

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


class CompliantWhen(
    Protocol38[CompliantFrameT, CompliantSeriesOrNativeExprT_co, CompliantExprT]
):
    _condition: CompliantExprT
    _then_value: CompliantExprT | CompliantSeriesOrNativeExprT_co | _Scalar
    _otherwise_value: CompliantExprT | CompliantSeriesOrNativeExprT_co | _Scalar
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    @property
    def _then(
        self,
    ) -> type[
        CompliantThen[CompliantFrameT, CompliantSeriesOrNativeExprT_co, CompliantExprT]
    ]: ...

    def __init__(self, condition: CompliantExprT, /, *, context: _FullContext) -> None:
        self._condition = condition
        self._then_value = None
        self._otherwise_value = None
        self._implementation = context._implementation
        self._backend_version = context._backend_version
        self._version = context._version

    def __call__(
        self, compliant_frame: CompliantFrameT, /
    ) -> Sequence[CompliantSeriesOrNativeExprT_co]: ...

    def then(
        self, value: CompliantExprT | CompliantSeriesOrNativeExprT_co | _Scalar, /
    ) -> CompliantThen[CompliantFrameT, CompliantSeriesOrNativeExprT_co, CompliantExprT]:
        return self._then.from_when(self, value)


# NOTE: error: Covariant type variable "CompliantSeriesOrNativeExprT_co" used in protocol where invariant one is expected [misc] (`mypy`)
# - May need to adjust later
class CompliantThen(  # type: ignore[misc]
    CompliantExpr[CompliantFrameT, CompliantSeriesOrNativeExprT_co],
    Protocol38[CompliantFrameT, CompliantSeriesOrNativeExprT_co, CompliantExprT],
):
    _call: CompliantWhen[CompliantFrameT, CompliantSeriesOrNativeExprT_co, CompliantExprT]
    _function_name: str
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version
    _call_kwargs: dict[str, Any]

    @classmethod
    def from_when(
        cls,
        when: CompliantWhen[
            CompliantFrameT, CompliantSeriesOrNativeExprT_co, CompliantExprT
        ],
        then_value: CompliantExprT | CompliantSeriesOrNativeExprT_co | _Scalar,
        /,
    ) -> Self:
        when._then_value = then_value
        obj = cls.__new__(cls)
        obj._call = when
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
        self, value: CompliantExprT | CompliantSeriesOrNativeExprT_co | _Scalar, /
    ) -> CompliantExprT:
        self._call._otherwise_value = value
        self._function_name = "whenotherwise"
        return cast("CompliantExprT", self)
