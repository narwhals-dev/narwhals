from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Protocol
from typing import TypeVar

from narwhals._translate import NumpyConvertible

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._compliant.expr import CompliantExpr  # noqa: F401
    from narwhals._compliant.expr import EagerExpr
    from narwhals._compliant.namespace import CompliantNamespace  # noqa: F401
    from narwhals._compliant.namespace import EagerNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import Into1DArray
    from narwhals.typing import NativeSeries
    from narwhals.typing import _1DArray  # noqa: F401
    from narwhals.utils import Implementation
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

__all__ = ["CompliantSeries", "EagerSeries"]

NativeSeriesT_co = TypeVar("NativeSeriesT_co", bound="NativeSeries", covariant=True)


class CompliantSeries(NumpyConvertible["_1DArray", "Into1DArray"], Protocol):
    @property
    def dtype(self) -> DType: ...
    @property
    def name(self) -> str: ...
    @property
    def native(self) -> Any: ...
    def __narwhals_series__(self) -> CompliantSeries: ...
    def alias(self, name: str) -> Self: ...
    def __narwhals_namespace__(self) -> Any: ...  # CompliantNamespace[Any, Self]: ...
    def _from_native_series(self, series: Any) -> Self: ...
    def _to_expr(self) -> Any: ...  # CompliantExpr[Any, Self]: ...
    @classmethod
    def from_numpy(cls, data: Into1DArray, /, *, context: _FullContext) -> Self: ...


class EagerSeries(CompliantSeries, Protocol[NativeSeriesT_co]):
    _native_series: Any
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version
    _broadcast: bool

    @property
    def native(self) -> NativeSeriesT_co: ...

    def _from_scalar(self, value: Any) -> Self:
        return self._from_iterable([value], name=self.name, context=self)

    @classmethod
    def _from_iterable(
        cls: type[Self], data: Iterable[Any], name: str, *, context: _FullContext
    ) -> Self: ...

    def __narwhals_namespace__(self) -> EagerNamespace[Any, Self, Any]: ...

    def _to_expr(self) -> EagerExpr[Any, Any]:
        return self.__narwhals_namespace__()._expr._from_series(self)  # type: ignore[no-any-return]

    def cast(self, dtype: DType | type[DType]) -> Self: ...
