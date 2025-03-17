from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Protocol

from narwhals._compliant.typing import NativeSeriesT_co
from narwhals._translate import FromIterable
from narwhals._translate import NumpyConvertible
from narwhals.utils import unstable

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._compliant.expr import CompliantExpr  # noqa: F401
    from narwhals._compliant.expr import EagerExpr
    from narwhals._compliant.namespace import CompliantNamespace  # noqa: F401
    from narwhals._compliant.namespace import EagerNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import Into1DArray
    from narwhals.typing import _1DArray  # noqa: F401
    from narwhals.utils import Implementation
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

__all__ = ["CompliantSeries", "EagerSeries"]


class CompliantSeries(
    NumpyConvertible["_1DArray", "Into1DArray"],
    FromIterable,
    Protocol[NativeSeriesT_co],
):
    @property
    def dtype(self) -> DType: ...
    @property
    def name(self) -> str: ...
    @property
    def native(self) -> NativeSeriesT_co: ...
    def __narwhals_series__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> Any: ...  # CompliantNamespace[Any, Self]: ...
    def __len__(self) -> int:
        return len(self.native)

    def _from_native_series(self, series: Any) -> Self: ...
    def _to_expr(self) -> Any: ...  # CompliantExpr[Any, Self]: ...
    @classmethod
    def from_numpy(cls, data: Into1DArray, /, *, context: _FullContext) -> Self: ...
    @classmethod
    def from_iterable(
        cls, data: Iterable[Any], /, *, context: _FullContext, name: str = ""
    ) -> Self: ...
    @unstable
    def _with_native(self, series: Any, /) -> Self:
        """Equivalent to `._from_native_series`, eventually replacing.

        - But can be the same method name for all protocols.
        - New `Compliant*`,
          - preserving all backend stuff
          - replacing only `.native`
        - Different to `.from_*` classmethods, which are usually called from outside the class
        """
        return self._from_native_series(series)

    def alias(self, name: str) -> Self: ...
    def cast(self, dtype: DType | type[DType]) -> Self: ...


class EagerSeries(CompliantSeries[NativeSeriesT_co], Protocol[NativeSeriesT_co]):
    _native_series: Any
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version
    _broadcast: bool

    def _from_scalar(self, value: Any) -> Self:
        return self.from_iterable([value], name=self.name, context=self)

    def __narwhals_namespace__(self) -> EagerNamespace[Any, Self, Any]: ...

    def _to_expr(self) -> EagerExpr[Any, Any]:
        return self.__narwhals_namespace__()._expr._from_series(self)  # type: ignore[no-any-return]
