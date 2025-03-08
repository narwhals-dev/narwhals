from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Protocol
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._compliant.expr import CompliantExpr  # noqa: F401
    from narwhals._compliant.expr import EagerExpr
    from narwhals._compliant.namespace import CompliantNamespace  # noqa: F401
    from narwhals._compliant.namespace import EagerNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import NativeSeries
    from narwhals.utils import Implementation
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

__all__ = ["CompliantSeries", "EagerSeries"]

NativeSeriesT_co = TypeVar("NativeSeriesT_co", bound="NativeSeries", covariant=True)


class CompliantSeries(Protocol):
    @property
    def dtype(self) -> DType: ...
    @property
    def name(self) -> str: ...
    def __narwhals_series__(self) -> CompliantSeries: ...
    def alias(self, name: str) -> Self: ...
    def __narwhals_namespace__(self) -> Any: ...  # CompliantNamespace[Any, Self]: ...
    def _to_expr(self) -> Any: ...  # CompliantExpr[Any, Self]: ...


class EagerSeries(CompliantSeries, Protocol[NativeSeriesT_co]):
    _native_series: Any
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version
    _broadcast: bool

    @property
    def native(self) -> NativeSeriesT_co: ...

    # NOTE: `ArrowSeries` needs to intercept `value` w/
    # if self._backend_version < (13,) and hasattr(value, "as_py"):
    #     value = value.as_py()  # noqa: ERA001
    def _from_scalar(self, value: Any) -> Self:
        return self._from_iterable([value], name=self.name, context=self)

    @classmethod
    def _from_iterable(
        cls: type[Self], data: Iterable[Any], name: str, *, context: _FullContext
    ) -> Self: ...

    def __narwhals_namespace__(self) -> EagerNamespace[Any, Self]: ...

    def _to_expr(self) -> EagerExpr[Any, Self]:
        return self.__narwhals_namespace__()._expr._from_series(self)

    # TODO @dangotbanned: replacing `Namespace._create_compliant_series``
    # - All usage within `*Expr.map_batches`
    #   - `PandasLikeExpr` uses that **once**
    #   - `ArrowExpr` uses **twice**
    # - `PandasLikeDataFrame.with_row_index` uses the wrapped `utils` function once
