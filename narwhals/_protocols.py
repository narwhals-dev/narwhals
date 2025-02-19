"""Refinements/extensions on `Compliant*` protocols.

Mentioned in https://github.com/narwhals-dev/narwhals/issues/2044#issuecomment-2668381264

Notes:
- `Reuse*` are shared `pyarrow` & `pandas`
- No strong feelings on the name, borrowed from `reuse_series_(namespace_?)implementation`
- https://github.com/narwhals-dev/narwhals/blob/70220169458599013a06fbf3024effcb860d62ed/narwhals/_expression_parsing.py#L112
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generic
from typing import Protocol
from typing import Sequence
from typing import TypeVar

from narwhals.typing import CompliantExpr
from narwhals.typing import CompliantNamespace
from narwhals.typing import CompliantSeries

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Iterable
    from typing import Iterator

    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.typing import CompliantDataFrame
    from narwhals.typing import NativeSeries
    from narwhals.typing import _1DArray
    from narwhals.utils import Implementation
    from narwhals.utils import Version

NativeSeriesT_co = TypeVar("NativeSeriesT_co", bound="NativeSeries", covariant=True)
ReuseSeriesT = TypeVar("ReuseSeriesT", bound="ReuseSeries")
ReuseSeriesT_co = TypeVar("ReuseSeriesT_co", bound="ReuseSeries", covariant=True)

# NOTE: Haven't needed a `Reuse` so far
CompliantDataFrameT_co = TypeVar(
    "CompliantDataFrameT_co", bound="CompliantDataFrame", covariant=True
)


class ReuseExpr(CompliantExpr[ReuseSeriesT_co], Generic[ReuseSeriesT_co], Protocol):
    _call: Any
    _depth: int
    _function_name: str
    _evaluate_output_names: Any
    _alias_output_names: Any
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version
    _kwargs: dict[str, Any]

    def __init__(
        self: Self,
        call: Callable[[CompliantDataFrameT_co], Sequence[ReuseSeriesT_co]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[CompliantDataFrameT_co], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None: ...

    def __call__(self, df: Any) -> Sequence[ReuseSeriesT_co]:
        return self._call(df)


class ReuseSeries(CompliantSeries, Generic["NativeSeriesT_co"], Protocol):  # type: ignore[misc]
    _name: Any
    _native_series: NativeSeriesT_co
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    def __init__(
        self,
        native_series: NativeSeriesT_co,
        *,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None: ...

    def __array__(self: Self, dtype: Any, copy: bool | None) -> _1DArray: ...
    def __contains__(self: Self, other: Any) -> bool: ...
    def __iter__(self: Self) -> Iterator[Any]: ...
    def __len__(self: Self) -> int:
        return len(self._native_series)

    def __narwhals_series__(self: Self) -> Self:
        return self

    # NOTE: Each side adds an `AssertionError` guard first
    # Would probably make more sense to require a ClassVar, either:
    # - defining the set of permitted impls
    # - setting an unbound method, which also gives the error message
    #   - E.g. `PandasLikeSeries._ensure = Implementation.ensure_pandas_like`
    #   - That gets called w/ `ReuseSeries._ensure(ReuseSeries._implementation)`
    def __native_namespace__(self: Self) -> ModuleType:
        return self._implementation.to_native_namespace()

    @classmethod
    def _from_iterable(
        cls: type[Self],
        data: Iterable[Any],
        name: str,
        *,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self: ...
    def _from_native_series(self: Self, series: NativeSeriesT_co | Any) -> Self: ...
    @property
    def dtype(self: Self) -> DType: ...
    @property
    def name(self: Self) -> str:
        return self._name


class ReuseNamespace(CompliantNamespace[ReuseSeriesT], Protocol):
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    def __init__(
        self,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None: ...

    def __narwhals_expr__(self) -> Callable[..., ReuseExpr[ReuseSeriesT]]: ...

    # Both do very similar things:
    # - `_pandas_like.utils.create_compliant_series`
    # - `_arrow.series.ArrowSeries(native_series=pa.chunked_array([value]))`
    def _create_compliant_series(self, value: Any) -> ReuseSeriesT: ...

    # NOTE: Fully spec'd
    def _create_expr_from_callable(
        self,
        func: Callable[[CompliantDataFrameT_co], Sequence[ReuseSeriesT]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[CompliantDataFrameT_co], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        kwargs: dict[str, Any],
    ) -> CompliantExpr[ReuseSeriesT]:
        return self.__narwhals_expr__()(
            func,
            depth=depth,
            function_name=function_name,
            evaluate_output_names=evaluate_output_names,
            alias_output_names=alias_output_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
            kwargs=kwargs,
        )

    # NOTE: Fully spec'd
    def _create_expr_from_series(
        self, series: ReuseSeriesT
    ) -> CompliantExpr[ReuseSeriesT]:
        return self.__narwhals_expr__()(
            lambda _df: [series],
            depth=0,
            function_name="series",
            evaluate_output_names=lambda _df: [series.name],
            alias_output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    # TODO @dangotbanned: ReuseSeries (Protocol) needs to define _from_iterable
    def _create_series_from_scalar(
        self, value: Any, *, reference_series: ReuseSeriesT
    ) -> ReuseSeriesT: ...
