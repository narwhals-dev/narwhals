from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from typing import Any
from typing import Container
from typing import Iterable
from typing import Mapping
from typing import Protocol
from typing import Sequence
from typing import overload

from narwhals._compliant.typing import CompliantExprT
from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import CompliantLazyFrameT
from narwhals._compliant.typing import DepthTrackingExprT
from narwhals._compliant.typing import EagerDataFrameT
from narwhals._compliant.typing import EagerExprT
from narwhals._compliant.typing import EagerSeriesT
from narwhals._compliant.typing import LazyExprT
from narwhals._compliant.typing import NativeFrameT
from narwhals._compliant.typing import NativeFrameT_co
from narwhals._compliant.typing import NativeSeriesT
from narwhals.dependencies import is_numpy_array_2d
from narwhals.utils import exclude_column_names
from narwhals.utils import get_column_names
from narwhals.utils import passthrough_column_names

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._compliant.selectors import CompliantSelectorNamespace
    from narwhals._compliant.when_then import CompliantWhen
    from narwhals._compliant.when_then import EagerWhen
    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.typing import ConcatMethod
    from narwhals.typing import Into1DArray
    from narwhals.typing import NonNestedLiteral
    from narwhals.typing import _2DArray
    from narwhals.utils import Implementation
    from narwhals.utils import Version

    Incomplete: TypeAlias = Any

__all__ = ["CompliantNamespace", "EagerNamespace"]


class CompliantNamespace(Protocol[CompliantFrameT, CompliantExprT]):
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    def all(self) -> CompliantExprT:
        return self._expr.from_column_names(get_column_names, context=self)

    def col(self, *column_names: str) -> CompliantExprT:
        return self._expr.from_column_names(
            passthrough_column_names(column_names), context=self
        )

    def exclude(self, excluded_names: Container[str]) -> CompliantExprT:
        return self._expr.from_column_names(
            partial(exclude_column_names, names=excluded_names), context=self
        )

    def nth(self, *column_indices: int) -> CompliantExprT:
        return self._expr.from_column_indices(*column_indices, context=self)

    def len(self) -> CompliantExprT: ...
    def lit(
        self, value: NonNestedLiteral, dtype: DType | type[DType] | None
    ) -> CompliantExprT: ...
    def all_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def any_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def sum_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def mean_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def min_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def max_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def concat(
        self, items: Iterable[CompliantFrameT], *, how: ConcatMethod
    ) -> CompliantFrameT: ...
    def when(
        self, predicate: CompliantExprT
    ) -> CompliantWhen[CompliantFrameT, Incomplete, CompliantExprT]: ...
    def concat_str(
        self,
        *exprs: CompliantExprT,
        separator: str,
        ignore_nulls: bool,
    ) -> CompliantExprT: ...
    @property
    def selectors(self) -> CompliantSelectorNamespace[Any, Any]: ...
    @property
    def _expr(self) -> type[CompliantExprT]: ...


class DepthTrackingNamespace(
    CompliantNamespace[CompliantFrameT, DepthTrackingExprT],
    Protocol[CompliantFrameT, DepthTrackingExprT],
):
    def all(self) -> DepthTrackingExprT:
        return self._expr.from_column_names(
            get_column_names, function_name="all", context=self
        )

    def col(self, *column_names: str) -> DepthTrackingExprT:
        return self._expr.from_column_names(
            passthrough_column_names(column_names), function_name="col", context=self
        )

    def exclude(self, excluded_names: Container[str]) -> DepthTrackingExprT:
        return self._expr.from_column_names(
            partial(exclude_column_names, names=excluded_names),
            function_name="exclude",
            context=self,
        )


class LazyNamespace(
    CompliantNamespace[CompliantLazyFrameT, LazyExprT],
    Protocol[CompliantLazyFrameT, LazyExprT, NativeFrameT_co],
):
    @property
    def _lazyframe(self) -> type[CompliantLazyFrameT]: ...

    def from_native(self, data: NativeFrameT_co | Any, /) -> CompliantLazyFrameT:
        if self._lazyframe._is_native(data):
            return self._lazyframe.from_native(data, context=self)
        else:  # pragma: no cover
            msg = f"Unsupported type: {type(data).__name__!r}"
            raise TypeError(msg)


class EagerNamespace(
    DepthTrackingNamespace[EagerDataFrameT, EagerExprT],
    Protocol[EagerDataFrameT, EagerSeriesT, EagerExprT, NativeFrameT, NativeSeriesT],
):
    @property
    def _dataframe(self) -> type[EagerDataFrameT]: ...
    @property
    def _series(self) -> type[EagerSeriesT]: ...
    def when(
        self, predicate: EagerExprT
    ) -> EagerWhen[EagerDataFrameT, EagerSeriesT, EagerExprT, NativeSeriesT]: ...

    @overload
    def from_native(self, data: NativeFrameT, /) -> EagerDataFrameT: ...
    @overload
    def from_native(self, data: NativeSeriesT, /) -> EagerSeriesT: ...
    def from_native(
        self, data: NativeFrameT | NativeSeriesT | Any, /
    ) -> EagerDataFrameT | EagerSeriesT:
        if self._dataframe._is_native(data):
            return self._dataframe.from_native(data, context=self)
        elif self._series._is_native(data):
            return self._series.from_native(data, context=self)
        msg = f"Unsupported type: {type(data).__name__!r}"
        raise TypeError(msg)

    @overload
    def from_numpy(
        self,
        data: Into1DArray,
        /,
        schema: None = ...,
    ) -> EagerSeriesT: ...

    @overload
    def from_numpy(
        self,
        data: _2DArray,
        /,
        schema: Mapping[str, DType] | Schema | Sequence[str] | None,
    ) -> EagerDataFrameT: ...

    def from_numpy(
        self,
        data: Into1DArray | _2DArray,
        /,
        schema: Mapping[str, DType] | Schema | Sequence[str] | None = None,
    ) -> EagerDataFrameT | EagerSeriesT:
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
