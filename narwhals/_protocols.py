"""Refinements/extensions on `Compliant*` protocols.

Mentioned in https://github.com/narwhals-dev/narwhals/issues/2044#issuecomment-2668381264

Notes:
- `Reuse*` are shared `pyarrow` & `pandas`
- No strong feelings on the name, borrowed from `reuse_series_(namespace_?)implementation`
- https://github.com/narwhals-dev/narwhals/blob/70220169458599013a06fbf3024effcb860d62ed/narwhals/_expression_parsing.py#L112
"""

from __future__ import annotations

from itertools import chain
from operator import methodcaller
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Protocol
from typing import Sequence
from typing import TypeVar

from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals.typing import CompliantDataFrame
from narwhals.typing import CompliantExpr
from narwhals.typing import CompliantNamespace
from narwhals.typing import CompliantSeries

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Iterable
    from typing import Iterator

    from typing_extensions import Self
    from typing_extensions import TypeIs

    from narwhals.dtypes import DType
    from narwhals.typing import NativeSeries
    from narwhals.typing import _1DArray
    from narwhals.utils import Implementation
    from narwhals.utils import Version

    T = TypeVar("T")

NativeSeriesT_co = TypeVar("NativeSeriesT_co", bound="NativeSeries", covariant=True)
ReuseSeriesT = TypeVar("ReuseSeriesT", bound="ReuseSeries[Any]")

ReuseDataFrameT = TypeVar("ReuseDataFrameT", bound="ReuseDataFrame[Any]")


class ReuseDataFrame(CompliantDataFrame, Protocol[ReuseSeriesT]):
    def _evaluate_into_expr(
        self: ReuseDataFrameT, expr: ReuseExpr[ReuseDataFrameT, ReuseSeriesT], /
    ) -> Sequence[ReuseSeriesT]:
        _, aliases = evaluate_output_names_and_aliases(expr, self, [])
        result = expr(self)
        if list(aliases) != [s.name for s in result]:
            msg = f"Safety assertion failed, expected {aliases}, got {result}"
            raise AssertionError(msg)
        return result

    def _evaluate_into_exprs(
        self, *exprs: ReuseExpr[Self, ReuseSeriesT]
    ) -> Sequence[ReuseSeriesT]:
        return list(chain.from_iterable(self._evaluate_into_expr(expr) for expr in exprs))

    # NOTE: `mypy` is **very** fragile here in what is permitted
    # DON'T CHANGE UNLESS IT SOLVES ANOTHER ISSUE
    def _maybe_evaluate_expr(
        self, expr: ReuseExpr[ReuseDataFrame[ReuseSeriesT], ReuseSeriesT] | T, /
    ) -> ReuseSeriesT | T:
        if is_reuse_expr(expr):
            result: Sequence[ReuseSeriesT] = expr(self)
            if len(result) > 1:
                msg = (
                    "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) "
                    "are not supported in this context"
                )
                raise ValueError(msg)
            return result[0]
        return expr


# NOTE: DON'T CHANGE THIS EITHER
def is_reuse_expr(
    obj: ReuseExpr[Any, ReuseSeriesT] | Any,
) -> TypeIs[ReuseExpr[Any, ReuseSeriesT]]:
    return hasattr(obj, "__narwhals_expr__")


class ReuseExpr(
    CompliantExpr[ReuseDataFrameT, ReuseSeriesT],
    Protocol[ReuseDataFrameT, ReuseSeriesT],
):
    _call: Callable[[ReuseDataFrameT], Sequence[ReuseSeriesT]]
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
        call: Callable[[ReuseDataFrameT], Sequence[ReuseSeriesT]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[ReuseDataFrameT], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None: ...

    def __call__(self, df: ReuseDataFrameT) -> Sequence[ReuseSeriesT]:
        return self._call(df)

    def __narwhals_namespace__(self) -> ReuseNamespace[ReuseDataFrameT, ReuseSeriesT]: ...

    def _reuse_series_implementation(
        self: ReuseExpr[ReuseDataFrameT, ReuseSeriesT],
        attr: str,
        *,
        returns_scalar: bool = False,
        **expressifiable_args: Any,
    ) -> ReuseExpr[ReuseDataFrameT, ReuseSeriesT]:
        from narwhals._expression_parsing import evaluate_output_names_and_aliases

        plx = self.__narwhals_namespace__()
        from_scalar = plx._create_series_from_scalar

        # NOTE: Ideally this would be implemented differently for `pandas` and `pyarrow`
        # - It wouldn't make sense to check the implementation for each call
        # - Just copying over from the function
        def func(df: ReuseDataFrameT, /) -> Sequence[ReuseSeriesT]:
            _kwargs = {
                arg_name: df._maybe_evaluate_expr(arg_value)
                for arg_name, arg_value in expressifiable_args.items()
            }
            # For PyArrow.Series, we return Python Scalars (like Polars does) instead of PyArrow Scalars.
            # However, when working with expressions, we keep everything PyArrow-native.
            extra_kwargs = (
                {"_return_py_scalar": False}
                if returns_scalar and self._implementation.is_pyarrow()
                else {}
            )
            method = methodcaller(attr, **extra_kwargs, **_kwargs)
            out: Sequence[ReuseSeriesT] = [
                from_scalar(method(series), reference_series=series)
                if returns_scalar
                else method(series)
                for series in self(df)
            ]
            _, aliases = evaluate_output_names_and_aliases(self, df, [])
            if [s.name for s in out] != list(aliases):  # pragma: no cover
                msg = (
                    f"Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues\n"
                    f"Expression aliases: {aliases}\n"
                    f"Series names: {[s.name for s in out]}"
                )
                raise AssertionError(msg)
            return out

        return plx._create_expr_from_callable(
            func,
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{attr}",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            kwargs={**self._kwargs, **expressifiable_args},
        )

    def _reuse_series_namespace_implementation(
        self: ReuseExpr[ReuseDataFrameT, ReuseSeriesT],
        series_namespace: str,
        attr: str,
        **kwargs: Any,
    ) -> ReuseExpr[ReuseDataFrameT, ReuseSeriesT]:
        plx = self.__narwhals_namespace__()
        return plx._create_expr_from_callable(
            lambda df: [
                getattr(getattr(series, series_namespace), attr)(**kwargs)
                for series in self(df)
            ],
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{series_namespace}.{attr}",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            kwargs={**self._kwargs, **kwargs},
        )


class ReuseSeries(CompliantSeries, Protocol[NativeSeriesT_co]):  # type: ignore[misc]
    _name: str
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
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self: ...
    def _from_native_series(self: Self, series: NativeSeriesT_co | Any) -> Self: ...
    @property
    def dtype(self: Self) -> DType: ...
    @property
    def name(self: Self) -> str:
        return self._name


class ReuseNamespace(
    CompliantNamespace[ReuseDataFrameT, ReuseSeriesT],
    Protocol[ReuseDataFrameT, ReuseSeriesT],
):
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    def __init__(
        self,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None: ...

    def __narwhals_expr__(
        self,
    ) -> Callable[..., ReuseExpr[ReuseDataFrameT, ReuseSeriesT]]: ...
    # Both do very similar things:
    # - `_pandas_like.utils.create_compliant_series`
    # - `_arrow.series.ArrowSeries(native_series=pa.chunked_array([value]))`
    def _create_compliant_series(self, value: Any) -> ReuseSeriesT: ...

    # NOTE: Fully spec'd
    def _create_expr_from_callable(
        self,
        func: Callable[[ReuseDataFrameT], Sequence[ReuseSeriesT]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[ReuseDataFrameT], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        kwargs: dict[str, Any],
    ) -> ReuseExpr[ReuseDataFrameT, ReuseSeriesT]:
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
    ) -> ReuseExpr[ReuseDataFrameT, ReuseSeriesT]:
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

    def _create_series_from_scalar(
        self, value: Any, *, reference_series: ReuseSeriesT
    ) -> ReuseSeriesT: ...
