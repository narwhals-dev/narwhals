from __future__ import annotations

import sys
from functools import partial
from operator import methodcaller
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generic
from typing import Literal
from typing import Mapping
from typing import Protocol
from typing import Sequence

from narwhals._compliant.any_namespace import CatNamespace
from narwhals._compliant.any_namespace import DateTimeNamespace
from narwhals._compliant.any_namespace import ListNamespace
from narwhals._compliant.any_namespace import NameNamespace
from narwhals._compliant.any_namespace import StringNamespace
from narwhals._compliant.any_namespace import StructNamespace
from narwhals._compliant.namespace import CompliantNamespace
from narwhals._compliant.typing import AliasName
from narwhals._compliant.typing import AliasNames
from narwhals._compliant.typing import CompliantExprT_co
from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import CompliantLazyFrameT
from narwhals._compliant.typing import CompliantSeriesOrNativeExprT_co
from narwhals._compliant.typing import EagerDataFrameT
from narwhals._compliant.typing import EagerExprT
from narwhals._compliant.typing import EagerSeriesT
from narwhals._compliant.typing import LazyExprT
from narwhals._compliant.typing import NativeExprT
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals.dependencies import get_numpy
from narwhals.dependencies import is_numpy_array
from narwhals.dtypes import DType
from narwhals.utils import _StoresCompliant
from narwhals.utils import deprecated
from narwhals.utils import not_implemented
from narwhals.utils import unstable

if not TYPE_CHECKING:  # pragma: no cover
    if sys.version_info >= (3, 9):
        from typing import Protocol as Protocol38
    else:
        from typing import Generic as Protocol38
else:  # pragma: no cover
    # TODO @dangotbanned: Remove after dropping `3.8` (#2084)
    # - https://github.com/narwhals-dev/narwhals/pull/2064#discussion_r1965921386
    from typing import Protocol as Protocol38

if TYPE_CHECKING:
    from typing import Mapping

    from typing_extensions import Self
    from typing_extensions import TypeIs

    from narwhals._compliant.namespace import CompliantNamespace
    from narwhals._compliant.namespace import EagerNamespace
    from narwhals._compliant.series import CompliantSeries
    from narwhals._compliant.typing import AliasNames
    from narwhals._compliant.typing import EvalNames
    from narwhals._compliant.typing import EvalSeries
    from narwhals._expression_parsing import ExprKind
    from narwhals._expression_parsing import ExprMetadata
    from narwhals.dtypes import DType
    from narwhals.typing import TimeUnit
    from narwhals.utils import Implementation
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

__all__ = ["CompliantExpr", "EagerExpr", "LazyExpr", "NativeExpr"]


class NativeExpr(Protocol):
    """An `Expr`-like object from a package with [Lazy-only support](https://narwhals-dev.github.io/narwhals/extending/#levels-of-support).

    Protocol members are chosen *purely* for matching statically - as they
    are common to all currently supported packages.
    """

    def between(self, *args: Any, **kwds: Any) -> Any: ...
    def isin(self, *args: Any, **kwds: Any) -> Any: ...


class CompliantExpr(Protocol38[CompliantFrameT, CompliantSeriesOrNativeExprT_co]):
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version
    _evaluate_output_names: EvalNames[CompliantFrameT]
    _alias_output_names: AliasNames | None
    _metadata: ExprMetadata | None

    def __call__(
        self, df: CompliantFrameT
    ) -> Sequence[CompliantSeriesOrNativeExprT_co]: ...
    def __narwhals_expr__(self) -> None: ...
    def __narwhals_namespace__(self) -> CompliantNamespace[CompliantFrameT, Self]: ...
    @classmethod
    def from_column_names(
        cls,
        evaluate_column_names: EvalNames[CompliantFrameT],
        /,
        *,
        context: _FullContext,
    ) -> Self: ...
    @classmethod
    def from_column_indices(
        cls: type[Self], *column_indices: int, context: _FullContext
    ) -> Self: ...

    def is_null(self) -> Self: ...
    def abs(self) -> Self: ...
    def all(self) -> Self: ...
    def any(self) -> Self: ...
    def alias(self, name: str) -> Self: ...
    def cast(self, dtype: DType | type[DType]) -> Self: ...
    def count(self) -> Self: ...
    def min(self) -> Self: ...
    def max(self) -> Self: ...
    def arg_min(self) -> Self: ...
    def arg_max(self) -> Self: ...
    def arg_true(self) -> Self: ...
    def mean(self) -> Self: ...
    def sum(self) -> Self: ...
    def median(self) -> Self: ...
    def skew(self) -> Self: ...
    def std(self, *, ddof: int) -> Self: ...
    def var(self, *, ddof: int) -> Self: ...
    def n_unique(self) -> Self: ...
    def null_count(self) -> Self: ...
    def drop_nulls(self) -> Self: ...
    def fill_null(
        self,
        value: Any | None,
        strategy: Literal["forward", "backward"] | None,
        limit: int | None,
    ) -> Self: ...
    def diff(self) -> Self: ...
    def unique(self) -> Self: ...
    def len(self) -> Self: ...
    def round(self, decimals: int) -> Self: ...
    def mode(self) -> Self: ...
    def head(self, n: int) -> Self: ...
    def tail(self, n: int) -> Self: ...
    def shift(self, n: int) -> Self: ...
    def is_finite(self) -> Self: ...
    def is_nan(self) -> Self: ...
    def is_unique(self) -> Self: ...
    def is_first_distinct(self) -> Self: ...
    def is_last_distinct(self) -> Self: ...
    def cum_sum(self, *, reverse: bool) -> Self: ...
    def cum_count(self, *, reverse: bool) -> Self: ...
    def cum_min(self, *, reverse: bool) -> Self: ...
    def cum_max(self, *, reverse: bool) -> Self: ...
    def cum_prod(self, *, reverse: bool) -> Self: ...
    def is_in(self, other: Any) -> Self: ...
    def sort(self, *, descending: bool, nulls_last: bool) -> Self: ...
    def rank(
        self,
        method: Literal["average", "min", "max", "dense", "ordinal"],
        *,
        descending: bool,
    ) -> Self: ...
    def replace_strict(
        self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any],
        *,
        return_dtype: DType | type[DType] | None,
    ) -> Self: ...
    def over(
        self, partition_by: Sequence[str], order_by: Sequence[str] | None
    ) -> Self: ...
    def sample(
        self,
        n: int | None,
        *,
        fraction: float | None,
        with_replacement: bool,
        seed: int | None,
    ) -> Self: ...
    def quantile(
        self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self: ...
    def map_batches(
        self,
        function: Callable[[CompliantSeries[Any]], CompliantExpr[Any, Any]],
        return_dtype: DType | type[DType] | None,
    ) -> Self: ...

    @property
    def str(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def dt(self) -> Any: ...
    @property
    def cat(self) -> Any: ...
    @property
    def list(self) -> Any: ...
    @property
    def struct(self) -> Any: ...

    @unstable
    def ewm_mean(
        self,
        *,
        com: float | None,
        span: float | None,
        half_life: float | None,
        alpha: float | None,
        adjust: bool,
        min_samples: int,
        ignore_nulls: bool,
    ) -> Self: ...

    @unstable
    def rolling_sum(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
    ) -> Self: ...

    @unstable
    def rolling_mean(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
    ) -> Self: ...

    @unstable
    def rolling_var(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
        ddof: int,
    ) -> Self: ...

    @unstable
    def rolling_std(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
        ddof: int,
    ) -> Self: ...

    @deprecated("Since `1.22.0`")
    def gather_every(self, n: int, offset: int) -> Self: ...
    def __and__(self, other: Any) -> Self: ...
    def __or__(self, other: Any) -> Self: ...
    def __add__(self, other: Any) -> Self: ...
    def __sub__(self, other: Any) -> Self: ...
    def __mul__(self, other: Any) -> Self: ...
    def __floordiv__(self, other: Any) -> Self: ...
    def __truediv__(self, other: Any) -> Self: ...
    def __mod__(self, other: Any) -> Self: ...
    def __pow__(self, other: Any) -> Self: ...
    def __gt__(self, other: Any) -> Self: ...
    def __ge__(self, other: Any) -> Self: ...
    def __lt__(self, other: Any) -> Self: ...
    def __le__(self, other: Any) -> Self: ...
    def __invert__(self) -> Self: ...
    def broadcast(
        self, kind: Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]
    ) -> Self: ...
    def _is_multi_output_unnamed(self) -> bool:
        """Return `True` for multi-output aggregations without names.

        For example, column `'a'` only appears in the output as a grouping key:

            df.group_by('a').agg(nw.all().sum())

        It does not get included in:

            nw.all().sum().
        """
        assert self._metadata is not None  # noqa: S101
        return self._metadata.expansion_kind.is_multi_unnamed()


class DepthTrackingExpr(
    CompliantExpr[CompliantFrameT, CompliantSeriesOrNativeExprT_co],
    Protocol38[CompliantFrameT, CompliantSeriesOrNativeExprT_co],
):
    _depth: int
    _function_name: str

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: EvalNames[CompliantFrameT],
        /,
        *,
        context: _FullContext,
        function_name: str = "",
    ) -> Self: ...

    def _is_elementary(self) -> bool:
        """Check if expr is elementary.

        Examples:
            - nw.col('a').mean()  # depth 1
            - nw.mean('a')  # depth 1
            - nw.len()  # depth 0

        as opposed to, say

            - nw.col('a').filter(nw.col('b')>nw.col('c')).max()

        Elementary expressions are the only ones supported properly in
        pandas, PyArrow, and Dask.
        """
        return self._depth < 2

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(depth={self._depth}, function_name={self._function_name})"


class EagerExpr(
    DepthTrackingExpr[EagerDataFrameT, EagerSeriesT],
    Protocol38[EagerDataFrameT, EagerSeriesT],
):
    _call: EvalSeries[EagerDataFrameT, EagerSeriesT]
    _call_kwargs: dict[str, Any]

    def __init__(
        self: Self,
        call: EvalSeries[EagerDataFrameT, EagerSeriesT],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: EvalNames[EagerDataFrameT],
        alias_output_names: AliasNames | None,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
        call_kwargs: dict[str, Any] | None = None,
    ) -> None: ...

    def __call__(self, df: EagerDataFrameT) -> Sequence[EagerSeriesT]:
        return self._call(df)

    def __narwhals_namespace__(
        self,
    ) -> EagerNamespace[EagerDataFrameT, EagerSeriesT, Self, Any, Any]: ...
    def __narwhals_expr__(self) -> None: ...

    @classmethod
    def _from_callable(
        cls,
        func: EvalSeries[EagerDataFrameT, EagerSeriesT],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: EvalNames[EagerDataFrameT],
        alias_output_names: AliasNames | None,
        context: _FullContext,
        call_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        return cls(
            func,
            depth=depth,
            function_name=function_name,
            evaluate_output_names=evaluate_output_names,
            alias_output_names=alias_output_names,
            implementation=context._implementation,
            backend_version=context._backend_version,
            version=context._version,
            call_kwargs=call_kwargs,
        )

    @classmethod
    def _from_series(cls, series: EagerSeriesT) -> Self:
        return cls(
            lambda _df: [series],
            depth=0,
            function_name="series",
            evaluate_output_names=lambda _df: [series.name],
            alias_output_names=None,
            implementation=series._implementation,
            backend_version=series._backend_version,
            version=series._version,
        )

    def _reuse_series(
        self: Self,
        method_name: str,
        *,
        returns_scalar: bool = False,
        call_kwargs: dict[str, Any] | None = None,
        **expressifiable_args: Any,
    ) -> Self:
        """Reuse Series implementation for expression.

        If Series.foo is already defined, and we'd like Expr.foo to be the same, we can
        leverage this method to do that for us.

        Arguments:
            method_name: name of method.
            returns_scalar: whether the Series version returns a scalar. In this case,
                the expression version should return a 1-row Series.
            call_kwargs: non-expressifiable args which we may need to reuse in `agg` or `over`,
                such as `ddof` for `std` and `var`.
            expressifiable_args: keyword arguments to pass to function, which may
                be expressifiable (e.g. `nw.col('a').is_between(3, nw.col('b')))`).
        """
        func = partial(
            self._reuse_series_inner,
            method_name=method_name,
            returns_scalar=returns_scalar,
            call_kwargs=call_kwargs or {},
            expressifiable_args=expressifiable_args,
        )
        return self._from_callable(
            func,
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{method_name}",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            call_kwargs=call_kwargs,
            context=self,
        )

    # For PyArrow.Series, we return Python Scalars (like Polars does) instead of PyArrow Scalars.
    # However, when working with expressions, we keep everything PyArrow-native.
    def _reuse_series_extra_kwargs(
        self, *, returns_scalar: bool = False
    ) -> dict[str, Any]:
        return {}

    @classmethod
    def _is_expr(cls, obj: Self | Any) -> TypeIs[Self]:
        return hasattr(obj, "__narwhals_expr__")

    def _reuse_series_inner(
        self,
        df: EagerDataFrameT,
        *,
        method_name: str,
        returns_scalar: bool,
        call_kwargs: dict[str, Any],
        expressifiable_args: dict[str, Any],
    ) -> Sequence[EagerSeriesT]:
        kwargs = {
            **call_kwargs,
            **{
                name: df._evaluate_expr(value) if self._is_expr(value) else value
                for name, value in expressifiable_args.items()
            },
        }
        method = methodcaller(
            method_name,
            **self._reuse_series_extra_kwargs(returns_scalar=returns_scalar),
            **kwargs,
        )
        out: Sequence[EagerSeriesT] = [
            series._from_scalar(method(series)) if returns_scalar else method(series)
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

    def _reuse_series_namespace(
        self: Self,
        series_namespace: Literal["cat", "dt", "list", "name", "str", "struct"],
        method_name: str,
        **kwargs: Any,
    ) -> Self:
        """Reuse Series implementation for expression.

        Just like `_reuse_series`, but for e.g. `Expr.dt.foo` instead
        of `Expr.foo`.

        Arguments:
            series_namespace: The Series namespace.
            method_name: name of method, within `series_namespace`.
            kwargs: keyword arguments to pass to function.
        """
        return self._from_callable(
            lambda df: [
                getattr(getattr(series, series_namespace), method_name)(**kwargs)
                for series in self(df)
            ],
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{series_namespace}.{method_name}",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            call_kwargs={**self._call_kwargs, **kwargs},
            context=self,
        )

    def broadcast(self, kind: Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]) -> Self:
        # Mark the resulting Series with `_broadcast = True`.
        # Then, when extracting native objects, `extract_native` will
        # know what to do.
        def func(df: EagerDataFrameT) -> list[EagerSeriesT]:
            results = []
            for result in self(df):
                result._broadcast = True
                results.append(result)
            return results

        return type(self)(
            func,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            implementation=self._implementation,
            version=self._version,
            call_kwargs=self._call_kwargs,
        )

    def cast(self, dtype: DType | type[DType]) -> Self:
        return self._reuse_series("cast", dtype=dtype)

    def __eq__(self, other: Self | Any) -> Self:  # type: ignore[override]
        return self._reuse_series("__eq__", other=other)

    def __ne__(self, other: Self | Any) -> Self:  # type: ignore[override]
        return self._reuse_series("__ne__", other=other)

    def __ge__(self, other: Self | Any) -> Self:
        return self._reuse_series("__ge__", other=other)

    def __gt__(self, other: Self | Any) -> Self:
        return self._reuse_series("__gt__", other=other)

    def __le__(self, other: Self | Any) -> Self:
        return self._reuse_series("__le__", other=other)

    def __lt__(self, other: Self | Any) -> Self:
        return self._reuse_series("__lt__", other=other)

    def __and__(self, other: Self | bool | Any) -> Self:
        return self._reuse_series("__and__", other=other)

    def __or__(self, other: Self | bool | Any) -> Self:
        return self._reuse_series("__or__", other=other)

    def __add__(self, other: Self | Any) -> Self:
        return self._reuse_series("__add__", other=other)

    def __sub__(self, other: Self | Any) -> Self:
        return self._reuse_series("__sub__", other=other)

    def __rsub__(self, other: Self | Any) -> Self:
        return self.alias("literal")._reuse_series("__rsub__", other=other)

    def __mul__(self, other: Self | Any) -> Self:
        return self._reuse_series("__mul__", other=other)

    def __truediv__(self, other: Self | Any) -> Self:
        return self._reuse_series("__truediv__", other=other)

    def __rtruediv__(self, other: Self | Any) -> Self:
        return self.alias("literal")._reuse_series("__rtruediv__", other=other)

    def __floordiv__(self, other: Self | Any) -> Self:
        return self._reuse_series("__floordiv__", other=other)

    def __rfloordiv__(self, other: Self | Any) -> Self:
        return self.alias("literal")._reuse_series("__rfloordiv__", other=other)

    def __pow__(self, other: Self | Any) -> Self:
        return self._reuse_series("__pow__", other=other)

    def __rpow__(self, other: Self | Any) -> Self:
        return self.alias("literal")._reuse_series("__rpow__", other=other)

    def __mod__(self, other: Self | Any) -> Self:
        return self._reuse_series("__mod__", other=other)

    def __rmod__(self, other: Self | Any) -> Self:
        return self.alias("literal")._reuse_series("__rmod__", other=other)

    # Unary
    def __invert__(self) -> Self:
        return self._reuse_series("__invert__")

    # Reductions
    def null_count(self) -> Self:
        return self._reuse_series("null_count", returns_scalar=True)

    def n_unique(self) -> Self:
        return self._reuse_series("n_unique", returns_scalar=True)

    def sum(self) -> Self:
        return self._reuse_series("sum", returns_scalar=True)

    def count(self) -> Self:
        return self._reuse_series("count", returns_scalar=True)

    def mean(self) -> Self:
        return self._reuse_series("mean", returns_scalar=True)

    def median(self) -> Self:
        return self._reuse_series("median", returns_scalar=True)

    def std(self, *, ddof: int) -> Self:
        return self._reuse_series("std", returns_scalar=True, call_kwargs={"ddof": ddof})

    def var(self, *, ddof: int) -> Self:
        return self._reuse_series("var", returns_scalar=True, call_kwargs={"ddof": ddof})

    def skew(self) -> Self:
        return self._reuse_series("skew", returns_scalar=True)

    def any(self) -> Self:
        return self._reuse_series("any", returns_scalar=True)

    def all(self) -> Self:
        return self._reuse_series("all", returns_scalar=True)

    def max(self) -> Self:
        return self._reuse_series("max", returns_scalar=True)

    def min(self) -> Self:
        return self._reuse_series("min", returns_scalar=True)

    def arg_min(self) -> Self:
        return self._reuse_series("arg_min", returns_scalar=True)

    def arg_max(self) -> Self:
        return self._reuse_series("arg_max", returns_scalar=True)

    # Other

    def clip(self, lower_bound: Any, upper_bound: Any) -> Self:
        return self._reuse_series(
            "clip", lower_bound=lower_bound, upper_bound=upper_bound
        )

    def is_null(self) -> Self:
        return self._reuse_series("is_null")

    def is_nan(self) -> Self:
        return self._reuse_series("is_nan")

    def fill_null(
        self,
        value: Any | None,
        strategy: Literal["forward", "backward"] | None,
        limit: int | None,
    ) -> Self:
        return self._reuse_series(
            "fill_null", value=value, strategy=strategy, limit=limit
        )

    def is_in(self, other: Any) -> Self:
        return self._reuse_series("is_in", other=other)

    def arg_true(self) -> Self:
        return self._reuse_series("arg_true")

    def filter(self, *predicates: Self) -> Self:
        plx = self.__narwhals_namespace__()
        predicate = plx.all_horizontal(*predicates)
        return self._reuse_series("filter", predicate=predicate)

    def drop_nulls(self) -> Self:
        return self._reuse_series("drop_nulls")

    def replace_strict(
        self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any],
        *,
        return_dtype: DType | type[DType] | None,
    ) -> Self:
        return self._reuse_series(
            "replace_strict", old=old, new=new, return_dtype=return_dtype
        )

    def sort(self, *, descending: bool, nulls_last: bool) -> Self:
        return self._reuse_series("sort", descending=descending, nulls_last=nulls_last)

    def abs(self) -> Self:
        return self._reuse_series("abs")

    def unique(self) -> Self:
        return self._reuse_series("unique", maintain_order=False)

    def diff(self) -> Self:
        return self._reuse_series("diff")

    def sample(
        self,
        n: int | None,
        *,
        fraction: float | None,
        with_replacement: bool,
        seed: int | None,
    ) -> Self:
        return self._reuse_series(
            "sample", n=n, fraction=fraction, with_replacement=with_replacement, seed=seed
        )

    def alias(self: Self, name: str) -> Self:
        def alias_output_names(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = f"Expected function with single output, found output names: {names}"
                raise ValueError(msg)
            return [name]

        # Define this one manually, so that we can
        # override `output_names` and not increase depth
        return type(self)(
            lambda df: [series.alias(name) for series in self(df)],
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            backend_version=self._backend_version,
            implementation=self._implementation,
            version=self._version,
            call_kwargs=self._call_kwargs,
        )

    def is_unique(self) -> Self:
        return self._reuse_series("is_unique")

    def is_first_distinct(self) -> Self:
        return self._reuse_series("is_first_distinct")

    def is_last_distinct(self) -> Self:
        return self._reuse_series("is_last_distinct")

    def quantile(
        self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        return self._reuse_series(
            "quantile",
            quantile=quantile,
            interpolation=interpolation,
            returns_scalar=True,
        )

    def head(self, n: int) -> Self:
        return self._reuse_series("head", n=n)

    def tail(self, n: int) -> Self:
        return self._reuse_series("tail", n=n)

    def round(self, decimals: int) -> Self:
        return self._reuse_series("round", decimals=decimals)

    def len(self) -> Self:
        return self._reuse_series("len", returns_scalar=True)

    def gather_every(self, n: int, offset: int) -> Self:
        return self._reuse_series("gather_every", n=n, offset=offset)

    def mode(self) -> Self:
        return self._reuse_series("mode")

    def is_finite(self) -> Self:
        return self._reuse_series("is_finite")

    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._reuse_series(
            "rolling_mean",
            window_size=window_size,
            min_samples=min_samples,
            center=center,
        )

    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._reuse_series(
            "rolling_std",
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            ddof=ddof,
        )

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._reuse_series(
            "rolling_sum", window_size=window_size, min_samples=min_samples, center=center
        )

    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._reuse_series(
            "rolling_var",
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            ddof=ddof,
        )

    def map_batches(
        self: Self,
        function: Callable[[Any], Any],
        return_dtype: DType | type[DType] | None,
    ) -> Self:
        def func(df: EagerDataFrameT) -> Sequence[EagerSeriesT]:
            input_series_list = self(df)
            output_names = [input_series.name for input_series in input_series_list]
            result = [function(series) for series in input_series_list]
            if is_numpy_array(result[0]) or (
                (np := get_numpy()) is not None and np.isscalar(result[0])
            ):
                from_numpy = partial(
                    self.__narwhals_namespace__()._series.from_numpy, context=self
                )
                result = [
                    from_numpy(array).alias(output_name)
                    for array, output_name in zip(result, output_names)
                ]
            if return_dtype is not None:
                result = [series.cast(return_dtype) for series in result]
            return result

        return self._from_callable(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->map_batches",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            context=self,
        )

    @property
    def cat(self) -> EagerExprCatNamespace[Self]:
        return EagerExprCatNamespace(self)

    @property
    def dt(self) -> EagerExprDateTimeNamespace[Self]:
        return EagerExprDateTimeNamespace(self)

    @property
    def list(self) -> EagerExprListNamespace[Self]:
        return EagerExprListNamespace(self)

    @property
    def name(self) -> EagerExprNameNamespace[Self]:
        return EagerExprNameNamespace(self)

    @property
    def str(self) -> EagerExprStringNamespace[Self]:
        return EagerExprStringNamespace(self)

    @property
    def struct(self) -> EagerExprStructNamespace[Self]:
        return EagerExprStructNamespace(self)


class LazyExpr(  # type: ignore[misc]
    CompliantExpr[CompliantLazyFrameT, NativeExprT],
    Protocol38[CompliantLazyFrameT, NativeExprT],
):
    arg_min: not_implemented = not_implemented()
    arg_max: not_implemented = not_implemented()
    arg_true: not_implemented = not_implemented()
    head: not_implemented = not_implemented()
    tail: not_implemented = not_implemented()
    mode: not_implemented = not_implemented()
    sort: not_implemented = not_implemented()
    sample: not_implemented = not_implemented()
    map_batches: not_implemented = not_implemented()
    ewm_mean: not_implemented = not_implemented()
    gather_every: not_implemented = not_implemented()
    replace_strict: not_implemented = not_implemented()
    cat: not_implemented = not_implemented()  # pyright: ignore[reportAssignmentType]

    @classmethod
    def _is_expr(cls, obj: Self | Any) -> TypeIs[Self]:
        return hasattr(obj, "__narwhals_expr__")

    def _with_callable(self, call: Callable[..., Any], /) -> Self: ...
    def _with_alias_output_names(self, func: AliasNames | None, /) -> Self: ...

    @property
    def name(self) -> LazyExprNameNamespace[Self]:
        return LazyExprNameNamespace(self)


class _ExprNamespace(  # type: ignore[misc]
    _StoresCompliant[CompliantExprT_co], Protocol[CompliantExprT_co]
):
    _compliant_expr: CompliantExprT_co

    @property
    def compliant(self) -> CompliantExprT_co:
        return self._compliant_expr


class EagerExprNamespace(_ExprNamespace[EagerExprT], Generic[EagerExprT]):
    def __init__(self, expr: EagerExprT, /) -> None:
        self._compliant_expr = expr


class LazyExprNamespace(_ExprNamespace[LazyExprT], Generic[LazyExprT]):
    def __init__(self, expr: LazyExprT, /) -> None:
        self._compliant_expr = expr


class EagerExprCatNamespace(
    EagerExprNamespace[EagerExprT], CatNamespace[EagerExprT], Generic[EagerExprT]
):
    def get_categories(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("cat", "get_categories")


class EagerExprDateTimeNamespace(
    EagerExprNamespace[EagerExprT], DateTimeNamespace[EagerExprT], Generic[EagerExprT]
):
    def to_string(self, format: str) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "to_string", format=format)

    def replace_time_zone(self, time_zone: str | None) -> EagerExprT:
        return self.compliant._reuse_series_namespace(
            "dt", "replace_time_zone", time_zone=time_zone
        )

    def convert_time_zone(self, time_zone: str) -> EagerExprT:
        return self.compliant._reuse_series_namespace(
            "dt", "convert_time_zone", time_zone=time_zone
        )

    def timestamp(self, time_unit: TimeUnit) -> EagerExprT:
        return self.compliant._reuse_series_namespace(
            "dt", "timestamp", time_unit=time_unit
        )

    def date(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "date")

    def year(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "year")

    def month(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "month")

    def day(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "day")

    def hour(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "hour")

    def minute(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "minute")

    def second(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "second")

    def millisecond(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "millisecond")

    def microsecond(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "microsecond")

    def nanosecond(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "nanosecond")

    def ordinal_day(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "ordinal_day")

    def weekday(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "weekday")

    def total_minutes(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "total_minutes")

    def total_seconds(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "total_seconds")

    def total_milliseconds(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "total_milliseconds")

    def total_microseconds(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "total_microseconds")

    def total_nanoseconds(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("dt", "total_nanoseconds")


class EagerExprListNamespace(
    EagerExprNamespace[EagerExprT], ListNamespace[EagerExprT], Generic[EagerExprT]
):
    def len(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("list", "len")


class CompliantExprNameNamespace(  # type: ignore[misc]
    _ExprNamespace[CompliantExprT_co],
    NameNamespace[CompliantExprT_co],
    Protocol[CompliantExprT_co],
):
    def keep(self) -> CompliantExprT_co:
        return self._from_callable(lambda name: name, alias=False)

    def map(self, function: AliasName) -> CompliantExprT_co:
        return self._from_callable(function)

    def prefix(self, prefix: str) -> CompliantExprT_co:
        return self._from_callable(lambda name: f"{prefix}{name}")

    def suffix(self, suffix: str) -> CompliantExprT_co:
        return self._from_callable(lambda name: f"{name}{suffix}")

    def to_lowercase(self) -> CompliantExprT_co:
        return self._from_callable(str.lower)

    def to_uppercase(self) -> CompliantExprT_co:
        return self._from_callable(str.upper)

    @staticmethod
    def _alias_output_names(func: AliasName, /) -> AliasNames:
        def fn(output_names: Sequence[str], /) -> Sequence[str]:
            return [func(name) for name in output_names]

        return fn

    def _from_callable(
        self, func: AliasName, /, *, alias: bool = True
    ) -> CompliantExprT_co: ...


class EagerExprNameNamespace(
    EagerExprNamespace[EagerExprT],
    CompliantExprNameNamespace[EagerExprT],
    Generic[EagerExprT],
):
    def _from_callable(self, func: AliasName, /, *, alias: bool = True) -> EagerExprT:
        expr = self.compliant
        return type(expr)(
            lambda df: [
                series.alias(func(name))
                for series, name in zip(expr(df), expr._evaluate_output_names(df))
            ],
            depth=expr._depth,
            function_name=expr._function_name,
            evaluate_output_names=expr._evaluate_output_names,
            alias_output_names=self._alias_output_names(func) if alias else None,
            backend_version=expr._backend_version,
            implementation=expr._implementation,
            version=expr._version,
            call_kwargs=expr._call_kwargs,
        )


class LazyExprNameNamespace(
    LazyExprNamespace[LazyExprT],
    CompliantExprNameNamespace[LazyExprT],
    Generic[LazyExprT],
):
    def _from_callable(self, func: AliasName, /, *, alias: bool = True) -> LazyExprT:
        expr = self.compliant
        output_names = self._alias_output_names(func) if alias else None
        return expr._with_alias_output_names(output_names)


class EagerExprStringNamespace(
    EagerExprNamespace[EagerExprT], StringNamespace[EagerExprT], Generic[EagerExprT]
):
    def len_chars(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("str", "len_chars")

    def replace(self, pattern: str, value: str, *, literal: bool, n: int) -> EagerExprT:
        return self.compliant._reuse_series_namespace(
            "str", "replace", pattern=pattern, value=value, literal=literal, n=n
        )

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> EagerExprT:
        return self.compliant._reuse_series_namespace(
            "str", "replace_all", pattern=pattern, value=value, literal=literal
        )

    def strip_chars(self, characters: str | None) -> EagerExprT:
        return self.compliant._reuse_series_namespace(
            "str", "strip_chars", characters=characters
        )

    def starts_with(self, prefix: str) -> EagerExprT:
        return self.compliant._reuse_series_namespace("str", "starts_with", prefix=prefix)

    def ends_with(self, suffix: str) -> EagerExprT:
        return self.compliant._reuse_series_namespace("str", "ends_with", suffix=suffix)

    def contains(self, pattern: str, *, literal: bool) -> EagerExprT:
        return self.compliant._reuse_series_namespace(
            "str", "contains", pattern=pattern, literal=literal
        )

    def slice(self, offset: int, length: int | None) -> EagerExprT:
        return self.compliant._reuse_series_namespace(
            "str", "slice", offset=offset, length=length
        )

    def split(self, by: str) -> EagerExprT:
        return self.compliant._reuse_series_namespace("str", "split", by=by)

    def to_datetime(self, format: str | None) -> EagerExprT:
        return self.compliant._reuse_series_namespace("str", "to_datetime", format=format)

    def to_lowercase(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("str", "to_lowercase")

    def to_uppercase(self) -> EagerExprT:
        return self.compliant._reuse_series_namespace("str", "to_uppercase")


class EagerExprStructNamespace(
    EagerExprNamespace[EagerExprT], StructNamespace[EagerExprT], Generic[EagerExprT]
):
    def field(self, name: str) -> EagerExprT:
        return self.compliant._reuse_series_namespace("struct", "field", name=name).alias(
            name
        )
