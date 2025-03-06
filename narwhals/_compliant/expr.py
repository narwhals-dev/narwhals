from __future__ import annotations

import sys
from functools import partial
from operator import methodcaller
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Protocol
from typing import Sequence

from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import CompliantLazyFrameT
from narwhals._compliant.typing import CompliantSeriesOrNativeExprT_co
from narwhals._compliant.typing import EagerDataFrameT
from narwhals._compliant.typing import EagerSeriesT
from narwhals._compliant.typing import NativeExprT_co
from narwhals._expression_parsing import evaluate_output_names_and_aliases
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

    from narwhals._compliant.namespace import CompliantNamespace
    from narwhals._compliant.series import CompliantSeries
    from narwhals._expression_parsing import ExprKind
    from narwhals.dtypes import DType
    from narwhals.utils import Implementation
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

__all__ = ["CompliantExpr"]


# NOTE: Only common methods for lazy expr-like objects
class NativeExpr(Protocol):
    def between(self, *args: Any, **kwds: Any) -> Any: ...
    def isin(self, *args: Any, **kwds: Any) -> Any: ...


class CompliantExpr(Protocol38[CompliantFrameT, CompliantSeriesOrNativeExprT_co]):
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version
    _evaluate_output_names: Callable[[CompliantFrameT], Sequence[str]]
    _alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None
    _depth: int
    _function_name: str

    def __call__(
        self, df: CompliantFrameT
    ) -> Sequence[CompliantSeriesOrNativeExprT_co]: ...
    def __narwhals_expr__(self) -> None: ...
    def __narwhals_namespace__(
        self,
    ) -> CompliantNamespace[CompliantFrameT, CompliantSeriesOrNativeExprT_co]: ...
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
    def over(self: Self, keys: Sequence[str], kind: ExprKind) -> Self: ...
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
        function: Callable[[CompliantSeries], CompliantExpr[Any, Any]],
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
        min_samples: int | None,
        center: bool,
    ) -> Self: ...

    @unstable
    def rolling_mean(
        self,
        window_size: int,
        *,
        min_samples: int | None,
        center: bool,
    ) -> Self: ...

    @unstable
    def rolling_var(
        self,
        window_size: int,
        *,
        min_samples: int | None,
        center: bool,
        ddof: int,
    ) -> Self: ...

    @unstable
    def rolling_std(
        self,
        window_size: int,
        *,
        min_samples: int | None,
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


class EagerExpr(
    CompliantExpr[EagerDataFrameT, EagerSeriesT],
    Protocol38[EagerDataFrameT, EagerSeriesT],
):
    _depth: int
    _function_name: str
    _evaluate_output_names: Any
    _alias_output_names: Any
    _call_kwargs: dict[str, Any]

    @property
    def _series(self) -> type[EagerSeriesT]: ...

    def __init__(
        self: Self,
        call: Callable[[EagerDataFrameT], Sequence[EagerSeriesT]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[EagerDataFrameT], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
        call_kwargs: dict[str, Any] | None = None,
    ) -> None: ...

    @classmethod
    def _from_callable(
        cls,
        func: Callable[[EagerDataFrameT], Sequence[EagerSeriesT]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[EagerDataFrameT], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
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
    def _from_series(cls, series: EagerSeriesT, *, context: _FullContext) -> Self:
        return cls(
            lambda _df: [series],
            depth=0,
            function_name="series",
            evaluate_output_names=lambda _df: [series.name],
            alias_output_names=None,
            implementation=context._implementation,
            backend_version=context._backend_version,
            version=context._version,
        )

    # https://github.com/narwhals-dev/narwhals/blob/35cef0b1e2c892fb24aa730902b08b6994008c18/narwhals/_protocols.py#L135
    def _reuse_series_implementation(
        self: EagerExpr[EagerDataFrameT, EagerSeriesT],
        attr: str,
        *,
        returns_scalar: bool = False,
        call_kwargs: dict[str, Any] | None = None,
        **expressifiable_args: Any,
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        func = partial(
            self._reuse_series_inner,
            method_name=attr,
            returns_scalar=returns_scalar,
            call_kwargs=call_kwargs or {},
            expressifiable_args=expressifiable_args,
        )
        return self._from_callable(
            func,
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{attr}",
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
                arg_name: df._maybe_evaluate_expr(arg_value)
                for arg_name, arg_value in expressifiable_args.items()
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

    def _reuse_series_namespace_implementation(
        self: EagerExpr[EagerDataFrameT, EagerSeriesT],
        series_namespace: str,
        attr: str,
        **kwargs: Any,
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._from_callable(
            lambda df: [
                getattr(getattr(series, series_namespace), attr)(**kwargs)
                for series in self(df)
            ],
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{series_namespace}.{attr}",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            call_kwargs={**self._call_kwargs, **kwargs},
            context=self,
        )


# NOTE: See (https://github.com/narwhals-dev/narwhals/issues/2044#issuecomment-2674262833)
class LazyExpr(
    CompliantExpr[CompliantLazyFrameT, NativeExprT_co],
    Protocol38[CompliantLazyFrameT, NativeExprT_co],
):
    arg_min: not_implemented = not_implemented()
    arg_max: not_implemented = not_implemented()
    arg_true: not_implemented = not_implemented()
    head: not_implemented = not_implemented()
    tail: not_implemented = not_implemented()
    mode: not_implemented = not_implemented()
    sort: not_implemented = not_implemented()
    rank: not_implemented = not_implemented()
    sample: not_implemented = not_implemented()
    map_batches: not_implemented = not_implemented()
    ewm_mean: not_implemented = not_implemented()
    rolling_sum: not_implemented = not_implemented()
    rolling_mean: not_implemented = not_implemented()
    rolling_var: not_implemented = not_implemented()
    rolling_std: not_implemented = not_implemented()
    gather_every: not_implemented = not_implemented()
    replace_strict: not_implemented = not_implemented()
    cat: not_implemented = not_implemented()  # pyright: ignore[reportAssignmentType]
