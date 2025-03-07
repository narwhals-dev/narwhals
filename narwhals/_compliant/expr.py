from __future__ import annotations

import sys
from functools import partial
from operator import methodcaller
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Mapping
from typing import Protocol
from typing import Sequence

from narwhals._compliant.namespace import CompliantNamespace
from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import CompliantLazyFrameT
from narwhals._compliant.typing import CompliantSeriesOrNativeExprT_co
from narwhals._compliant.typing import EagerDataFrameT
from narwhals._compliant.typing import EagerSeriesT
from narwhals._compliant.typing import NativeExprT_co
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals.dtypes import DType
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
    from narwhals._compliant.namespace import EagerNamespace
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
    _call: Callable[[EagerDataFrameT], Sequence[EagerSeriesT]]
    _depth: int
    _function_name: str
    _evaluate_output_names: Any
    _alias_output_names: Any
    _call_kwargs: dict[str, Any]

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

    def __call__(self, df: EagerDataFrameT) -> Sequence[EagerSeriesT]:
        return self._call(df)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(depth={self._depth}, function_name={self._function_name})"

    def __narwhals_namespace__(self) -> EagerNamespace[EagerDataFrameT, EagerSeriesT]: ...
    def __narwhals_expr__(self) -> None: ...

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

    def cast(
        self, dtype: DType | type[DType]
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("cast", dtype=dtype)

    def __eq__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:  # type: ignore[override]
        return self._reuse_series_implementation("__eq__", other=other)

    def __ne__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:  # type: ignore[override]
        return self._reuse_series_implementation("__ne__", other=other)

    def __ge__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__ge__", other=other)

    def __gt__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__gt__", other=other)

    def __le__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__le__", other=other)

    def __lt__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__lt__", other=other)

    def __and__(
        self, other: Self | bool | Any
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__and__", other=other)

    def __or__(
        self, other: Self | bool | Any
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__or__", other=other)

    def __add__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__add__", other=other)

    def __sub__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__sub__", other=other)

    def __rsub__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self.alias("literal")._reuse_series_implementation("__rsub__", other=other)

    def __mul__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__mul__", other=other)

    def __truediv__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__truediv__", other=other)

    def __rtruediv__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self.alias("literal")._reuse_series_implementation(
            "__rtruediv__", other=other
        )

    def __floordiv__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__floordiv__", other=other)

    def __rfloordiv__(
        self, other: Self | Any
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self.alias("literal")._reuse_series_implementation(
            "__rfloordiv__", other=other
        )

    def __pow__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__pow__", other=other)

    def __rpow__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self.alias("literal")._reuse_series_implementation("__rpow__", other=other)

    def __mod__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__mod__", other=other)

    def __rmod__(self, other: Self | Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self.alias("literal")._reuse_series_implementation("__rmod__", other=other)

    # Unary
    def __invert__(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("__invert__")

    # Reductions
    def null_count(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("null_count", returns_scalar=True)

    def n_unique(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("n_unique", returns_scalar=True)

    def sum(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("sum", returns_scalar=True)

    def mean(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("mean", returns_scalar=True)

    def median(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("median", returns_scalar=True)

    def std(self, *, ddof: int) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "std", returns_scalar=True, call_kwargs={"ddof": ddof}
        )

    def var(self, *, ddof: int) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "var", returns_scalar=True, call_kwargs={"ddof": ddof}
        )

    def skew(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("skew", returns_scalar=True)

    def any(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("any", returns_scalar=True)

    def all(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("all", returns_scalar=True)

    def max(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("max", returns_scalar=True)

    def mix(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("min", returns_scalar=True)

    def arg_min(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("arg_min", returns_scalar=True)

    def arg_max(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("arg_max", returns_scalar=True)

    # Other

    def clip(
        self, lower_bound: Any, upper_bound: Any
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "clip", lower_bound=lower_bound, upper_bound=upper_bound
        )

    def is_null(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("is_null")

    def is_nan(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("is_nan")

    def fill_null(
        self,
        value: Any | None,
        strategy: Literal["forward", "backward"] | None,
        limit: int | None,
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "fill_null", value=value, strategy=strategy, limit=limit
        )

    def is_in(self, other: Any) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("is_in", other="other")

    def arg_true(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("arg_true")

    # NOTE: `ewm_mean` not implemented `pyarrow`

    def filter(self, *predicates: Self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        plx = self.__narwhals_namespace__()
        other = plx.all_horizontal(*predicates)
        return self._reuse_series_implementation("filter", other=other)

    def drop_nulls(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("drop_nulls")

    def replace_strict(
        self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any],
        *,
        return_dtype: DType | type[DType] | None,
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "replace_strict", old=old, new=new, return_dtype=return_dtype
        )

    def sort(
        self, *, descending: bool, nulls_last: bool
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "sort", descending=descending, nulls_last=nulls_last
        )

    def abs(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("abs")

    def unique(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("unique", maintain_order=False)

    def diff(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("diff")

    # NOTE: `shift` differs

    def sample(
        self,
        n: int | None,
        *,
        fraction: float | None,
        with_replacement: bool,
        seed: int | None,
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
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

    # NOTE: `over` differs

    def is_unique(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("is_unique")

    def is_first_distinct(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("is_first_distinct")

    def is_last_distinct(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("is_last_distinct")

    def quantile(
        self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "quantile",
            quantile=quantile,
            interpolation=interpolation,
            returns_scalar=True,
        )

    def head(self, n: int) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("head", n=n)

    def tail(self, n: int) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("tail", n=n)

    def round(self, decimals: int) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("round", decimals=decimals)

    def len(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("len", returns_scalar=True)

    def gather_every(
        self, n: int, offset: int
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("gather_every", n=n, offset=offset)

    def mode(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("mode")

    # NOTE: `map_batches` differs

    def is_finite(self) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation("is_finite")

    # NOTE: `cum_(sum|count|min|max|prod)` differ

    def rolling_mean(
        self, window_size: int, *, min_samples: int | None, center: bool
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "rolling_mean",
            window_size=window_size,
            min_samples=min_samples,
            center=center,
        )

    def rolling_std(
        self, window_size: int, *, min_samples: int | None, center: bool, ddof: int
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "rolling_std",
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            ddof=ddof,
        )

    def rolling_sum(
        self, window_size: int, *, min_samples: int | None, center: bool
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "rolling_sum", window_size=window_size, min_samples=min_samples, center=center
        )

    def rolling_var(
        self, window_size: int, *, min_samples: int | None, center: bool, ddof: int
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]:
        return self._reuse_series_implementation(
            "rolling_var",
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            ddof=ddof,
        )

    # NOTE: `rank` differs

    # NOTE: All namespaces differ


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
