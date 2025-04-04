from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Mapping
from typing import Sequence

import polars as pl

from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import extract_native
from narwhals._polars.utils import narwhals_to_native_dtype
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._expression_parsing import ExprKind
    from narwhals._expression_parsing import ExprMetadata
    from narwhals._polars.dataframe import Method
    from narwhals._polars.namespace import PolarsNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class PolarsExpr:
    def __init__(
        self: Self, expr: pl.Expr, version: Version, backend_version: tuple[int, ...]
    ) -> None:
        self._native_expr = expr
        self._implementation = Implementation.POLARS
        self._version = version
        self._backend_version = backend_version
        self._metadata: ExprMetadata | None = None

    @property
    def native(self) -> pl.Expr:
        return self._native_expr

    def __repr__(self: Self) -> str:  # pragma: no cover
        return "PolarsExpr"

    def _with_native(self: Self, expr: pl.Expr) -> Self:
        return self.__class__(expr, self._version, self._backend_version)

    @classmethod
    def _from_series(cls, series: Any) -> Self:
        return cls(series.native, series._version, series._backend_version)

    def broadcast(self, kind: Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]) -> Self:
        # Let Polars do its thing.
        return self

    def __getattr__(self: Self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._with_native(getattr(self.native, attr)(*pos, **kwds))

        return func

    def _renamed_min_periods(self, min_samples: int, /) -> dict[str, Any]:
        name = "min_periods" if self._backend_version < (1, 21, 0) else "min_samples"
        return {name: min_samples}

    def cast(self, dtype: DType | type[DType]) -> Self:
        dtype_pl = narwhals_to_native_dtype(dtype, self._version, self._backend_version)
        return self._with_native(self.native.cast(dtype_pl))

    def ewm_mean(
        self: Self,
        *,
        com: float | None,
        span: float | None,
        half_life: float | None,
        alpha: float | None,
        adjust: bool,
        min_samples: int,
        ignore_nulls: bool,
    ) -> Self:
        native = self.native.ewm_mean(
            com=com,
            span=span,
            half_life=half_life,
            alpha=alpha,
            adjust=adjust,
            ignore_nulls=ignore_nulls,
            **self._renamed_min_periods(min_samples),
        )
        if self._backend_version < (1,):  # pragma: no cover
            native = pl.when(~self.native.is_null()).then(native).otherwise(None)
        return self._with_native(native)

    def is_nan(self: Self) -> Self:
        if self._backend_version >= (1, 18):
            native = self.native.is_nan()
        else:  # pragma: no cover
            native = pl.when(self.native.is_not_null()).then(self.native.is_nan())
        return self._with_native(native)

    def over(self, partition_by: Sequence[str], order_by: Sequence[str] | None) -> Self:
        if self._backend_version < (1, 9):
            if order_by:
                msg = "`order_by` in Polars requires version 1.10 or greater"
                raise NotImplementedError(msg)
            native = self.native.over(partition_by or pl.lit(1))
        else:
            native = self.native.over(partition_by or pl.lit(1), order_by=order_by)
        return self._with_native(native)

    def rolling_var(
        self: Self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        if self._backend_version < (1,):  # pragma: no cover
            msg = "`rolling_var` not implemented for polars older than 1.0"
            raise NotImplementedError(msg)
        kwds = self._renamed_min_periods(min_samples)
        native = self.native.rolling_var(
            window_size=window_size, center=center, ddof=ddof, **kwds
        )
        return self._with_native(native)

    def rolling_std(
        self: Self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        if self._backend_version < (1,):  # pragma: no cover
            msg = "`rolling_std` not implemented for polars older than 1.0"
            raise NotImplementedError(msg)
        kwds = self._renamed_min_periods(min_samples)
        native = self.native.rolling_std(
            window_size=window_size, center=center, ddof=ddof, **kwds
        )
        return self._with_native(native)

    def rolling_sum(
        self: Self, window_size: int, *, min_samples: int, center: bool
    ) -> Self:
        kwds = self._renamed_min_periods(min_samples)
        native = self.native.rolling_sum(window_size=window_size, center=center, **kwds)
        return self._with_native(native)

    def rolling_mean(
        self: Self, window_size: int, *, min_samples: int, center: bool
    ) -> Self:
        kwds = self._renamed_min_periods(min_samples)
        native = self.native.rolling_mean(window_size=window_size, center=center, **kwds)
        return self._with_native(native)

    def map_batches(
        self, function: Callable[[Any], Any], return_dtype: DType | type[DType] | None
    ) -> Self:
        return_dtype_pl = (
            narwhals_to_native_dtype(return_dtype, self._version, self._backend_version)
            if return_dtype
            else None
        )
        native = self.native.map_batches(function, return_dtype_pl)
        return self._with_native(native)

    def replace_strict(
        self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any],
        *,
        return_dtype: DType | type[DType] | None,
    ) -> Self:
        if self._backend_version < (1,):
            msg = f"`replace_strict` is only available in Polars>=1.0, found version {self._backend_version}"
            raise NotImplementedError(msg)
        return_dtype_pl = (
            narwhals_to_native_dtype(return_dtype, self._version, self._backend_version)
            if return_dtype
            else None
        )
        native = self.native.replace_strict(old, new, return_dtype=return_dtype_pl)
        return self._with_native(native)

    def __eq__(self: Self, other: object) -> Self:  # type: ignore[override]
        return self._with_native(self.native.__eq__(extract_native(other)))  # type: ignore[operator]

    def __ne__(self: Self, other: object) -> Self:  # type: ignore[override]
        return self._with_native(self.native.__ne__(extract_native(other)))  # type: ignore[operator]

    def __ge__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__ge__(extract_native(other)))

    def __gt__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__gt__(extract_native(other)))

    def __le__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__le__(extract_native(other)))

    def __lt__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__lt__(extract_native(other)))

    def __and__(self: Self, other: PolarsExpr | bool | Any) -> Self:
        return self._with_native(self.native.__and__(extract_native(other)))  # type: ignore[operator]

    def __or__(self: Self, other: PolarsExpr | bool | Any) -> Self:
        return self._with_native(self.native.__or__(extract_native(other)))  # type: ignore[operator]

    def __add__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__add__(extract_native(other)))

    def __sub__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__sub__(extract_native(other)))

    def __mul__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__mul__(extract_native(other)))

    def __pow__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__pow__(extract_native(other)))

    def __truediv__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__truediv__(extract_native(other)))

    def __floordiv__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__floordiv__(extract_native(other)))

    def __mod__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__mod__(extract_native(other)))

    def __invert__(self: Self) -> Self:
        return self._with_native(self.native.__invert__())

    def cum_count(self: Self, *, reverse: bool) -> Self:
        if self._backend_version < (0, 20, 4):
            result = (~self.native.is_null()).cum_sum(reverse=reverse)
        else:
            result = self.native.cum_count(reverse=reverse)
        return self._with_native(result)

    def __narwhals_expr__(self) -> None: ...
    def __narwhals_namespace__(self) -> PolarsNamespace:  # pragma: no cover
        from narwhals._polars.namespace import PolarsNamespace

        return PolarsNamespace(
            backend_version=self._backend_version, version=self._version
        )

    @property
    def dt(self: Self) -> PolarsExprDateTimeNamespace:
        return PolarsExprDateTimeNamespace(self)

    @property
    def str(self: Self) -> PolarsExprStringNamespace:
        return PolarsExprStringNamespace(self)

    @property
    def cat(self: Self) -> PolarsExprCatNamespace:
        return PolarsExprCatNamespace(self)

    @property
    def name(self: Self) -> PolarsExprNameNamespace:
        return PolarsExprNameNamespace(self)

    @property
    def list(self: Self) -> PolarsExprListNamespace:
        return PolarsExprListNamespace(self)

    @property
    def struct(self: Self) -> PolarsExprStructNamespace:
        return PolarsExprStructNamespace(self)

    # CompliantExpr
    _alias_output_names: Any
    _evaluate_output_names: Any
    _is_multi_output_unnamed: Any
    __call__: Any
    from_column_names: Any
    from_column_indices: Any

    # Polars
    abs: Method[Self]
    all: Method[Self]
    any: Method[Self]
    alias: Method[Self]
    arg_max: Method[Self]
    arg_min: Method[Self]
    arg_true: Method[Self]
    count: Method[Self]
    cum_max: Method[Self]
    cum_min: Method[Self]
    cum_prod: Method[Self]
    cum_sum: Method[Self]
    diff: Method[Self]
    drop_nulls: Method[Self]
    fill_null: Method[Self]
    gather_every: Method[Self]
    head: Method[Self]
    is_finite: Method[Self]
    is_first_distinct: Method[Self]
    is_in: Method[Self]
    is_last_distinct: Method[Self]
    is_null: Method[Self]
    is_unique: Method[Self]
    len: Method[Self]
    max: Method[Self]
    mean: Method[Self]
    median: Method[Self]
    min: Method[Self]
    mode: Method[Self]
    n_unique: Method[Self]
    null_count: Method[Self]
    quantile: Method[Self]
    rank: Method[Self]
    round: Method[Self]
    sample: Method[Self]
    shift: Method[Self]
    skew: Method[Self]
    std: Method[Self]
    sum: Method[Self]
    sort: Method[Self]
    tail: Method[Self]
    unique: Method[Self]
    var: Method[Self]


class PolarsExprDateTimeNamespace:
    def __init__(self: Self, expr: PolarsExpr) -> None:
        self._compliant_expr = expr

    def __getattr__(self: Self, attr: str) -> Callable[[Any], PolarsExpr]:
        def func(*args: Any, **kwargs: Any) -> PolarsExpr:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._compliant_expr._with_native(
                getattr(self._compliant_expr.native.dt, attr)(*pos, **kwds)
            )

        return func


class PolarsExprStringNamespace:
    def __init__(self: Self, expr: PolarsExpr) -> None:
        self._compliant_expr = expr

    def __getattr__(self: Self, attr: str) -> Callable[[Any], PolarsExpr]:
        def func(*args: Any, **kwargs: Any) -> PolarsExpr:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._compliant_expr._with_native(
                getattr(self._compliant_expr.native.str, attr)(*pos, **kwds)
            )

        return func


class PolarsExprCatNamespace:
    def __init__(self: Self, expr: PolarsExpr) -> None:
        self._compliant_expr = expr

    def __getattr__(self: Self, attr: str) -> Callable[[Any], PolarsExpr]:
        def func(*args: Any, **kwargs: Any) -> PolarsExpr:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._compliant_expr._with_native(
                getattr(self._compliant_expr.native.cat, attr)(*pos, **kwds)
            )

        return func


class PolarsExprNameNamespace:
    def __init__(self: Self, expr: PolarsExpr) -> None:
        self._compliant_expr = expr

    def __getattr__(self: Self, attr: str) -> Callable[[Any], PolarsExpr]:
        def func(*args: Any, **kwargs: Any) -> PolarsExpr:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._compliant_expr._with_native(
                getattr(self._compliant_expr.native.name, attr)(*pos, **kwds)
            )

        return func


class PolarsExprListNamespace:
    def __init__(self: Self, expr: PolarsExpr) -> None:
        self._expr = expr

    def len(self: Self) -> PolarsExpr:
        native_expr = self._expr._native_expr
        native_result = native_expr.list.len()

        if self._expr._backend_version < (1, 16):  # pragma: no cover
            native_result = (
                pl.when(~native_expr.is_null()).then(native_result).cast(pl.UInt32())
            )
        elif self._expr._backend_version < (1, 17):  # pragma: no cover
            native_result = native_result.cast(pl.UInt32())

        return self._expr._with_native(native_result)

    # TODO(FBruzzesi): Remove `pragma: no cover` once other namespace methods are added
    def __getattr__(
        self: Self, attr: str
    ) -> Callable[[Any], PolarsExpr]:  # pragma: no cover
        def func(*args: Any, **kwargs: Any) -> PolarsExpr:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._expr._with_native(
                getattr(self._expr.native.list, attr)(*pos, **kwds)
            )

        return func


class PolarsExprStructNamespace:
    def __init__(self: Self, expr: PolarsExpr) -> None:
        self._expr = expr

    def __getattr__(
        self: Self, attr: str
    ) -> Callable[[Any], PolarsExpr]:  # pragma: no cover
        def func(*args: Any, **kwargs: Any) -> PolarsExpr:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._expr._with_native(
                getattr(self._expr.native.struct, attr)(*pos, **kwds)
            )

        return func
