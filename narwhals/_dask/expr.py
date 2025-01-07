from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import NoReturn
from typing import Sequence

from narwhals._dask.utils import add_row_index
from narwhals._dask.utils import binary_operation_returns_scalar
from narwhals._dask.utils import maybe_evaluate
from narwhals._dask.utils import narwhals_to_native_dtype
from narwhals._expression_parsing import infer_new_root_output_names
from narwhals._pandas_like.utils import calculate_timestamp_date
from narwhals._pandas_like.utils import calculate_timestamp_datetime
from narwhals._pandas_like.utils import native_to_narwhals_dtype
from narwhals.exceptions import ColumnNotFoundError
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import CompliantExpr
from narwhals.utils import Implementation
from narwhals.utils import generate_temporary_column_name
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    import dask_expr
    from typing_extensions import Self

    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals._dask.namespace import DaskNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class DaskExpr(CompliantExpr["dask_expr.Series"]):
    _implementation: Implementation = Implementation.DASK

    def __init__(
        self,
        call: Callable[[DaskLazyFrame], Sequence[dask_expr.Series]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        # Whether the expression is a length-1 Series resulting from
        # a reduction, such as `nw.col('a').sum()`
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names
        self._returns_scalar = returns_scalar
        self._backend_version = backend_version
        self._version = version
        self._kwargs = kwargs

    def __call__(self, df: DaskLazyFrame) -> Sequence[dask_expr.Series]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DaskNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._dask.namespace import DaskNamespace

        return DaskNamespace(backend_version=self._backend_version, version=self._version)

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            try:
                return [df._native_frame[column_name] for column_name in column_names]
            except KeyError as e:
                missing_columns = [x for x in column_names if x not in df.columns]
                raise ColumnNotFoundError.from_missing_and_available_column_names(
                    missing_columns=missing_columns,
                    available_columns=df.columns,
                ) from e

        return cls(
            func,
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
            kwargs={},
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            return [
                df._native_frame.iloc[:, column_index] for column_index in column_indices
            ]

        return cls(
            func,
            depth=0,
            function_name="nth",
            root_names=None,
            output_names=None,
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
            kwargs={},
        )

    def _from_call(
        self,
        # First argument to `call` should be `dask_expr.Series`
        call: Callable[..., dask_expr.Series],
        expr_name: str,
        *,
        returns_scalar: bool,
        **kwargs: Any,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            results = []
            inputs = self._call(df)
            _kwargs = {key: maybe_evaluate(df, value) for key, value in kwargs.items()}
            for _input in inputs:
                name = _input.name
                if self._returns_scalar:
                    _input = _input[0]
                result = call(_input, **_kwargs)
                if returns_scalar:
                    result = result.to_series()
                result = result.rename(name)
                results.append(result)
            return results

        root_names, output_names = infer_new_root_output_names(self, **kwargs)

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{expr_name}",
            root_names=root_names,
            output_names=output_names,
            returns_scalar=returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={**self._kwargs, **kwargs},
        )

    def alias(self, name: str) -> Self:
        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            inputs = self._call(df)
            return [_input.rename(name) for _input in inputs]

        return self.__class__(
            func,
            depth=self._depth,
            function_name=self._function_name,
            root_names=self._root_names,
            output_names=[name],
            returns_scalar=self._returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={**self._kwargs, "name": name},
        )

    def __add__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__add__(other),
            "__add__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __sub__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__sub__(other),
            "__sub__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __mul__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mul__(other),
            "__mul__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __truediv__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__truediv__(other),
            "__truediv__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __floordiv__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__floordiv__(other),
            "__floordiv__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __pow__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__pow__(other),
            "__pow__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __mod__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mod__(other),
            "__mod__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __eq__(self, other: DaskExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input.__eq__(other),
            "__eq__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __ne__(self, other: DaskExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input.__ne__(other),
            "__ne__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __ge__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__ge__(other),
            "__ge__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __gt__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__gt__(other),
            "__gt__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __le__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__le__(other),
            "__le__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __lt__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__lt__(other),
            "__lt__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __and__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__and__(other),
            "__and__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __or__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__or__(other),
            "__or__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __invert__(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.__invert__(),
            "__invert__",
            returns_scalar=self._returns_scalar,
        )

    def mean(self) -> Self:
        return self._from_call(
            lambda _input: _input.mean(),
            "mean",
            returns_scalar=True,
        )

    def median(self) -> Self:
        from narwhals.exceptions import InvalidOperationError

        def func(s: dask_expr.Series) -> dask_expr.Series:
            dtype = native_to_narwhals_dtype(s, self._version, Implementation.DASK)
            if not dtype.is_numeric():
                msg = "`median` operation not supported for non-numeric input type."
                raise InvalidOperationError(msg)
            return s.median_approximate()

        return self._from_call(func, "median", returns_scalar=True)

    def min(self) -> Self:
        return self._from_call(
            lambda _input: _input.min(),
            "min",
            returns_scalar=True,
        )

    def max(self) -> Self:
        return self._from_call(
            lambda _input: _input.max(),
            "max",
            returns_scalar=True,
        )

    def std(self, ddof: int) -> Self:
        return self._from_call(
            lambda _input, ddof: _input.std(ddof=ddof),
            "std",
            ddof=ddof,
            returns_scalar=True,
        )

    def var(self, ddof: int) -> Self:
        return self._from_call(
            lambda _input, ddof: _input.var(ddof=ddof),
            "var",
            ddof=ddof,
            returns_scalar=True,
        )

    def skew(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.skew(),
            "skew",
            returns_scalar=True,
        )

    def shift(self, n: int) -> Self:
        return self._from_call(
            lambda _input, n: _input.shift(n),
            "shift",
            n=n,
            returns_scalar=self._returns_scalar,
        )

    def cum_sum(self: Self, *, reverse: bool) -> Self:
        if reverse:
            msg = "`cum_sum(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: _input.cumsum(),
            "cum_sum",
            returns_scalar=self._returns_scalar,
        )

    def cum_count(self: Self, *, reverse: bool) -> Self:
        if reverse:
            msg = "`cum_count(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: (~_input.isna()).astype(int).cumsum(),
            "cum_count",
            returns_scalar=self._returns_scalar,
        )

    def cum_min(self: Self, *, reverse: bool) -> Self:
        if reverse:
            msg = "`cum_min(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: _input.cummin(),
            "cum_min",
            returns_scalar=self._returns_scalar,
        )

    def cum_max(self: Self, *, reverse: bool) -> Self:
        if reverse:
            msg = "`cum_max(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: _input.cummax(),
            "cum_max",
            returns_scalar=self._returns_scalar,
        )

    def cum_prod(self: Self, *, reverse: bool) -> Self:
        if reverse:
            msg = "`cum_prod(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: _input.cumprod(),
            "cum_prod",
            returns_scalar=self._returns_scalar,
        )

    def is_between(
        self,
        lower_bound: Self | Any,
        upper_bound: Self | Any,
        closed: str = "both",
    ) -> Self:
        if closed == "none":
            closed = "neither"
        return self._from_call(
            lambda _input, lower_bound, upper_bound, closed: _input.between(
                lower_bound,
                upper_bound,
                closed,
            ),
            "is_between",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            closed=closed,
            returns_scalar=self._returns_scalar,
        )

    def sum(self) -> Self:
        return self._from_call(
            lambda _input: _input.sum(),
            "sum",
            returns_scalar=True,
        )

    def count(self) -> Self:
        return self._from_call(
            lambda _input: _input.count(),
            "count",
            returns_scalar=True,
        )

    def round(self, decimals: int) -> Self:
        return self._from_call(
            lambda _input, decimals: _input.round(decimals),
            "round",
            decimals=decimals,
            returns_scalar=self._returns_scalar,
        )

    def ewm_mean(
        self: Self,
        *,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_periods: int = 1,
        ignore_nulls: bool = False,
    ) -> NoReturn:
        msg = "`Expr.ewm_mean` is not supported for the Dask backend"
        raise NotImplementedError(msg)

    def unique(self) -> NoReturn:
        # We can't (yet?) allow methods which modify the index
        msg = "`Expr.unique` is not supported for the Dask backend. Please use `LazyFrame.unique` instead."
        raise NotImplementedError(msg)

    def drop_nulls(self) -> NoReturn:
        # We can't (yet?) allow methods which modify the index
        msg = "`Expr.drop_nulls` is not supported for the Dask backend. Please use `LazyFrame.drop_nulls` instead."
        raise NotImplementedError(msg)

    def head(self) -> NoReturn:
        # We can't (yet?) allow methods which modify the index
        msg = "`Expr.head` is not supported for the Dask backend. Please use `LazyFrame.head` instead."
        raise NotImplementedError(msg)

    def replace_strict(
        self, old: Sequence[Any], new: Sequence[Any], *, return_dtype: DType | None
    ) -> Self:
        msg = "`replace_strict` is not yet supported for Dask expressions"
        raise NotImplementedError(msg)

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> NoReturn:
        # We can't (yet?) allow methods which modify the index
        msg = "`Expr.sort` is not supported for the Dask backend. Please use `LazyFrame.sort` instead."
        raise NotImplementedError(msg)

    def abs(self) -> Self:
        return self._from_call(
            lambda _input: _input.abs(),
            "abs",
            returns_scalar=self._returns_scalar,
        )

    def all(self) -> Self:
        return self._from_call(
            lambda _input: _input.all(
                axis=None, skipna=True, split_every=False, out=None
            ),
            "all",
            returns_scalar=True,
        )

    def any(self) -> Self:
        return self._from_call(
            lambda _input: _input.any(axis=0, skipna=True, split_every=False),
            "any",
            returns_scalar=True,
        )

    def fill_null(
        self: Self,
        value: Any | None = None,
        strategy: Literal["forward", "backward"] | None = None,
        limit: int | None = None,
    ) -> DaskExpr:
        def func(
            _input: dask_expr.Series,
            value: Any | None,
            strategy: str | None,
            limit: int | None,
        ) -> dask_expr.Series:
            if value is not None:
                res_ser = _input.fillna(value)
            else:
                res_ser = (
                    _input.ffill(limit=limit)
                    if strategy == "forward"
                    else _input.bfill(limit=limit)
                )
            return res_ser

        return self._from_call(
            func,
            "fillna",
            value=value,
            strategy=strategy,
            limit=limit,
            returns_scalar=self._returns_scalar,
        )

    def clip(
        self: Self,
        lower_bound: Self | Any | None,
        upper_bound: Self | Any | None,
    ) -> Self:
        return self._from_call(
            lambda _input, lower_bound, upper_bound: _input.clip(
                lower=lower_bound, upper=upper_bound
            ),
            "clip",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            returns_scalar=self._returns_scalar,
        )

    def diff(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.diff(),
            "diff",
            returns_scalar=self._returns_scalar,
        )

    def n_unique(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.nunique(dropna=False),
            "n_unique",
            returns_scalar=True,
        )

    def is_null(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.isna(),
            "is_null",
            returns_scalar=self._returns_scalar,
        )

    def is_nan(self: Self) -> Self:
        def func(_input: dask_expr.Series) -> dask_expr.Series:
            dtype = native_to_narwhals_dtype(_input, self._version, self._implementation)
            if dtype.is_numeric():
                return _input != _input  # noqa: PLR0124
            msg = f"`.is_nan` only supported for numeric dtypes and not {dtype}, did you mean `.is_null`?"
            raise InvalidOperationError(msg)

        return self._from_call(
            func,
            "is_null",
            returns_scalar=self._returns_scalar,
        )

    def len(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.size,
            "len",
            returns_scalar=True,
        )

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        if interpolation == "linear":

            def func(_input: dask_expr.Series, quantile: float) -> dask_expr.Series:
                if _input.npartitions > 1:
                    msg = "`Expr.quantile` is not supported for Dask backend with multiple partitions."
                    raise NotImplementedError(msg)
                return _input.quantile(q=quantile, method="dask")  # pragma: no cover

            return self._from_call(
                func,
                "quantile",
                quantile=quantile,
                returns_scalar=True,
            )
        else:
            msg = "`higher`, `lower`, `midpoint`, `nearest` - interpolation methods are not supported by Dask. Please use `linear` instead."
            raise NotImplementedError(msg)

    def is_first_distinct(self: Self) -> Self:
        def func(_input: dask_expr.Series) -> dask_expr.Series:
            _name = _input.name
            col_token = generate_temporary_column_name(n_bytes=8, columns=[_name])
            _input = add_row_index(
                _input.to_frame(),
                col_token,
                backend_version=self._backend_version,
                implementation=self._implementation,
            )
            first_distinct_index = _input.groupby(_name).agg({col_token: "min"})[
                col_token
            ]

            return _input[col_token].isin(first_distinct_index)

        return self._from_call(
            func,
            "is_first_distinct",
            returns_scalar=self._returns_scalar,
        )

    def is_last_distinct(self: Self) -> Self:
        def func(_input: dask_expr.Series) -> dask_expr.Series:
            _name = _input.name
            col_token = generate_temporary_column_name(n_bytes=8, columns=[_name])
            _input = add_row_index(
                _input.to_frame(),
                col_token,
                backend_version=self._backend_version,
                implementation=self._implementation,
            )
            last_distinct_index = _input.groupby(_name).agg({col_token: "max"})[col_token]

            return _input[col_token].isin(last_distinct_index)

        return self._from_call(
            func,
            "is_last_distinct",
            returns_scalar=self._returns_scalar,
        )

    def is_duplicated(self: Self) -> Self:
        def func(_input: dask_expr.Series) -> dask_expr.Series:
            _name = _input.name
            return (
                _input.to_frame()
                .groupby(_name, dropna=False)
                .transform("size", meta=(_name, int))
                > 1
            )

        return self._from_call(
            func,
            "is_duplicated",
            returns_scalar=self._returns_scalar,
        )

    def is_unique(self: Self) -> Self:
        def func(_input: dask_expr.Series) -> dask_expr.Series:
            _name = _input.name
            return (
                _input.to_frame()
                .groupby(_name, dropna=False)
                .transform("size", meta=(_name, int))
                == 1
            )

        return self._from_call(
            func,
            "is_unique",
            returns_scalar=self._returns_scalar,
        )

    def is_in(self: Self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.isin(other),
            "is_in",
            other=other,
            returns_scalar=self._returns_scalar,
        )

    def null_count(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.isna().sum(),
            "null_count",
            returns_scalar=True,
        )

    def tail(self: Self) -> NoReturn:
        # We can't (yet?) allow methods which modify the index
        msg = "`Expr.tail` is not supported for the Dask backend. Please use `LazyFrame.tail` instead."
        raise NotImplementedError(msg)

    def gather_every(self: Self, n: int, offset: int = 0) -> NoReturn:
        # We can't (yet?) allow methods which modify the index
        msg = "`Expr.gather_every` is not supported for the Dask backend. Please use `LazyFrame.gather_every` instead."
        raise NotImplementedError(msg)

    def over(self: Self, keys: list[str]) -> Self:
        def func(df: DaskLazyFrame) -> list[Any]:
            if self._output_names is None:
                msg = (
                    "Anonymous expressions are not supported in over.\n"
                    "Instead of `nw.all()`, try using a named expression, such as "
                    "`nw.col('a', 'b')`\n"
                )
                raise ValueError(msg)

            if df._native_frame.npartitions == 1:  # pragma: no cover
                tmp = df.group_by(*keys, drop_null_keys=False).agg(self)
                tmp_native = (
                    df.select(*keys)
                    .join(tmp, how="left", left_on=keys, right_on=keys, suffix="_right")
                    ._native_frame
                )
                return [tmp_native[name] for name in self._output_names]
            msg = (
                "`Expr.over` is not supported for Dask backend with multiple partitions."
            )
            raise NotImplementedError(msg)

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->over",
            root_names=self._root_names,
            output_names=self._output_names,
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={**self._kwargs, "keys": keys},
        )

    @property
    def str(self: Self) -> DaskExprStringNamespace:
        return DaskExprStringNamespace(self)

    @property
    def dt(self: Self) -> DaskExprDateTimeNamespace:
        return DaskExprDateTimeNamespace(self)

    @property
    def name(self: Self) -> DaskExprNameNamespace:
        return DaskExprNameNamespace(self)

    def cast(
        self: Self,
        dtype: DType | type[DType],
    ) -> Self:
        def func(_input: Any, dtype: DType | type[DType]) -> Any:
            dtype = narwhals_to_native_dtype(dtype, self._version)
            return _input.astype(dtype)

        return self._from_call(
            func,
            "cast",
            dtype=dtype,
            returns_scalar=self._returns_scalar,
        )

    def is_finite(self: Self) -> Self:
        import dask.array as da

        return self._from_call(
            lambda _input: da.isfinite(_input),
            "is_finite",
            returns_scalar=self._returns_scalar,
        )


class DaskExprStringNamespace:
    def __init__(self, expr: DaskExpr) -> None:
        self._compliant_expr = expr

    def len_chars(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.str.len(),
            "len",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def replace(
        self,
        pattern: str,
        value: str,
        *,
        literal: bool = False,
        n: int = 1,
    ) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input, pattern, value, literal, n: _input.str.replace(
                pattern, value, regex=not literal, n=n
            ),
            "replace",
            pattern=pattern,
            value=value,
            literal=literal,
            n=n,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def replace_all(
        self,
        pattern: str,
        value: str,
        *,
        literal: bool = False,
    ) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input, pattern, value, literal: _input.str.replace(
                pattern, value, n=-1, regex=not literal
            ),
            "replace",
            pattern=pattern,
            value=value,
            literal=literal,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def strip_chars(self, characters: str | None = None) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input, characters: _input.str.strip(characters),
            "strip",
            characters=characters,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def starts_with(self, prefix: str) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input, prefix: _input.str.startswith(prefix),
            "starts_with",
            prefix=prefix,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def ends_with(self, suffix: str) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input, suffix: _input.str.endswith(suffix),
            "ends_with",
            suffix=suffix,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def contains(self, pattern: str, *, literal: bool = False) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input, pattern, literal: _input.str.contains(
                pat=pattern, regex=not literal
            ),
            "contains",
            pattern=pattern,
            literal=literal,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def slice(self, offset: int, length: int | None = None) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input, offset, length: _input.str.slice(
                start=offset, stop=offset + length if length else None
            ),
            "slice",
            offset=offset,
            length=length,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def to_datetime(self: Self, format: str | None) -> DaskExpr:  # noqa: A002
        import dask.dataframe as dd

        return self._compliant_expr._from_call(
            lambda _input, format: dd.to_datetime(_input, format=format),
            "to_datetime",
            format=format,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def to_uppercase(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.str.upper(),
            "to_uppercase",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def to_lowercase(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.str.lower(),
            "to_lowercase",
            returns_scalar=self._compliant_expr._returns_scalar,
        )


class DaskExprDateTimeNamespace:
    def __init__(self, expr: DaskExpr) -> None:
        self._compliant_expr = expr

    def date(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.date,
            "date",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def year(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.year,
            "year",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def month(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.month,
            "month",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def day(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.day,
            "day",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def hour(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.hour,
            "hour",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def minute(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.minute,
            "minute",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def second(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.second,
            "second",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def millisecond(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.microsecond // 1000,
            "millisecond",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def microsecond(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.microsecond,
            "microsecond",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def nanosecond(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.microsecond * 1000 + _input.dt.nanosecond,
            "nanosecond",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def ordinal_day(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.dayofyear,
            "ordinal_day",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def weekday(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.weekday + 1,  # Dask is 0-6
            "weekday",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def to_string(self, format: str) -> DaskExpr:  # noqa: A002
        return self._compliant_expr._from_call(
            lambda _input, format: _input.dt.strftime(format.replace("%.f", ".%f")),
            "strftime",
            format=format,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def replace_time_zone(self, time_zone: str | None) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input, time_zone: _input.dt.tz_localize(None).dt.tz_localize(
                time_zone
            )
            if time_zone is not None
            else _input.dt.tz_localize(None),
            "tz_localize",
            time_zone=time_zone,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def convert_time_zone(self, time_zone: str) -> DaskExpr:
        def func(s: dask_expr.Series, time_zone: str) -> dask_expr.Series:
            dtype = native_to_narwhals_dtype(
                s, self._compliant_expr._version, Implementation.DASK
            )
            if dtype.time_zone is None:  # type: ignore[attr-defined]
                return s.dt.tz_localize("UTC").dt.tz_convert(time_zone)
            else:
                return s.dt.tz_convert(time_zone)

        return self._compliant_expr._from_call(
            func,
            "tz_convert",
            time_zone=time_zone,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def timestamp(self, time_unit: Literal["ns", "us", "ms"] = "us") -> DaskExpr:
        def func(
            s: dask_expr.Series, time_unit: Literal["ns", "us", "ms"] = "us"
        ) -> dask_expr.Series:
            dtype = native_to_narwhals_dtype(
                s, self._compliant_expr._version, Implementation.DASK
            )
            is_pyarrow_dtype = "pyarrow" in str(dtype)
            mask_na = s.isna()
            dtypes = import_dtypes_module(self._compliant_expr._version)
            if dtype == dtypes.Date:
                # Date is only supported in pandas dtypes if pyarrow-backed
                s_cast = s.astype("Int32[pyarrow]")
                result = calculate_timestamp_date(s_cast, time_unit)
            elif dtype == dtypes.Datetime:
                original_time_unit = dtype.time_unit  # type: ignore[attr-defined]
                s_cast = (
                    s.astype("Int64[pyarrow]") if is_pyarrow_dtype else s.astype("int64")
                )
                result = calculate_timestamp_datetime(
                    s_cast, original_time_unit, time_unit
                )
            else:
                msg = "Input should be either of Date or Datetime type"
                raise TypeError(msg)
            return result.where(~mask_na)

        return self._compliant_expr._from_call(
            func,
            "datetime",
            time_unit=time_unit,
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def total_minutes(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.total_seconds() // 60,
            "total_minutes",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def total_seconds(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.total_seconds() // 1,
            "total_seconds",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def total_milliseconds(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.total_seconds() * 1000 // 1,
            "total_milliseconds",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def total_microseconds(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.total_seconds() * 1_000_000 // 1,
            "total_microseconds",
            returns_scalar=self._compliant_expr._returns_scalar,
        )

    def total_nanoseconds(self) -> DaskExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.dt.total_seconds() * 1_000_000_000 // 1,
            "total_nanoseconds",
            returns_scalar=self._compliant_expr._returns_scalar,
        )


class DaskExprNameNamespace:
    def __init__(self: Self, expr: DaskExpr) -> None:
        self._compliant_expr = expr

    def keep(self: Self) -> DaskExpr:
        root_names = self._compliant_expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.keep`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        return self._compliant_expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._compliant_expr._call(df), root_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=root_names,
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )

    def map(self: Self, function: Callable[[str], str]) -> DaskExpr:
        root_names = self._compliant_expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.map`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        output_names = [function(str(name)) for name in root_names]

        return self._compliant_expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._compliant_expr._call(df), output_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "function": function},
        )

    def prefix(self: Self, prefix: str) -> DaskExpr:
        root_names = self._compliant_expr._root_names
        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.prefix`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        output_names = [prefix + str(name) for name in root_names]
        return self._compliant_expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._compliant_expr._call(df), output_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "prefix": prefix},
        )

    def suffix(self: Self, suffix: str) -> DaskExpr:
        root_names = self._compliant_expr._root_names
        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.suffix`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        output_names = [str(name) + suffix for name in root_names]

        return self._compliant_expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._compliant_expr._call(df), output_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs={**self._compliant_expr._kwargs, "suffix": suffix},
        )

    def to_lowercase(self: Self) -> DaskExpr:
        root_names = self._compliant_expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.to_lowercase`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)
        output_names = [str(name).lower() for name in root_names]

        return self._compliant_expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._compliant_expr._call(df), output_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )

    def to_uppercase(self: Self) -> DaskExpr:
        root_names = self._compliant_expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.to_uppercase`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)
        output_names = [str(name).upper() for name in root_names]

        return self._compliant_expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._compliant_expr._call(df), output_names)
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._compliant_expr._returns_scalar,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            kwargs=self._compliant_expr._kwargs,
        )
