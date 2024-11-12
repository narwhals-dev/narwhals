from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import NoReturn
from typing import Sequence

from narwhals._dask.utils import add_row_index
from narwhals._dask.utils import maybe_evaluate
from narwhals._dask.utils import narwhals_to_native_dtype
from narwhals._pandas_like.utils import calculate_timestamp_date
from narwhals._pandas_like.utils import calculate_timestamp_datetime
from narwhals._pandas_like.utils import native_to_narwhals_dtype
from narwhals.utils import Implementation
from narwhals.utils import generate_temporary_column_name

if TYPE_CHECKING:
    import dask_expr
    from typing_extensions import Self

    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals._dask.namespace import DaskNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes


class DaskExpr:
    def __init__(
        self,
        call: Callable[[DaskLazyFrame], list[dask_expr.Series]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        # Whether the expression is a length-1 Series resulting from
        # a reduction, such as `nw.col('a').sum()`
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names
        self._returns_scalar = returns_scalar
        self._backend_version = backend_version
        self._dtypes = dtypes

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DaskNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._dask.namespace import DaskNamespace

        return DaskNamespace(backend_version=self._backend_version, dtypes=self._dtypes)

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            return [df._native_frame[column_name] for column_name in column_names]

        return cls(
            func,
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            returns_scalar=False,
            backend_version=backend_version,
            dtypes=dtypes,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
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
            dtypes=dtypes,
        )

    def _from_call(
        self,
        # First argument to `call` should be `dask_expr.Series`
        call: Callable[..., dask_expr.Series],
        expr_name: str,
        *args: Any,
        returns_scalar: bool,
        **kwargs: Any,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            results = []
            inputs = self._call(df)
            _args = [maybe_evaluate(df, x) for x in args]
            _kwargs = {key: maybe_evaluate(df, value) for key, value in kwargs.items()}
            for _input in inputs:
                result = call(_input, *_args, **_kwargs)
                if returns_scalar:
                    result = result.to_series()
                result = result.rename(_input.name)
                results.append(result)
            return results

        # Try tracking root and output names by combining them from all
        # expressions appearing in args and kwargs. If any anonymous
        # expression appears (e.g. nw.all()), then give up on tracking root names
        # and just set it to None.
        root_names = copy(self._root_names)
        output_names = self._output_names
        for arg in list(args) + list(kwargs.values()):
            if root_names is not None and isinstance(arg, self.__class__):
                if arg._root_names is not None:
                    root_names.extend(arg._root_names)
                else:
                    root_names = None
                    output_names = None
                    break
            elif root_names is None:
                output_names = None
                break

        if not (
            (output_names is None and root_names is None)
            or (output_names is not None and root_names is not None)
        ):  # pragma: no cover
            msg = "Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues"
            raise AssertionError(msg)

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{expr_name}",
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._returns_scalar or returns_scalar,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
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
            dtypes=self._dtypes,
        )

    def __add__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__add__(other),
            "__add__",
            other,
            returns_scalar=False,
        )

    def __radd__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__radd__(other),
            "__radd__",
            other,
            returns_scalar=False,
        )

    def __sub__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__sub__(other),
            "__sub__",
            other,
            returns_scalar=False,
        )

    def __rsub__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rsub__(other),
            "__rsub__",
            other,
            returns_scalar=False,
        )

    def __mul__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mul__(other),
            "__mul__",
            other,
            returns_scalar=False,
        )

    def __rmul__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rmul__(other),
            "__rmul__",
            other,
            returns_scalar=False,
        )

    def __truediv__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__truediv__(other),
            "__truediv__",
            other,
            returns_scalar=False,
        )

    def __rtruediv__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rtruediv__(other),
            "__rtruediv__",
            other,
            returns_scalar=False,
        )

    def __floordiv__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__floordiv__(other),
            "__floordiv__",
            other,
            returns_scalar=False,
        )

    def __rfloordiv__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rfloordiv__(other),
            "__rfloordiv__",
            other,
            returns_scalar=False,
        )

    def __pow__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__pow__(other),
            "__pow__",
            other,
            returns_scalar=False,
        )

    def __rpow__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rpow__(other),
            "__rpow__",
            other,
            returns_scalar=False,
        )

    def __mod__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mod__(other),
            "__mod__",
            other,
            returns_scalar=False,
        )

    def __rmod__(self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__rmod__(other),
            "__rmod__",
            other,
            returns_scalar=False,
        )

    def __eq__(self, other: DaskExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input.__eq__(other),
            "__eq__",
            other,
            returns_scalar=False,
        )

    def __ne__(self, other: DaskExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input.__ne__(other),
            "__ne__",
            other,
            returns_scalar=False,
        )

    def __ge__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__ge__(other),
            "__ge__",
            other,
            returns_scalar=False,
        )

    def __gt__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__gt__(other),
            "__gt__",
            other,
            returns_scalar=False,
        )

    def __le__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__le__(other),
            "__le__",
            other,
            returns_scalar=False,
        )

    def __lt__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__lt__(other),
            "__lt__",
            other,
            returns_scalar=False,
        )

    def __and__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__and__(other),
            "__and__",
            other,
            returns_scalar=False,
        )

    def __rand__(self, other: DaskExpr) -> Self:  # pragma: no cover
        return self._from_call(
            lambda _input, other: _input.__rand__(other),
            "__rand__",
            other,
            returns_scalar=False,
        )

    def __or__(self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__or__(other),
            "__or__",
            other,
            returns_scalar=False,
        )

    def __ror__(self, other: DaskExpr) -> Self:  # pragma: no cover
        return self._from_call(
            lambda _input, other: _input.__ror__(other),
            "__ror__",
            other,
            returns_scalar=False,
        )

    def __invert__(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.__invert__(),
            "__invert__",
            returns_scalar=False,
        )

    def map_batches(
        self: Self,
        function: Callable[[Any], Any],
        return_dtype: DType | None = None,
    ) -> NoReturn:
        msg = "`Expr.map_batches` is not implemented for Dask yet"
        raise NotImplementedError(msg)

    def mean(self) -> Self:
        return self._from_call(
            lambda _input: _input.mean(),
            "mean",
            returns_scalar=True,
        )

    def median(self) -> Self:
        from dask_expr._shuffle import _is_numeric_cast_type

        from narwhals._exceptions import InvalidOperationError

        def func(_input: dask_expr.Series) -> dask_expr.Series:
            if not _is_numeric_cast_type(_input.dtype):
                msg = "`median` operation not supported for non-numeric input type."
                raise InvalidOperationError(msg)
            return _input.median_approximate()

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

    def std(self, ddof: int = 1) -> Self:
        return self._from_call(
            lambda _input, ddof: _input.std(ddof=ddof),
            "std",
            ddof,
            returns_scalar=True,
        )

    def shift(self, n: int) -> Self:
        return self._from_call(
            lambda _input, n: _input.shift(n),
            "shift",
            n,
            returns_scalar=False,
        )

    def cum_sum(self) -> Self:
        return self._from_call(
            lambda _input: _input.cumsum(),
            "cum_sum",
            returns_scalar=False,
        )

    def is_between(
        self,
        lower_bound: Any,
        upper_bound: Any,
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
            lower_bound,
            upper_bound,
            closed,
            returns_scalar=False,
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
            decimals,
            returns_scalar=False,
        )

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
            returns_scalar=False,
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
            value,
            strategy,
            limit,
            returns_scalar=False,
        )

    def clip(
        self: Self,
        lower_bound: Any | None = None,
        upper_bound: Any | None = None,
    ) -> Self:
        return self._from_call(
            lambda _input, _lower, _upper: _input.clip(lower=_lower, upper=_upper),
            "clip",
            lower_bound,
            upper_bound,
            returns_scalar=False,
        )

    def diff(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.diff(),
            "diff",
            returns_scalar=False,
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
            returns_scalar=False,
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

            def func(_input: dask_expr.Series, _quantile: float) -> dask_expr.Series:
                if _input.npartitions > 1:
                    msg = "`Expr.quantile` is not supported for Dask backend with multiple partitions."
                    raise NotImplementedError(msg)
                return _input.quantile(q=_quantile, method="dask")  # pragma: no cover

            return self._from_call(
                func,
                "quantile",
                quantile,
                returns_scalar=True,
            )
        else:
            msg = "`higher`, `lower`, `midpoint`, `nearest` - interpolation methods are not supported by Dask. Please use `linear` instead."
            raise NotImplementedError(msg)

    def is_first_distinct(self: Self) -> Self:
        def func(_input: dask_expr.Series) -> dask_expr.Series:
            _name = _input.name
            col_token = generate_temporary_column_name(n_bytes=8, columns=[_name])
            _input = add_row_index(_input.to_frame(), col_token)
            first_distinct_index = _input.groupby(_name).agg({col_token: "min"})[
                col_token
            ]

            return _input[col_token].isin(first_distinct_index)

        return self._from_call(
            func,
            "is_first_distinct",
            returns_scalar=False,
        )

    def is_last_distinct(self: Self) -> Self:
        def func(_input: dask_expr.Series) -> dask_expr.Series:
            _name = _input.name
            col_token = generate_temporary_column_name(n_bytes=8, columns=[_name])
            _input = add_row_index(_input.to_frame(), col_token)
            last_distinct_index = _input.groupby(_name).agg({col_token: "max"})[col_token]

            return _input[col_token].isin(last_distinct_index)

        return self._from_call(
            func,
            "is_last_distinct",
            returns_scalar=False,
        )

    def is_duplicated(self: Self) -> Self:
        def func(_input: dask_expr.Series) -> dask_expr.Series:
            _name = _input.name
            return (
                _input.to_frame().groupby(_name).transform("size", meta=(_name, int)) > 1
            )

        return self._from_call(
            func,
            "is_duplicated",
            returns_scalar=False,
        )

    def is_unique(self: Self) -> Self:
        def func(_input: dask_expr.Series) -> dask_expr.Series:
            _name = _input.name
            return (
                _input.to_frame().groupby(_name).transform("size", meta=(_name, int)) == 1
            )

        return self._from_call(
            func,
            "is_unique",
            returns_scalar=False,
        )

    def is_in(self: Self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.isin(other),
            "is_in",
            other,
            returns_scalar=False,
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
            dtypes=self._dtypes,
        )

    def mode(self: Self) -> Self:
        msg = "`Expr.mode` is not supported for the Dask backend."
        raise NotImplementedError(msg)

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
            dtype = narwhals_to_native_dtype(dtype, self._dtypes)
            return _input.astype(dtype)

        return self._from_call(
            func,
            "cast",
            dtype,
            returns_scalar=False,
        )


class DaskExprStringNamespace:
    def __init__(self, expr: DaskExpr) -> None:
        self._expr = expr

    def len_chars(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.str.len(), "len", returns_scalar=False
        )

    def replace(
        self,
        pattern: str,
        value: str,
        *,
        literal: bool = False,
        n: int = 1,
    ) -> DaskExpr:
        return self._expr._from_call(
            lambda _input, _pattern, _value, _literal, _n: _input.str.replace(
                _pattern, _value, regex=not _literal, n=_n
            ),
            "replace",
            pattern,
            value,
            literal,
            n,
            returns_scalar=False,
        )

    def replace_all(
        self,
        pattern: str,
        value: str,
        *,
        literal: bool = False,
    ) -> DaskExpr:
        return self._expr._from_call(
            lambda _input, _pattern, _value, _literal: _input.str.replace(
                _pattern, _value, n=-1, regex=not _literal
            ),
            "replace",
            pattern,
            value,
            literal,
            returns_scalar=False,
        )

    def strip_chars(self, characters: str | None = None) -> DaskExpr:
        return self._expr._from_call(
            lambda _input, characters: _input.str.strip(characters),
            "strip",
            characters,
            returns_scalar=False,
        )

    def starts_with(self, prefix: str) -> DaskExpr:
        return self._expr._from_call(
            lambda _input, prefix: _input.str.startswith(prefix),
            "starts_with",
            prefix,
            returns_scalar=False,
        )

    def ends_with(self, suffix: str) -> DaskExpr:
        return self._expr._from_call(
            lambda _input, suffix: _input.str.endswith(suffix),
            "ends_with",
            suffix,
            returns_scalar=False,
        )

    def contains(self, pattern: str, *, literal: bool = False) -> DaskExpr:
        return self._expr._from_call(
            lambda _input, pat, regex: _input.str.contains(pat=pat, regex=regex),
            "contains",
            pattern,
            not literal,
            returns_scalar=False,
        )

    def slice(self, offset: int, length: int | None = None) -> DaskExpr:
        stop = offset + length if length else None
        return self._expr._from_call(
            lambda _input, start, stop: _input.str.slice(start=start, stop=stop),
            "slice",
            offset,
            stop,
            returns_scalar=False,
        )

    def to_datetime(self: Self, format: str | None) -> DaskExpr:  # noqa: A002
        import dask.dataframe as dd  # ignore-banned-import()

        return self._expr._from_call(
            lambda _input, fmt: dd.to_datetime(_input, format=fmt),
            "to_datetime",
            format,
            returns_scalar=False,
        )

    def to_uppercase(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.str.upper(),
            "to_uppercase",
            returns_scalar=False,
        )

    def to_lowercase(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.str.lower(),
            "to_lowercase",
            returns_scalar=False,
        )


class DaskExprDateTimeNamespace:
    def __init__(self, expr: DaskExpr) -> None:
        self._expr = expr

    def date(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.date,
            "date",
            returns_scalar=False,
        )

    def year(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.year,
            "year",
            returns_scalar=False,
        )

    def month(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.month,
            "month",
            returns_scalar=False,
        )

    def day(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.day,
            "day",
            returns_scalar=False,
        )

    def hour(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.hour,
            "hour",
            returns_scalar=False,
        )

    def minute(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.minute,
            "minute",
            returns_scalar=False,
        )

    def second(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.second,
            "second",
            returns_scalar=False,
        )

    def millisecond(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.microsecond // 1000,
            "millisecond",
            returns_scalar=False,
        )

    def microsecond(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.microsecond,
            "microsecond",
            returns_scalar=False,
        )

    def nanosecond(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.microsecond * 1000 + _input.dt.nanosecond,
            "nanosecond",
            returns_scalar=False,
        )

    def ordinal_day(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.dayofyear,
            "ordinal_day",
            returns_scalar=False,
        )

    def to_string(self, format: str) -> DaskExpr:  # noqa: A002
        return self._expr._from_call(
            lambda _input, _format: _input.dt.strftime(_format),
            "strftime",
            format.replace("%.f", ".%f"),
            returns_scalar=False,
        )

    def replace_time_zone(self, time_zone: str | None) -> DaskExpr:
        return self._expr._from_call(
            lambda _input, _time_zone: _input.dt.tz_localize(None).dt.tz_localize(
                _time_zone
            )
            if _time_zone is not None
            else _input.dt.tz_localize(None),
            "tz_localize",
            time_zone,
            returns_scalar=False,
        )

    def convert_time_zone(self, time_zone: str) -> DaskExpr:
        def func(s: dask_expr.Series, time_zone: str) -> dask_expr.Series:
            dtype = native_to_narwhals_dtype(s, self._expr._dtypes, Implementation.DASK)
            if dtype.time_zone is None:  # type: ignore[attr-defined]
                return s.dt.tz_localize("UTC").dt.tz_convert(time_zone)
            else:
                return s.dt.tz_convert(time_zone)

        return self._expr._from_call(
            func,
            "tz_convert",
            time_zone,
            returns_scalar=False,
        )

    def timestamp(self, time_unit: Literal["ns", "us", "ms"] = "us") -> DaskExpr:
        def func(
            s: dask_expr.Series, time_unit: Literal["ns", "us", "ms"] = "us"
        ) -> dask_expr.Series:
            dtype = native_to_narwhals_dtype(s, self._expr._dtypes, Implementation.DASK)
            is_pyarrow_dtype = "pyarrow" in str(dtype)
            mask_na = s.isna()
            if dtype == self._expr._dtypes.Date:
                # Date is only supported in pandas dtypes if pyarrow-backed
                s_cast = s.astype("Int32[pyarrow]")
                result = calculate_timestamp_date(s_cast, time_unit)
            elif dtype == self._expr._dtypes.Datetime:
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

        return self._expr._from_call(
            func,
            "datetime",
            time_unit,
            returns_scalar=False,
        )

    def total_minutes(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.total_seconds() // 60,
            "total_minutes",
            returns_scalar=False,
        )

    def total_seconds(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.total_seconds() // 1,
            "total_seconds",
            returns_scalar=False,
        )

    def total_milliseconds(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.total_seconds() * 1000 // 1,
            "total_milliseconds",
            returns_scalar=False,
        )

    def total_microseconds(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.total_seconds() * 1_000_000 // 1,
            "total_microseconds",
            returns_scalar=False,
        )

    def total_nanoseconds(self) -> DaskExpr:
        return self._expr._from_call(
            lambda _input: _input.dt.total_seconds() * 1_000_000_000 // 1,
            "total_nanoseconds",
            returns_scalar=False,
        )


class DaskExprNameNamespace:
    def __init__(self: Self, expr: DaskExpr) -> None:
        self._expr = expr

    def keep(self: Self) -> DaskExpr:
        root_names = self._expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.keep`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        return self._expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._expr._call(df), root_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=root_names,
            returns_scalar=self._expr._returns_scalar,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )

    def map(self: Self, function: Callable[[str], str]) -> DaskExpr:
        root_names = self._expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.map`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        output_names = [function(str(name)) for name in root_names]

        return self._expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._expr._call(df), output_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._expr._returns_scalar,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )

    def prefix(self: Self, prefix: str) -> DaskExpr:
        root_names = self._expr._root_names
        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.prefix`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        output_names = [prefix + str(name) for name in root_names]
        return self._expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._expr._call(df), output_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._expr._returns_scalar,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )

    def suffix(self: Self, suffix: str) -> DaskExpr:
        root_names = self._expr._root_names
        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.suffix`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)

        output_names = [str(name) + suffix for name in root_names]

        return self._expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._expr._call(df), output_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._expr._returns_scalar,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )

    def to_lowercase(self: Self) -> DaskExpr:
        root_names = self._expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.to_lowercase`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)
        output_names = [str(name).lower() for name in root_names]

        return self._expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._expr._call(df), output_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._expr._returns_scalar,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )

    def to_uppercase(self: Self) -> DaskExpr:
        root_names = self._expr._root_names

        if root_names is None:
            msg = (
                "Anonymous expressions are not supported in `.name.to_uppercase`.\n"
                "Instead of `nw.all()`, try using a named expression, such as "
                "`nw.col('a', 'b')`\n"
            )
            raise ValueError(msg)
        output_names = [str(name).upper() for name in root_names]

        return self._expr.__class__(
            lambda df: [
                series.rename(name)
                for series, name in zip(self._expr._call(df), output_names)
            ],
            depth=self._expr._depth,
            function_name=self._expr._function_name,
            root_names=root_names,
            output_names=output_names,
            returns_scalar=self._expr._returns_scalar,
            backend_version=self._expr._backend_version,
            dtypes=self._expr._dtypes,
        )
