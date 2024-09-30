from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import NoReturn
from typing import cast

from narwhals._dask.dataframe import DaskLazyFrame
from narwhals._dask.expr import DaskExpr
from narwhals._dask.selectors import DaskSelectorNamespace
from narwhals._dask.utils import narwhals_to_native_dtype
from narwhals._dask.utils import validate_comparand
from narwhals._expression_parsing import combine_root_names
from narwhals._expression_parsing import parse_into_exprs
from narwhals._expression_parsing import reduce_output_names

if TYPE_CHECKING:
    import dask_expr

    from narwhals._dask.typing import IntoDaskExpr
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes


class DaskNamespace:
    @property
    def selectors(self) -> DaskSelectorNamespace:
        return DaskSelectorNamespace(
            backend_version=self._backend_version, dtypes=self._dtypes
        )

    def __init__(self, *, backend_version: tuple[int, ...], dtypes: DTypes) -> None:
        self._backend_version = backend_version
        self._dtypes = dtypes

    def all(self) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            return [df._native_frame.loc[:, column_name] for column_name in df.columns]

        return DaskExpr(
            func,
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
            returns_scalar=False,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def col(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def nth(self, *column_indices: int) -> DaskExpr:
        return DaskExpr.from_column_indices(
            *column_indices, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def lit(self, value: Any, dtype: DType | None) -> DaskExpr:
        def convert_if_dtype(
            series: dask_expr.Series, dtype: DType | type[DType]
        ) -> dask_expr.Series:
            return (
                series.astype(narwhals_to_native_dtype(dtype, self._dtypes))
                if dtype
                else series
            )

        return DaskExpr(
            lambda df: [
                df._native_frame.assign(lit=value)
                .loc[:, "lit"]
                .pipe(convert_if_dtype, dtype)
            ],
            depth=0,
            function_name="lit",
            root_names=None,
            output_names=["lit"],
            returns_scalar=False,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def min(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names, backend_version=self._backend_version, dtypes=self._dtypes
        ).min()

    def max(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names, backend_version=self._backend_version, dtypes=self._dtypes
        ).max()

    def mean(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names, backend_version=self._backend_version, dtypes=self._dtypes
        ).mean()

    def sum(self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names, backend_version=self._backend_version, dtypes=self._dtypes
        ).sum()

    def len(self) -> DaskExpr:
        import dask.dataframe as dd  # ignore-banned-import
        import pandas as pd  # ignore-banned-import

        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            if not df.columns:
                return [
                    dd.from_pandas(
                        pd.Series([0], name="len"),
                        npartitions=df._native_frame.npartitions,
                    )
                ]
            return [df._native_frame.loc[:, df.columns[0]].size.to_series().rename("len")]

        # coverage bug? this is definitely hit
        return DaskExpr(  # pragma: no cover
            func,
            depth=0,
            function_name="len",
            root_names=None,
            output_names=["len"],
            returns_scalar=True,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def all_horizontal(self, *exprs: IntoDaskExpr) -> DaskExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            series = [s for _expr in parsed_exprs for s in _expr._call(df)]
            return [reduce(lambda x, y: x & y, series).rename(series[0].name)]

        return DaskExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="all_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def any_horizontal(self, *exprs: IntoDaskExpr) -> DaskExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            series = [s for _expr in parsed_exprs for s in _expr._call(df)]
            return [reduce(lambda x, y: x | y, series).rename(series[0].name)]

        return DaskExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="any_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def sum_horizontal(self, *exprs: IntoDaskExpr) -> DaskExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: DaskLazyFrame) -> list[dask_expr.Series]:
            series = [s.fillna(0) for _expr in parsed_exprs for s in _expr._call(df)]
            return [reduce(lambda x, y: x + y, series).rename(series[0].name)]

        return DaskExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="sum_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def concat(
        self,
        items: Iterable[DaskLazyFrame],
        *,
        how: Literal["horizontal", "vertical"],
    ) -> DaskLazyFrame:
        import dask.dataframe as dd  # ignore-banned-import

        if len(list(items)) == 0:
            msg = "No items to concatenate"  # pragma: no cover
            raise AssertionError(msg)
        native_frames = [i._native_frame for i in items]
        if how == "vertical":
            if not all(
                tuple(i.columns) == tuple(native_frames[0].columns) for i in native_frames
            ):  # pragma: no cover
                msg = "unable to vstack with non-matching columns"
                raise AssertionError(msg)
            return DaskLazyFrame(
                dd.concat(native_frames, axis=0, join="inner"),
                backend_version=self._backend_version,
                dtypes=self._dtypes,
            )
        if how == "horizontal":
            all_column_names: list[str] = [
                column for frame in native_frames for column in frame.columns
            ]
            if len(all_column_names) != len(set(all_column_names)):  # pragma: no cover
                duplicates = [
                    i for i in all_column_names if all_column_names.count(i) > 1
                ]
                msg = (
                    f"Columns with name(s): {', '.join(duplicates)} "
                    "have more than one occurrence"
                )
                raise AssertionError(msg)
            return DaskLazyFrame(
                dd.concat(native_frames, axis=1, join="outer"),
                backend_version=self._backend_version,
                dtypes=self._dtypes,
            )
        raise NotImplementedError

    def mean_horizontal(self, *exprs: IntoDaskExpr) -> IntoDaskExpr:
        dask_exprs = parse_into_exprs(*exprs, namespace=self)
        total = reduce(lambda x, y: x + y, (e.fill_null(0.0) for e in dask_exprs))
        n_non_zero = reduce(lambda x, y: x + y, ((1 - e.is_null()) for e in dask_exprs))
        return total / n_non_zero

    def _create_expr_from_series(self, _: Any) -> NoReturn:
        msg = "`_create_expr_from_series` for DaskNamespace exists only for compatibility"
        raise NotImplementedError(msg)

    def _create_compliant_series(self, _: Any) -> NoReturn:
        msg = "`_create_compliant_series` for DaskNamespace exists only for compatibility"
        raise NotImplementedError(msg)

    def _create_series_from_scalar(self, *_: Any) -> NoReturn:
        msg = (
            "`_create_series_from_scalar` for DaskNamespace exists only for compatibility"
        )
        raise NotImplementedError(msg)

    def _create_expr_from_callable(  # pragma: no cover
        self,
        func: Callable[[DaskLazyFrame], list[DaskExpr]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
    ) -> DaskExpr:
        msg = (
            "`_create_expr_from_callable` for DaskNamespace exists only for compatibility"
        )
        raise NotImplementedError(msg)

    def when(
        self,
        *predicates: IntoDaskExpr,
    ) -> DaskWhen:
        plx = self.__class__(backend_version=self._backend_version, dtypes=self._dtypes)
        if predicates:
            condition = plx.all_horizontal(*predicates)
        else:
            msg = "at least one predicate needs to be provided"
            raise TypeError(msg)

        return DaskWhen(
            condition, self._backend_version, returns_scalar=False, dtypes=self._dtypes
        )


class DaskWhen:
    def __init__(
        self,
        condition: DaskExpr,
        backend_version: tuple[int, ...],
        then_value: Any = None,
        otherwise_value: Any = None,
        *,
        returns_scalar: bool,
        dtypes: DTypes,
    ) -> None:
        self._backend_version = backend_version
        self._condition = condition
        self._then_value = then_value
        self._otherwise_value = otherwise_value
        self._returns_scalar = returns_scalar
        self._dtypes = dtypes

    def __call__(self, df: DaskLazyFrame) -> list[dask_expr.Series]:
        from narwhals._dask.namespace import DaskNamespace
        from narwhals._expression_parsing import parse_into_expr

        plx = DaskNamespace(backend_version=self._backend_version, dtypes=self._dtypes)

        condition = parse_into_expr(self._condition, namespace=plx)._call(df)[0]  # type: ignore[arg-type]
        condition = cast("dask_expr.Series", condition)
        try:
            value_series = parse_into_expr(self._then_value, namespace=plx)._call(df)[0]  # type: ignore[arg-type]
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression
            _df = condition.to_frame("a")
            _df["tmp"] = self._then_value
            value_series = _df["tmp"]
        value_series = cast("dask_expr.Series", value_series)
        validate_comparand(condition, value_series)

        if self._otherwise_value is None:
            return [value_series.where(condition)]
        try:
            otherwise_series = parse_into_expr(
                self._otherwise_value, namespace=plx
            )._call(df)[0]  # type: ignore[arg-type]
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression
            return [value_series.where(condition, self._otherwise_value)]
        validate_comparand(condition, otherwise_series)
        return [value_series.where(condition, otherwise_series)]

    def then(self, value: DaskExpr | Any) -> DaskThen:
        self._then_value = value

        return DaskThen(
            self,
            depth=0,
            function_name="whenthen",
            root_names=None,
            output_names=None,
            returns_scalar=self._returns_scalar,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )


class DaskThen(DaskExpr):
    def __init__(
        self,
        call: DaskWhen,
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> None:
        self._backend_version = backend_version
        self._dtypes = dtypes
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names
        self._returns_scalar = returns_scalar

    def otherwise(self, value: DaskExpr | Any) -> DaskExpr:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `DaskWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self
