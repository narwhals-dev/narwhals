from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

import dask.dataframe as dd
import pandas as pd

from narwhals._compliant import CompliantThen
from narwhals._compliant import CompliantWhen
from narwhals._compliant import LazyNamespace
from narwhals._compliant.namespace import DepthTrackingNamespace
from narwhals._dask.dataframe import DaskLazyFrame
from narwhals._dask.expr import DaskExpr
from narwhals._dask.selectors import DaskSelectorNamespace
from narwhals._dask.utils import align_series_full_broadcast
from narwhals._dask.utils import name_preserving_div
from narwhals._dask.utils import name_preserving_sum
from narwhals._dask.utils import narwhals_to_native_dtype
from narwhals._dask.utils import validate_comparand
from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.utils import Version

    try:
        import dask.dataframe.dask_expr as dx
    except ModuleNotFoundError:  # pragma: no cover
        import dask_expr as dx


class DaskNamespace(
    LazyNamespace[DaskLazyFrame, DaskExpr, dd.DataFrame],
    DepthTrackingNamespace[DaskLazyFrame, DaskExpr],
):
    _implementation: Implementation = Implementation.DASK

    @property
    def selectors(self: Self) -> DaskSelectorNamespace:
        return DaskSelectorNamespace.from_namespace(self)

    @property
    def _expr(self) -> type[DaskExpr]:
        return DaskExpr

    @property
    def _lazyframe(self) -> type[DaskLazyFrame]:
        return DaskLazyFrame

    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._version = version

    def lit(self: Self, value: Any, dtype: DType | type[DType] | None) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            if dtype is not None:
                native_dtype = narwhals_to_native_dtype(dtype, self._version)
                native_pd_series = pd.Series([value], dtype=native_dtype, name="literal")
            else:
                native_pd_series = pd.Series([value], name="literal")
            npartitions = df._native_frame.npartitions
            dask_series = dd.from_pandas(native_pd_series, npartitions=npartitions)
            return [dask_series[0].to_series()]

        return self._expr(
            func,
            depth=0,
            function_name="lit",
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )

    def len(self: Self) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            # We don't allow dataframes with 0 columns, so `[0]` is safe.
            return [df._native_frame[df.columns[0]].size.to_series()]

        return self._expr(
            func,
            depth=0,
            function_name="len",
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )

    def all_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = align_series_full_broadcast(
                df, *(s for _expr in exprs for s in _expr(df))
            )
            return [reduce(operator.and_, series)]

        return self._expr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="all_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def any_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = align_series_full_broadcast(
                df, *(s for _expr in exprs for s in _expr(df))
            )
            return [reduce(operator.or_, series)]

        return self._expr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="any_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def sum_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = align_series_full_broadcast(
                df, *(s for _expr in exprs for s in _expr(df))
            )
            return [dd.concat(series, axis=1).sum(axis=1)]

        return self._expr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="sum_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def concat(
        self: Self,
        items: Iterable[DaskLazyFrame],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> DaskLazyFrame:
        if not items:
            msg = "No items to concatenate"  # pragma: no cover
            raise AssertionError(msg)
        dfs = [i._native_frame for i in items]
        cols_0 = dfs[0].columns
        if how == "vertical":
            for i, df in enumerate(dfs[1:], start=1):
                cols_current = df.columns
                if not (
                    (len(cols_current) == len(cols_0)) and (cols_current == cols_0).all()
                ):
                    msg = (
                        "unable to vstack, column names don't match:\n"
                        f"   - dataframe 0: {cols_0.to_list()}\n"
                        f"   - dataframe {i}: {cols_current.to_list()}\n"
                    )
                    raise TypeError(msg)
            return DaskLazyFrame(
                dd.concat(dfs, axis=0, join="inner"),
                backend_version=self._backend_version,
                version=self._version,
            )
        if how == "horizontal":
            all_column_names: list[str] = [
                column for frame in dfs for column in frame.columns
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
                dd.concat(dfs, axis=1, join="outer"),
                backend_version=self._backend_version,
                version=self._version,
            )
        if how == "diagonal":
            return DaskLazyFrame(
                dd.concat(dfs, axis=0, join="outer"),
                backend_version=self._backend_version,
                version=self._version,
            )

        raise NotImplementedError

    def mean_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            expr_results = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(df, *(s.fillna(0) for s in expr_results))
            non_na = align_series_full_broadcast(
                df, *(1 - s.isna() for s in expr_results)
            )
            return [
                name_preserving_div(
                    reduce(name_preserving_sum, series),
                    reduce(name_preserving_sum, non_na),
                )
            ]

        return self._expr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="mean_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def min_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = align_series_full_broadcast(
                df, *(s for _expr in exprs for s in _expr(df))
            )

            return [dd.concat(series, axis=1).min(axis=1)]

        return self._expr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="min_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def max_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = align_series_full_broadcast(
                df, *(s for _expr in exprs for s in _expr(df))
            )

            return [dd.concat(series, axis=1).max(axis=1)]

        return self._expr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="max_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
        )

    def when(self: Self, predicate: DaskExpr) -> DaskWhen:
        return DaskWhen.from_expr(predicate, context=self)

    def concat_str(
        self: Self,
        *exprs: DaskExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            expr_results = [s for _expr in exprs for s in _expr(df)]
            series = (
                s.astype(str) for s in align_series_full_broadcast(df, *expr_results)
            )
            null_mask = [s.isna() for s in align_series_full_broadcast(df, *expr_results)]

            if not ignore_nulls:
                null_mask_result = reduce(operator.or_, null_mask)
                result = reduce(lambda x, y: x + separator + y, series).where(
                    ~null_mask_result, None
                )
            else:
                init_value, *values = [
                    s.where(~nm, "") for s, nm in zip(series, null_mask)
                ]

                separators = (
                    nm.map({True: "", False: separator}, meta=str)
                    for nm in null_mask[:-1]
                )
                result = reduce(
                    operator.add,
                    (s + v for s, v in zip(separators, values)),
                    init_value,
                )

            return [result]

        return self._expr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="concat_str",
            evaluate_output_names=getattr(
                exprs[0], "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(exprs[0], "_alias_output_names", None),
            backend_version=self._backend_version,
            version=self._version,
        )


class DaskWhen(CompliantWhen[DaskLazyFrame, "dx.Series", DaskExpr]):
    @property
    def _then(self) -> type[DaskThen]:
        return DaskThen

    def __call__(self: Self, df: DaskLazyFrame) -> Sequence[dx.Series]:
        condition = self._condition(df)[0]

        if isinstance(self._then_value, DaskExpr):
            then_value = self._then_value(df)[0]
        else:
            then_value = self._then_value
        (then_series,) = align_series_full_broadcast(df, then_value)
        validate_comparand(condition, then_series)

        if self._otherwise_value is None:
            return [then_series.where(condition)]

        if isinstance(self._otherwise_value, DaskExpr):
            otherwise_value = self._otherwise_value(df)[0]
        else:
            return [then_series.where(condition, self._otherwise_value)]  # pyright: ignore[reportArgumentType]
        (otherwise_series,) = align_series_full_broadcast(df, otherwise_value)
        validate_comparand(condition, otherwise_series)
        return [then_series.where(condition, otherwise_series)]  # pyright: ignore[reportArgumentType]


class DaskThen(CompliantThen[DaskLazyFrame, "dx.Series", DaskExpr], DaskExpr): ...
