from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal

from narwhals._compliant import CompliantThen
from narwhals._compliant import EagerNamespace
from narwhals._compliant import EagerWhen
from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals._pandas_like.dataframe import PandasLikeDataFrame
from narwhals._pandas_like.expr import PandasLikeExpr
from narwhals._pandas_like.selectors import PandasSelectorNamespace
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals._pandas_like.utils import align_series_full_broadcast
from narwhals._pandas_like.utils import diagonal_concat
from narwhals._pandas_like.utils import horizontal_concat
from narwhals._pandas_like.utils import vertical_concat
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.utils import Implementation
    from narwhals.utils import Version


class PandasLikeNamespace(
    EagerNamespace[PandasLikeDataFrame, PandasLikeSeries, PandasLikeExpr]
):
    @property
    def _dataframe(self) -> type[PandasLikeDataFrame]:
        return PandasLikeDataFrame

    @property
    def _expr(self) -> type[PandasLikeExpr]:
        return PandasLikeExpr

    @property
    def _series(self) -> type[PandasLikeSeries]:
        return PandasLikeSeries

    @property
    def selectors(self: Self) -> PandasSelectorNamespace:
        return PandasSelectorNamespace(self)

    # --- not in spec ---
    def __init__(
        self: Self,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._implementation = implementation
        self._backend_version = backend_version
        self._version = version

    def lit(self: Self, value: Any, dtype: DType | type[DType] | None) -> PandasLikeExpr:
        def _lit_pandas_series(df: PandasLikeDataFrame) -> PandasLikeSeries:
            pandas_series = self._series.from_iterable(
                data=[value],
                name="literal",
                index=df._native_frame.index[0:1],
                context=self,
            )
            if dtype:
                return pandas_series.cast(dtype)
            return pandas_series

        return PandasLikeExpr(
            lambda df: [_lit_pandas_series(df)],
            depth=0,
            function_name="lit",
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )

    def len(self: Self) -> PandasLikeExpr:
        return PandasLikeExpr(
            lambda df: [
                self._series.from_iterable(
                    [len(df._native_frame)], name="len", index=[0], context=self
                )
            ],
            depth=0,
            function_name="len",
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )

    # --- horizontal ---
    def sum_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            series = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(*series)
            native_series = (s.fill_null(0, None, None) for s in series)
            return [reduce(operator.add, native_series)]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="sum_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def all_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            series = align_series_full_broadcast(
                *(s for _expr in exprs for s in _expr(df))
            )
            return [reduce(operator.and_, series)]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="all_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def any_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            series = align_series_full_broadcast(
                *(s for _expr in exprs for s in _expr(df))
            )
            return [reduce(operator.or_, series)]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="any_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def mean_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            expr_results = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(
                *(s.fill_null(0, strategy=None, limit=None) for s in expr_results)
            )
            non_na = align_series_full_broadcast(*(1 - s.is_null() for s in expr_results))
            return [reduce(operator.add, series) / reduce(operator.add, non_na)]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="mean_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def min_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            series = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(*series)

            return [
                PandasLikeSeries(
                    self.concat(
                        (s.to_frame() for s in series), how="horizontal"
                    )._native_frame.min(axis=1),
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                    version=self._version,
                ).alias(series[0].name)
            ]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="min_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def max_horizontal(self: Self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            series = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(*series)

            return [
                PandasLikeSeries(
                    self.concat(
                        (s.to_frame() for s in series), how="horizontal"
                    )._native_frame.max(axis=1),
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                    version=self._version,
                ).alias(series[0].name)
            ]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="max_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def concat(
        self: Self,
        items: Iterable[PandasLikeDataFrame],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> PandasLikeDataFrame:
        dfs: list[Any] = [item._native_frame for item in items]
        if how == "horizontal":
            return PandasLikeDataFrame(
                horizontal_concat(
                    dfs,
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ),
                implementation=self._implementation,
                backend_version=self._backend_version,
                version=self._version,
                validate_column_names=True,
            )
        if how == "vertical":
            return PandasLikeDataFrame(
                vertical_concat(
                    dfs,
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ),
                implementation=self._implementation,
                backend_version=self._backend_version,
                version=self._version,
                validate_column_names=True,
            )

        if how == "diagonal":
            return PandasLikeDataFrame(
                diagonal_concat(
                    dfs,
                    implementation=self._implementation,
                    backend_version=self._backend_version,
                ),
                implementation=self._implementation,
                backend_version=self._backend_version,
                version=self._version,
                validate_column_names=True,
            )
        raise NotImplementedError

    def when(self: Self, predicate: PandasLikeExpr) -> PandasWhen:
        return PandasWhen.from_expr(predicate, context=self)

    def concat_str(
        self: Self,
        *exprs: PandasLikeExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> PandasLikeExpr:
        dtypes = import_dtypes_module(self._version)

        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            expr_results = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(
                *(s.cast(dtypes.String()) for s in expr_results)
            )
            null_mask = align_series_full_broadcast(*(s.is_null() for s in expr_results))

            if not ignore_nulls:
                null_mask_result = reduce(operator.or_, null_mask)
                result = reduce(lambda x, y: x + separator + y, series).zip_with(
                    ~null_mask_result, None
                )
            else:
                init_value, *values = [
                    s.zip_with(~nm, "") for s, nm in zip(series, null_mask)
                ]

                sep_array = init_value.from_iterable(
                    data=[separator] * len(init_value),
                    name="sep",
                    index=init_value.native.index,
                    context=self,
                )
                separators = (sep_array.zip_with(~nm, "") for nm in null_mask[:-1])
                result = reduce(
                    operator.add,
                    (s + v for s, v in zip(separators, values)),
                    init_value,
                )

            return [result]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="concat_str",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )


class PandasWhen(
    EagerWhen[PandasLikeDataFrame, PandasLikeSeries, PandasLikeExpr, "pd.Series[Any]"]
):
    @property
    def _then(self) -> type[PandasThen]:
        return PandasThen

    def _if_then_else(
        self, when: pd.Series[Any], then: pd.Series[Any], otherwise: Any, /
    ) -> pd.Series[Any]:
        return then.where(when) if otherwise is None else then.where(when, otherwise)


class PandasThen(
    CompliantThen[PandasLikeDataFrame, PandasLikeSeries, PandasLikeExpr], PandasLikeExpr
): ...
