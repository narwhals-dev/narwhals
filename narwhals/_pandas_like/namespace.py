from __future__ import annotations

import operator
import warnings
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, Sequence

from narwhals._compliant import CompliantThen, EagerNamespace, EagerWhen
from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._pandas_like.dataframe import PandasLikeDataFrame
from narwhals._pandas_like.expr import PandasLikeExpr
from narwhals._pandas_like.selectors import PandasSelectorNamespace
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals._pandas_like.utils import align_series_full_broadcast

if TYPE_CHECKING:
    import pandas as pd

    from narwhals._pandas_like.typing import NDFrameT
    from narwhals.dtypes import DType
    from narwhals.typing import NonNestedLiteral
    from narwhals.utils import Implementation, Version

VERTICAL: Literal[0] = 0
HORIZONTAL: Literal[1] = 1


class PandasLikeNamespace(
    EagerNamespace[
        PandasLikeDataFrame,
        PandasLikeSeries,
        PandasLikeExpr,
        "pd.DataFrame",
        "pd.Series[Any]",
    ]
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
    def selectors(self) -> PandasSelectorNamespace:
        return PandasSelectorNamespace.from_namespace(self)

    # --- not in spec ---
    def __init__(
        self,
        implementation: Implementation,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._implementation = implementation
        self._backend_version = backend_version
        self._version = version

    def lit(
        self, value: NonNestedLiteral, dtype: DType | type[DType] | None
    ) -> PandasLikeExpr:
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

    def len(self) -> PandasLikeExpr:
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
    def sum_horizontal(self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
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

    def all_horizontal(self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
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

    def any_horizontal(self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
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

    def mean_horizontal(self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
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

    def min_horizontal(self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
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

    def max_horizontal(self, *exprs: PandasLikeExpr) -> PandasLikeExpr:
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

    @property
    def _concat(self):  # type: ignore[no-untyped-def] # noqa: ANN202
        """Return the **native** equivalent of `pd.concat`."""
        # NOTE: Leave un-annotated to allow `@overload` matching via inference.
        if TYPE_CHECKING:
            import pandas as pd

            return pd.concat
        return self._implementation.to_native_namespace().concat

    def _concat_diagonal(self, dfs: Sequence[pd.DataFrame], /) -> pd.DataFrame:
        if self._implementation.is_pandas() and self._backend_version < (3,):
            if self._backend_version < (1,):
                return self._concat(dfs, axis=VERTICAL, copy=False, sort=False)
            return self._concat(dfs, axis=VERTICAL, copy=False)
        return self._concat(dfs, axis=VERTICAL)

    def _concat_horizontal(self, dfs: Sequence[NDFrameT], /) -> pd.DataFrame:
        if self._implementation.is_cudf():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The behavior of array concatenation with empty entries is deprecated",
                    category=FutureWarning,
                )
                return self._concat(dfs, axis=HORIZONTAL)
        elif self._implementation.is_pandas() and self._backend_version < (3,):
            return self._concat(dfs, axis=HORIZONTAL, copy=False)
        return self._concat(dfs, axis=HORIZONTAL)

    def _concat_vertical(self, dfs: Sequence[pd.DataFrame], /) -> pd.DataFrame:
        cols_0 = dfs[0].columns
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
        if self._implementation.is_pandas() and self._backend_version < (3,):
            return self._concat(dfs, axis=VERTICAL, copy=False)
        return self._concat(dfs, axis=VERTICAL)

    def when(self, predicate: PandasLikeExpr) -> PandasWhen:
        return PandasWhen.from_expr(predicate, context=self)

    def concat_str(
        self, *exprs: PandasLikeExpr, separator: str, ignore_nulls: bool
    ) -> PandasLikeExpr:
        string = self._version.dtypes.String()

        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            expr_results = [s for _expr in exprs for s in _expr(df)]
            series = align_series_full_broadcast(*(s.cast(string) for s in expr_results))
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
                    operator.add, (s + v for s, v in zip(separators, values)), init_value
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
        self,
        when: pd.Series[Any],
        then: pd.Series[Any],
        otherwise: pd.Series[Any] | NonNestedLiteral,
        /,
    ) -> pd.Series[Any]:
        return then.where(when) if otherwise is None else then.where(when, otherwise)


class PandasThen(
    CompliantThen[PandasLikeDataFrame, PandasLikeSeries, PandasLikeExpr], PandasLikeExpr
): ...
