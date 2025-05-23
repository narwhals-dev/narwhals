from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Iterable, Sequence

from narwhals._compliant import CompliantThen, LazyNamespace, LazyWhen
from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._spark_like.dataframe import SparkLikeLazyFrame
from narwhals._spark_like.expr import SparkLikeExpr
from narwhals._spark_like.selectors import SparkLikeSelectorNamespace
from narwhals._spark_like.utils import narwhals_to_native_dtype

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.dataframe import SQLFrameDataFrame  # noqa: F401
    from narwhals.dtypes import DType
    from narwhals.typing import ConcatMethod, NonNestedLiteral
    from narwhals.utils import Implementation, Version


class SparkLikeNamespace(
    LazyNamespace[SparkLikeLazyFrame, SparkLikeExpr, "SQLFrameDataFrame"]
):
    def __init__(
        self,
        *,
        backend_version: tuple[int, ...],
        version: Version,
        implementation: Implementation,
    ) -> None:
        self._backend_version = backend_version
        self._version = version
        self._implementation = implementation

    @property
    def selectors(self) -> SparkLikeSelectorNamespace:
        return SparkLikeSelectorNamespace.from_namespace(self)

    @property
    def _expr(self) -> type[SparkLikeExpr]:
        return SparkLikeExpr

    @property
    def _lazyframe(self) -> type[SparkLikeLazyFrame]:
        return SparkLikeLazyFrame

    def lit(
        self, value: NonNestedLiteral, dtype: DType | type[DType] | None
    ) -> SparkLikeExpr:
        def _lit(df: SparkLikeLazyFrame) -> list[Column]:
            column = df._F.lit(value)
            if dtype:
                native_dtype = narwhals_to_native_dtype(
                    dtype, version=self._version, spark_types=df._native_dtypes
                )
                column = column.cast(native_dtype)

            return [column]

        return self._expr(
            call=_lit,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def len(self) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [df._F.count("*")]

        return self._expr(
            func,
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def all_horizontal(self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.and_, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def any_horizontal(self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.or_, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def sum_horizontal(self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = (
                df._F.coalesce(col, df._F.lit(0)) for _expr in exprs for col in _expr(df)
            )
            return [reduce(operator.add, cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def mean_horizontal(self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in exprs for c in _expr(df)]
            F = exprs[0]._F  # noqa: N806
            # PySpark before 3.5 doesn't have `try_divide`, SQLFrame doesn't have it.
            divide = getattr(F, "try_divide", operator.truediv)
            return [
                divide(
                    reduce(
                        operator.add, (df._F.coalesce(col, df._F.lit(0)) for col in cols)
                    ),
                    reduce(
                        operator.add,
                        (
                            col.isNotNull().cast(df._native_dtypes.IntegerType())
                            for col in cols
                        ),
                    ),
                )
            ]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def max_horizontal(self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [df._F.greatest(*cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def min_horizontal(self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [df._F.least(*cols)]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def concat(
        self, items: Iterable[SparkLikeLazyFrame], *, how: ConcatMethod
    ) -> SparkLikeLazyFrame:
        dfs = [item._native_frame for item in items]
        if how == "vertical":
            cols_0 = dfs[0].columns
            for i, df in enumerate(dfs[1:], start=1):
                cols_current = df.columns
                if not ((len(cols_current) == len(cols_0)) and (cols_current == cols_0)):
                    msg = (
                        "unable to vstack, column names don't match:\n"
                        f"   - dataframe 0: {cols_0}\n"
                        f"   - dataframe {i}: {cols_current}\n"
                    )
                    raise TypeError(msg)

            return SparkLikeLazyFrame(
                native_dataframe=reduce(lambda x, y: x.union(y), dfs),
                backend_version=self._backend_version,
                version=self._version,
                implementation=self._implementation,
            )

        if how == "diagonal":
            return SparkLikeLazyFrame(
                native_dataframe=reduce(
                    lambda x, y: x.unionByName(y, allowMissingColumns=True), dfs
                ),
                backend_version=self._backend_version,
                version=self._version,
                implementation=self._implementation,
            )
        raise NotImplementedError

    def concat_str(
        self, *exprs: SparkLikeExpr, separator: str, ignore_nulls: bool
    ) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [s for _expr in exprs for s in _expr(df)]
            cols_casted = [s.cast(df._native_dtypes.StringType()) for s in cols]
            null_mask = [df._F.isnull(s) for s in cols]

            if not ignore_nulls:
                null_mask_result = reduce(operator.or_, null_mask)
                result = df._F.when(
                    ~null_mask_result,
                    reduce(
                        lambda x, y: df._F.format_string(f"%s{separator}%s", x, y),
                        cols_casted,
                    ),
                ).otherwise(df._F.lit(None))
            else:
                init_value, *values = [
                    df._F.when(~nm, col).otherwise(df._F.lit(""))
                    for col, nm in zip(cols_casted, null_mask)
                ]

                separators = (
                    df._F.when(nm, df._F.lit("")).otherwise(df._F.lit(separator))
                    for nm in null_mask[:-1]
                )
                result = reduce(
                    lambda x, y: df._F.format_string("%s%s", x, y),
                    (
                        df._F.format_string("%s%s", s, v)
                        for s, v in zip(separators, values)
                    ),
                    init_value,
                )

            return [result]

        return self._expr(
            call=func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def when(self, predicate: SparkLikeExpr) -> SparkLikeWhen:
        return SparkLikeWhen.from_expr(predicate, context=self)


class SparkLikeWhen(LazyWhen[SparkLikeLazyFrame, "Column", SparkLikeExpr]):
    @property
    def _then(self) -> type[SparkLikeThen]:
        return SparkLikeThen

    def __call__(self, df: SparkLikeLazyFrame) -> Sequence[Column]:
        self.when = df._F.when
        self.lit = df._F.lit
        return super().__call__(df)


class SparkLikeThen(
    CompliantThen[SparkLikeLazyFrame, "Column", SparkLikeExpr], SparkLikeExpr
): ...
