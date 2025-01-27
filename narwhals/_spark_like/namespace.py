from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Sequence

from pyspark.sql import functions as F  # noqa: N812
from pyspark.sql.types import IntegerType

from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals._spark_like.dataframe import SparkLikeLazyFrame
from narwhals._spark_like.expr import SparkLikeExpr
from narwhals._spark_like.selectors import SparkLikeSelectorNamespace
from narwhals.typing import CompliantNamespace

if TYPE_CHECKING:
    from pyspark.sql import Column
    from pyspark.sql import DataFrame
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.utils import Version


class SparkLikeNamespace(CompliantNamespace["Column"]):
    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._version = version

    @property
    def selectors(self: Self) -> SparkLikeSelectorNamespace:
        return SparkLikeSelectorNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def all(self: Self) -> SparkLikeExpr:
        def _all(df: SparkLikeLazyFrame) -> list[Column]:
            return [F.col(col_name) for col_name in df.columns]

        return SparkLikeExpr(
            call=_all,
            depth=0,
            function_name="all",
            evaluate_output_names=lambda df: df.columns,
            alias_output_names=None,
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
        )

    def col(self: Self, *column_names: str) -> SparkLikeExpr:
        return SparkLikeExpr.from_column_names(
            *column_names, backend_version=self._backend_version, version=self._version
        )

    def nth(self: Self, *column_indices: int) -> SparkLikeExpr:
        return SparkLikeExpr.from_column_indices(
            *column_indices, backend_version=self._backend_version, version=self._version
        )

    def lit(self: Self, value: object, dtype: DType | None) -> SparkLikeExpr:
        if dtype is not None:
            msg = "todo"
            raise NotImplementedError(msg)

        def _lit(_: SparkLikeLazyFrame) -> list[Column]:
            import pyspark.sql.functions as F  # noqa: N812

            return [F.lit(value)]

        return SparkLikeExpr(
            call=_lit,
            depth=0,
            function_name="lit",
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            returns_scalar=True,
            backend_version=self._backend_version,
            version=self._version,
        )

    def len(self: Self) -> SparkLikeExpr:
        def func(_: SparkLikeLazyFrame) -> list[Column]:
            return [F.count("*")]

        return SparkLikeExpr(
            func,
            depth=0,
            function_name="len",
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            returns_scalar=True,
            backend_version=self._backend_version,
            version=self._version,
        )

    def all_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in exprs for c in _expr(df)]
            return [reduce(operator.and_, cols)]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="all_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
        )

    def any_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in exprs for c in _expr(df)]
            return [reduce(operator.or_, cols)]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="any_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
        )

    def sum_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in exprs for c in _expr(df)]
            return [
                reduce(
                    operator.add,
                    (F.coalesce(col, F.lit(0)) for col in cols),
                )
            ]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="sum_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
        )

    def mean_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in exprs for c in _expr(df)]
            return [
                (
                    reduce(operator.add, (F.coalesce(col, F.lit(0)) for col in cols))
                    / reduce(
                        operator.add,
                        (col.isNotNull().cast(IntegerType()) for col in cols),
                    )
                )
            ]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="mean_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
        )

    def max_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in exprs for c in _expr(df)]
            return [F.greatest(*cols)]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="max_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
        )

    def min_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in exprs for c in _expr(df)]
            return [F.least(*cols)]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="min_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
        )

    def concat(
        self: Self,
        items: Iterable[SparkLikeLazyFrame],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> SparkLikeLazyFrame:
        dfs: list[DataFrame] = [item._native_frame for item in items]
        if how == "horizontal":
            msg = (
                "Horizontal concatenation is not supported for LazyFrame backed by "
                "a PySpark DataFrame."
            )
            raise NotImplementedError(msg)

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
            )

        if how == "diagonal":
            return SparkLikeLazyFrame(
                native_dataframe=reduce(
                    lambda x, y: x.unionByName(y, allowMissingColumns=True), dfs
                ),
                backend_version=self._backend_version,
                version=self._version,
            )
        raise NotImplementedError

    def concat_str(
        self: Self,
        *exprs: SparkLikeExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> SparkLikeExpr:
        from pyspark.sql.types import StringType

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [s for _expr in exprs for s in _expr(df)]
            cols_casted = [s.cast(StringType()) for s in cols]
            null_mask = [F.isnull(s) for _expr in exprs for s in _expr(df)]

            if not ignore_nulls:
                null_mask_result = reduce(lambda x, y: x | y, null_mask)
                result = F.when(
                    ~null_mask_result,
                    reduce(
                        lambda x, y: F.format_string(f"%s{separator}%s", x, y),
                        cols_casted,
                    ),
                ).otherwise(F.lit(None))
            else:
                init_value, *values = [
                    F.when(~nm, col).otherwise(F.lit(""))
                    for col, nm in zip(cols_casted, null_mask)
                ]

                separators = (
                    F.when(nm, F.lit("")).otherwise(F.lit(separator))
                    for nm in null_mask[:-1]
                )
                result = reduce(
                    lambda x, y: F.format_string("%s%s", x, y),
                    (F.format_string("%s%s", s, v) for s, v in zip(separators, values)),
                    init_value,
                )

            return [result]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="concat_str",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
        )

    def when(self: Self, *predicates: SparkLikeExpr) -> SparkLikeWhen:
        plx = self.__class__(backend_version=self._backend_version, version=self._version)
        condition = plx.all_horizontal(*predicates)
        return SparkLikeWhen(
            condition, self._backend_version, returns_scalar=False, version=self._version
        )


class SparkLikeWhen:
    def __init__(
        self: Self,
        condition: SparkLikeExpr,
        backend_version: tuple[int, ...],
        then_value: Any | None = None,
        otherwise_value: Any | None = None,
        *,
        returns_scalar: bool,
        version: Version,
    ) -> None:
        self._backend_version = backend_version
        self._condition = condition
        self._then_value = then_value
        self._otherwise_value = otherwise_value
        self._returns_scalar = returns_scalar
        self._version = version

    def __call__(self: Self, df: SparkLikeLazyFrame) -> list[Column]:
        condition = self._condition(df)[0]

        if isinstance(self._then_value, SparkLikeExpr):
            value_ = self._then_value(df)[0]
        else:
            # `self._then_value` is a scalar
            value_ = F.lit(self._then_value)

        if isinstance(self._otherwise_value, SparkLikeExpr):
            other_ = self._otherwise_value(df)[0]
        else:
            # `self._otherwise_value` is a scalar
            other_ = F.lit(self._otherwise_value)

        return [F.when(condition=condition, value=value_).otherwise(value=other_)]

    def then(self: Self, value: SparkLikeExpr | Any) -> SparkLikeThen:
        self._then_value = value

        return SparkLikeThen(
            self,
            depth=0,
            function_name="whenthen",
            evaluate_output_names=getattr(
                value, "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(value, "_alias_output_names", None),
            returns_scalar=self._returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
        )


class SparkLikeThen(SparkLikeExpr):
    def __init__(
        self: Self,
        call: SparkLikeWhen,
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[SparkLikeLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._backend_version = backend_version
        self._version = version
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._returns_scalar = returns_scalar

    def otherwise(self: Self, value: SparkLikeExpr | Any) -> SparkLikeExpr:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `SparkLikeWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self
