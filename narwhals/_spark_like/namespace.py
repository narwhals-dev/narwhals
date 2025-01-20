from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal

from pyspark.sql import functions as F  # noqa: N812

from narwhals._expression_parsing import combine_root_names
from narwhals._expression_parsing import parse_into_expr
from narwhals._expression_parsing import parse_into_exprs
from narwhals._expression_parsing import reduce_output_names
from narwhals._spark_like.dataframe import SparkLikeLazyFrame
from narwhals._spark_like.expr import SparkLikeExpr
from narwhals._spark_like.utils import get_column_name
from narwhals.typing import CompliantNamespace

if TYPE_CHECKING:
    from pyspark.sql import Column
    from pyspark.sql import DataFrame

    from narwhals._spark_like.typing import IntoSparkLikeExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class SparkLikeNamespace(CompliantNamespace["Column"]):
    def __init__(self, *, backend_version: tuple[int, ...], version: Version) -> None:
        self._backend_version = backend_version
        self._version = version

    def all(self) -> SparkLikeExpr:
        def _all(df: SparkLikeLazyFrame) -> list[Column]:
            import pyspark.sql.functions as F  # noqa: N812

            return [F.col(col_name) for col_name in df.columns]

        return SparkLikeExpr(
            call=_all,
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def col(self, *column_names: str) -> SparkLikeExpr:
        return SparkLikeExpr.from_column_names(
            *column_names, backend_version=self._backend_version, version=self._version
        )

    def nth(self, *column_indices: int) -> SparkLikeExpr:
        return SparkLikeExpr.from_column_indices(
            *column_indices, backend_version=self._backend_version, version=self._version
        )

    def lit(self, value: object, dtype: DType | None) -> SparkLikeExpr:
        if dtype is not None:
            msg = "todo"
            raise NotImplementedError(msg)

        def _lit(_: SparkLikeLazyFrame) -> list[Column]:
            import pyspark.sql.functions as F  # noqa: N812

            return [F.lit(value).alias("literal")]

        return SparkLikeExpr(
            call=_lit,
            depth=0,
            function_name="lit",
            root_names=None,
            output_names=["literal"],
            returns_scalar=True,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def len(self) -> SparkLikeExpr:
        def func(_: SparkLikeLazyFrame) -> list[Column]:
            import pyspark.sql.functions as F  # noqa: N812

            return [F.count("*").alias("len")]

        return SparkLikeExpr(
            func,
            depth=0,
            function_name="len",
            root_names=None,
            output_names=["len"],
            returns_scalar=True,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def all_horizontal(self, *exprs: IntoSparkLikeExpr) -> SparkLikeExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [reduce(operator.and_, cols).alias(col_name)]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="all_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def any_horizontal(self, *exprs: IntoSparkLikeExpr) -> SparkLikeExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [reduce(operator.or_, cols).alias(col_name)]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="any_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def sum_horizontal(self, *exprs: IntoSparkLikeExpr) -> SparkLikeExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            import pyspark.sql.functions as F  # noqa: N812

            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [
                reduce(
                    operator.add,
                    (F.coalesce(col, F.lit(0)) for col in cols),
                ).alias(col_name)
            ]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="sum_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def mean_horizontal(self, *exprs: IntoSparkLikeExpr) -> SparkLikeExpr:
        from pyspark.sql.types import IntegerType

        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [
                (
                    reduce(operator.add, (F.coalesce(col, F.lit(0)) for col in cols))
                    / reduce(
                        operator.add,
                        (col.isNotNull().cast(IntegerType()) for col in cols),
                    )
                ).alias(col_name)
            ]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="mean_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def max_horizontal(self, *exprs: IntoSparkLikeExpr) -> SparkLikeExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [F.greatest(*cols).alias(col_name)]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="max_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def min_horizontal(self, *exprs: IntoSparkLikeExpr) -> SparkLikeExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [F.least(*cols).alias(col_name)]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="min_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def concat(
        self,
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
        self,
        exprs: Iterable[IntoSparkLikeExpr],
        *more_exprs: IntoSparkLikeExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> SparkLikeExpr:
        from pyspark.sql.types import StringType

        parsed_exprs = [
            *parse_into_exprs(*exprs, namespace=self),
            *parse_into_exprs(*more_exprs, namespace=self),
        ]

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [s for _expr in parsed_exprs for s in _expr(df)]
            cols_casted = [s.cast(StringType()) for s in cols]
            null_mask = [F.isnull(s) for _expr in parsed_exprs for s in _expr(df)]
            first_column_name = get_column_name(df, cols[0])

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

            return [result.alias(first_column_name)]

        return SparkLikeExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="concat_str",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={
                "exprs": exprs,
                "more_exprs": more_exprs,
                "separator": separator,
                "ignore_nulls": ignore_nulls,
            },
        )

    def when(self, *predicates: IntoSparkLikeExpr) -> SparkLikeWhen:
        plx = self.__class__(backend_version=self._backend_version, version=self._version)
        condition = plx.all_horizontal(*predicates)
        return SparkLikeWhen(
            condition, self._backend_version, returns_scalar=False, version=self._version
        )


class SparkLikeWhen:
    def __init__(
        self,
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

    def __call__(self, df: SparkLikeLazyFrame) -> list[Column]:
        plx = df.__narwhals_namespace__()
        condition = parse_into_expr(self._condition, namespace=plx)(df)[0]

        try:
            value_ = parse_into_expr(self._then_value, namespace=plx)(df)[0]
            col_name = get_column_name(df, value_)
        except TypeError:
            # `self._then_value` is a scalar and can't be converted to an expression
            value_ = F.lit(self._then_value)
            col_name = "literal"

        try:
            other_ = parse_into_expr(self._otherwise_value, namespace=plx)(df)[0]
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression
            other_ = F.lit(self._otherwise_value)

        return [
            F.when(condition=condition, value=value_)
            .otherwise(value=other_)
            .alias(col_name)
        ]

    def then(self, value: SparkLikeExpr | Any) -> SparkLikeThen:
        self._then_value = value

        return SparkLikeThen(
            self,
            depth=0,
            function_name="whenthen",
            root_names=None,
            output_names=None,
            returns_scalar=self._returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"value": value},
        )


class SparkLikeThen(SparkLikeExpr):
    def __init__(
        self,
        call: SparkLikeWhen,
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None:
        self._backend_version = backend_version
        self._version = version
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names
        self._returns_scalar = returns_scalar
        self._kwargs = kwargs

    def otherwise(self, value: SparkLikeExpr | Any) -> SparkLikeExpr:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `SparkLikeWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self
