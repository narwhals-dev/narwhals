from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

from narwhals._expression_parsing import combine_root_names
from narwhals._expression_parsing import parse_into_exprs
from narwhals._expression_parsing import reduce_output_names
from narwhals._spark_like.expr import SparkLikeExpr
from narwhals._spark_like.utils import get_column_name
from narwhals.typing import CompliantNamespace

if TYPE_CHECKING:
    from pyspark.sql import Column

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.typing import IntoSparkLikeExpr
    from narwhals.utils import Version


class SparkLikeNamespace(CompliantNamespace["Column"]):
    def __init__(self, *, backend_version: tuple[int, ...], version: Version) -> None:
        self._backend_version = backend_version
        self._version = version

    def all(self) -> SparkLikeExpr:
        def _all(df: SparkLikeLazyFrame) -> list[Column]:
            import pyspark.sql.functions as F  # noqa: N812

            return [F.col(col_name) for col_name in df.columns]

        return SparkLikeExpr(  # type: ignore[abstract]
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

    def all_horizontal(self, *exprs: IntoSparkLikeExpr) -> SparkLikeExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [reduce(operator.and_, cols).alias(col_name)]

        return SparkLikeExpr(  # type: ignore[abstract]
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

    def col(self, *column_names: str) -> SparkLikeExpr:
        return SparkLikeExpr.from_column_names(
            *column_names, backend_version=self._backend_version, version=self._version
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

        return SparkLikeExpr(  # type: ignore[abstract]
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
