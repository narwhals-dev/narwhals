from __future__ import annotations

import functools
import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Sequence

from narwhals._duckdb.expr import DuckDBExpr
from narwhals._duckdb.utils import narwhals_to_native_dtype
from narwhals._expression_parsing import combine_root_names
from narwhals._expression_parsing import parse_into_exprs
from narwhals._expression_parsing import reduce_output_names
from narwhals.typing import CompliantNamespace

if TYPE_CHECKING:
    import duckdb

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.typing import IntoDuckDBExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version


def get_column_name(df: DuckDBLazyFrame, column: duckdb.Expression) -> str:
    return str(df._native_frame.select(column).columns[0])


class DuckDBNamespace(CompliantNamespace["duckdb.Expression"]):
    def __init__(self, *, backend_version: tuple[int, ...], version: Version) -> None:
        self._backend_version = backend_version
        self._version = version

    def all(self) -> DuckDBExpr:
        def _all(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            from duckdb import ColumnExpression

            return [ColumnExpression(col_name) for col_name in df.columns]

        return DuckDBExpr(
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

    def concat(
        self,
        items: Sequence[DuckDBLazyFrame],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> DuckDBLazyFrame:
        if how == "horizontal":
            msg = "horizontal concat not supported for duckdb. Please join instead"
            raise TypeError(msg)
        if how == "diagonal":
            msg = "Not implemented yet"
            raise NotImplementedError(msg)
        first = items[0]
        schema = first.schema
        if how == "vertical" and not all(x.schema == schema for x in items[1:]):
            msg = "inputs should all have the same schema"
            raise TypeError(msg)
        res = functools.reduce(
            lambda x, y: x.union(y), (item._native_frame for item in items)
        )
        return first._from_native_frame(res)

    def all_horizontal(self, *exprs: IntoDuckDBExpr) -> DuckDBExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [reduce(operator.and_, cols).alias(col_name)]

        return DuckDBExpr(
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

    def any_horizontal(self, *exprs: IntoDuckDBExpr) -> DuckDBExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [reduce(operator.or_, cols).alias(col_name)]

        return DuckDBExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="or_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def max_horizontal(self, *exprs: IntoDuckDBExpr) -> DuckDBExpr:
        from duckdb import FunctionExpression

        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [FunctionExpression("greatest", *cols).alias(col_name)]

        return DuckDBExpr(
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

    def min_horizontal(self, *exprs: IntoDuckDBExpr) -> DuckDBExpr:
        from duckdb import FunctionExpression

        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [FunctionExpression("least", *cols).alias(col_name)]

        return DuckDBExpr(
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

    def col(self, *column_names: str) -> DuckDBExpr:
        return DuckDBExpr.from_column_names(
            *column_names, backend_version=self._backend_version, version=self._version
        )

    def lit(self, value: Any, dtype: DType | None) -> DuckDBExpr:
        from duckdb import ConstantExpression

        def func(_df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            if dtype is not None:
                return [
                    ConstantExpression(value)
                    .cast(narwhals_to_native_dtype(dtype, version=self._version))
                    .alias("literal")
                ]
            return [ConstantExpression(value).alias("literal")]

        return DuckDBExpr(
            func,
            depth=0,
            function_name="lit",
            root_names=None,
            output_names=["literal"],
            returns_scalar=True,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def len(self) -> DuckDBExpr:
        def func(_df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            from duckdb import FunctionExpression

            return [FunctionExpression("count").alias("len")]

        return DuckDBExpr(
            call=func,
            depth=0,
            function_name="len",
            root_names=None,
            output_names=["len"],
            returns_scalar=True,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )
