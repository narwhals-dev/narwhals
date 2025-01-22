from __future__ import annotations

import functools
import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence
from typing import cast

from duckdb import CaseExpression
from duckdb import CoalesceOperator
from duckdb import ColumnExpression
from duckdb import ConstantExpression
from duckdb import FunctionExpression

from narwhals._duckdb.expr import DuckDBExpr
from narwhals._duckdb.utils import narwhals_to_native_dtype
from narwhals._expression_parsing import combine_root_names
from narwhals._expression_parsing import parse_into_exprs
from narwhals._expression_parsing import reduce_output_names
from narwhals.typing import CompliantNamespace

if TYPE_CHECKING:
    import duckdb
    from typing_extensions import Self

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.typing import IntoDuckDBExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version


def get_column_name(df: DuckDBLazyFrame, column: duckdb.Expression) -> str:
    return str(df._native_frame.select(column).columns[0])


class DuckDBNamespace(CompliantNamespace["duckdb.Expression"]):
    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._version = version

    def all(self: Self) -> DuckDBExpr:
        def _all(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
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
        self: Self,
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

    def concat_str(
        self: Self,
        exprs: Iterable[IntoDuckDBExpr],
        *more_exprs: IntoDuckDBExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> DuckDBExpr:
        parsed_exprs = [
            *parse_into_exprs(*exprs, namespace=self),
            *parse_into_exprs(*more_exprs, namespace=self),
        ]

        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = [s for _expr in parsed_exprs for s in _expr(df)]
            null_mask = [s.isnull() for _expr in parsed_exprs for s in _expr(df)]
            first_column_name = get_column_name(df, cols[0])

            if not ignore_nulls:
                null_mask_result = reduce(lambda x, y: x | y, null_mask)
                cols_separated = [
                    y
                    for x in [
                        (col.cast("string"),)
                        if i == len(cols) - 1
                        else (col.cast("string"), ConstantExpression(separator))
                        for i, col in enumerate(cols)
                    ]
                    for y in x
                ]
                result = CaseExpression(
                    condition=~null_mask_result,
                    value=FunctionExpression("concat", *cols_separated),
                )
            else:
                init_value, *values = [
                    CaseExpression(~nm, col.cast("string")).otherwise(
                        ConstantExpression("")
                    )
                    for col, nm in zip(cols, null_mask)
                ]
                separators = (
                    CaseExpression(nm, ConstantExpression("")).otherwise(
                        ConstantExpression(separator)
                    )
                    for nm in null_mask[:-1]
                )
                result = reduce(
                    lambda x, y: FunctionExpression("concat", x, y),
                    (
                        FunctionExpression("concat", s, v)
                        for s, v in zip(separators, values)
                    ),
                    init_value,
                )

            return [result.alias(first_column_name)]

        return DuckDBExpr(
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

    def all_horizontal(self: Self, *exprs: IntoDuckDBExpr) -> DuckDBExpr:
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

    def any_horizontal(self: Self, *exprs: IntoDuckDBExpr) -> DuckDBExpr:
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

    def max_horizontal(self: Self, *exprs: IntoDuckDBExpr) -> DuckDBExpr:
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

    def min_horizontal(self: Self, *exprs: IntoDuckDBExpr) -> DuckDBExpr:
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

    def sum_horizontal(self: Self, *exprs: IntoDuckDBExpr) -> DuckDBExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [
                reduce(
                    operator.add,
                    (CoalesceOperator(col, ConstantExpression(0)) for col in cols),
                ).alias(col_name)
            ]

        return DuckDBExpr(
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

    def mean_horizontal(self: Self, *exprs: IntoDuckDBExpr) -> DuckDBExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            cols = [c for _expr in parsed_exprs for c in _expr(df)]
            col_name = get_column_name(df, cols[0])
            return [
                (
                    reduce(
                        operator.add,
                        (CoalesceOperator(col, ConstantExpression(0)) for col in cols),
                    )
                    / reduce(operator.add, (col.isnotnull().cast("int") for col in cols))
                ).alias(col_name)
            ]

        return DuckDBExpr(
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

    def when(
        self: Self,
        *predicates: IntoDuckDBExpr,
    ) -> DuckDBWhen:
        plx = self.__class__(backend_version=self._backend_version, version=self._version)
        condition = plx.all_horizontal(*predicates)
        return DuckDBWhen(
            condition, self._backend_version, returns_scalar=False, version=self._version
        )

    def col(self: Self, *column_names: str) -> DuckDBExpr:
        return DuckDBExpr.from_column_names(
            *column_names, backend_version=self._backend_version, version=self._version
        )

    def nth(self: Self, *column_indices: int) -> DuckDBExpr:
        return DuckDBExpr.from_column_indices(
            *column_indices, backend_version=self._backend_version, version=self._version
        )

    def lit(self: Self, value: Any, dtype: DType | None) -> DuckDBExpr:
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

    def len(self: Self) -> DuckDBExpr:
        def func(_df: DuckDBLazyFrame) -> list[duckdb.Expression]:
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


class DuckDBWhen:
    def __init__(
        self: Self,
        condition: DuckDBExpr,
        backend_version: tuple[int, ...],
        then_value: Any = None,
        otherwise_value: Any = None,
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

    def __call__(self: Self, df: DuckDBLazyFrame) -> Sequence[duckdb.Expression]:
        from narwhals._expression_parsing import parse_into_expr

        plx = df.__narwhals_namespace__()
        condition = parse_into_expr(self._condition, namespace=plx)(df)[0]
        condition = cast("duckdb.Expression", condition)

        try:
            value = parse_into_expr(self._then_value, namespace=plx)(df)[0]
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression
            value = ConstantExpression(self._then_value).alias("literal")
        value = cast("duckdb.Expression", value)
        value_name = get_column_name(df, value)

        if self._otherwise_value is None:
            return [CaseExpression(condition=condition, value=value).alias(value_name)]
        try:
            otherwise_expr = parse_into_expr(self._otherwise_value, namespace=plx)
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression
            return [
                CaseExpression(condition=condition, value=value)
                .otherwise(ConstantExpression(self._otherwise_value))
                .alias(value_name)
            ]
        otherwise = otherwise_expr(df)[0]
        return [
            CaseExpression(condition=condition, value=value)
            .otherwise(otherwise)
            .alias(value_name)
        ]

    def then(self: Self, value: DuckDBExpr | Any) -> DuckDBThen:
        self._then_value = value

        return DuckDBThen(
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


class DuckDBThen(DuckDBExpr):
    def __init__(
        self: Self,
        call: DuckDBWhen,
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

    def otherwise(self: Self, value: DuckDBExpr | Any) -> DuckDBExpr:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `DuckDBWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self
