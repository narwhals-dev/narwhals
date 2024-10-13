from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import NoReturn

from narwhals._expression_parsing import combine_root_names
from narwhals._expression_parsing import parse_into_exprs
from narwhals._expression_parsing import reduce_output_names
from narwhals._pyspark.expr import PySparkExpr
from narwhals._pyspark.utils import get_column_name

if TYPE_CHECKING:
    from pyspark.sql import Column

    from narwhals._pyspark.dataframe import PySparkLazyFrame
    from narwhals._pyspark.typing import IntoPySparkExpr
    from narwhals.typing import DTypes


class PySparkNamespace:
    def __init__(self, *, dtypes: DTypes) -> None:
        self._dtypes = dtypes

    def _create_expr_from_series(self, _: Any) -> NoReturn:
        msg = "`_create_expr_from_series` for PySparkNamespace exists only for compatibility"
        raise NotImplementedError(msg)

    def _create_compliant_series(self, _: Any) -> NoReturn:
        msg = "`_create_compliant_series` for PySparkNamespace exists only for compatibility"
        raise NotImplementedError(msg)

    def _create_series_from_scalar(self, *_: Any) -> NoReturn:
        msg = "`_create_series_from_scalar` for PySparkNamespace exists only for compatibility"
        raise NotImplementedError(msg)

    def _create_expr_from_callable(  # pragma: no cover
        self,
        func: Callable[[PySparkLazyFrame], list[PySparkExpr]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
    ) -> PySparkExpr:
        msg = "`_create_expr_from_callable` for PySparkNamespace exists only for compatibility"
        raise NotImplementedError(msg)

    def all(self) -> PySparkExpr:
        def _all(df: PySparkLazyFrame) -> list[Column]:
            import pyspark.sql.functions as F  # noqa: N812

            return [F.col(col_name) for col_name in df.columns]

        return PySparkExpr(
            call=_all,
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
            returns_scalar=False,
            dtypes=self._dtypes,
        )

    def all_horizontal(self, *exprs: IntoPySparkExpr) -> PySparkExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: PySparkLazyFrame) -> list[Column]:
            cols = [c for _expr in parsed_exprs for c in _expr._call(df)]
            col_name = get_column_name(df, cols[0])
            return [reduce(operator.and_, cols).alias(col_name)]

        return PySparkExpr(
            call=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="all_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
            returns_scalar=False,
            dtypes=self._dtypes,
        )

    def col(self, *column_names: str) -> PySparkExpr:
        return PySparkExpr.from_column_names(*column_names, dtypes=self._dtypes)
