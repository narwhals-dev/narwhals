from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Sequence

import pyarrow as pa
import pyarrow.compute as pc

from narwhals._arrow.dataframe import ArrowDataFrame
from narwhals._arrow.expr import ArrowExpr
from narwhals._arrow.selectors import ArrowSelectorNamespace
from narwhals._arrow.series import ArrowSeries
from narwhals._arrow.utils import diagonal_concat
from narwhals._arrow.utils import horizontal_concat
from narwhals._arrow.utils import vertical_concat
from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals.typing import CompliantNamespace
from narwhals.utils import Implementation
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from typing import Callable

    from typing_extensions import Self

    from narwhals._arrow.typing import IntoArrowExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class ArrowNamespace(CompliantNamespace[ArrowSeries]):
    def _create_expr_from_callable(
        self: Self,
        func: Callable[[ArrowDataFrame], Sequence[ArrowSeries]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[ArrowDataFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        kwargs: dict[str, Any],
    ) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr(
            func,
            depth=depth,
            function_name=function_name,
            evaluate_output_names=evaluate_output_names,
            alias_output_names=alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            kwargs=kwargs,
        )

    def _create_expr_from_series(self: Self, series: ArrowSeries) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr(
            lambda _df: [series],
            depth=0,
            function_name="series",
            evaluate_output_names=lambda _df: [series.name],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def _create_series_from_scalar(
        self: Self, value: Any, *, reference_series: ArrowSeries
    ) -> ArrowSeries:
        from narwhals._arrow.series import ArrowSeries

        if self._backend_version < (13,) and hasattr(value, "as_py"):
            value = value.as_py()
        return ArrowSeries._from_iterable(
            [value],
            name=reference_series.name,
            backend_version=self._backend_version,
            version=self._version,
        )

    def _create_compliant_series(self: Self, value: Any) -> ArrowSeries:
        from narwhals._arrow.series import ArrowSeries

        return ArrowSeries(
            native_series=pa.chunked_array([value]),
            name="",
            backend_version=self._backend_version,
            version=self._version,
        )

    # --- not in spec ---
    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._implementation = Implementation.PYARROW
        self._version = version

    # --- selection ---
    def col(self: Self, *column_names: str) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr.from_column_names(
            *column_names, backend_version=self._backend_version, version=self._version
        )

    def nth(self: Self, *column_indices: int) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr.from_column_indices(
            *column_indices, backend_version=self._backend_version, version=self._version
        )

    def len(self: Self) -> ArrowExpr:
        # coverage bug? this is definitely hit
        return ArrowExpr(  # pragma: no cover
            lambda df: [
                ArrowSeries._from_iterable(
                    [len(df._native_frame)],
                    name="len",
                    backend_version=self._backend_version,
                    version=self._version,
                )
            ],
            depth=0,
            function_name="len",
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def all(self: Self) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr
        from narwhals._arrow.series import ArrowSeries

        return ArrowExpr(
            lambda df: [
                ArrowSeries(
                    df._native_frame[column_name],
                    name=column_name,
                    backend_version=df._backend_version,
                    version=df._version,
                )
                for column_name in df.columns
            ],
            depth=0,
            function_name="all",
            evaluate_output_names=lambda df: df.columns,
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def lit(self: Self, value: Any, dtype: DType | None) -> ArrowExpr:
        def _lit_arrow_series(_: ArrowDataFrame) -> ArrowSeries:
            arrow_series = ArrowSeries._from_iterable(
                data=[value],
                name="literal",
                backend_version=self._backend_version,
                version=self._version,
            )
            if dtype:
                return arrow_series.cast(dtype)
            return arrow_series

        return ArrowExpr(
            lambda df: [_lit_arrow_series(df)],
            depth=0,
            function_name="lit",
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def all_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = (s for _expr in exprs for s in _expr(df))
            return [reduce(operator.and_, series)]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="all_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            kwargs={"exprs": exprs},
        )

    def any_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = (s for _expr in exprs for s in _expr(df))
            return [reduce(operator.or_, series)]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="any_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            kwargs={"exprs": exprs},
        )

    def sum_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = (
                s.fill_null(0, strategy=None, limit=None)
                for _expr in exprs
                for s in _expr(df)
            )
            return [reduce(operator.add, series)]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="sum_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            kwargs={"exprs": exprs},
        )

    def mean_horizontal(self: Self, *exprs: ArrowExpr) -> IntoArrowExpr:
        dtypes = import_dtypes_module(self._version)

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = (
                s.fill_null(0, strategy=None, limit=None)
                for _expr in exprs
                for s in _expr(df)
            )
            non_na = (
                1 - s.is_null().cast(dtypes.Int64()) for _expr in exprs for s in _expr(df)
            )
            return [reduce(operator.add, series) / reduce(operator.add, non_na)]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="mean_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            kwargs={"exprs": exprs},
        )

    def min_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            init_series, *series = [s for _expr in exprs for s in _expr(df)]
            return [
                ArrowSeries(
                    native_series=reduce(
                        pc.min_element_wise,
                        [s._native_series for s in series],
                        init_series._native_series,
                    ),
                    name=init_series.name,
                    backend_version=self._backend_version,
                    version=self._version,
                )
            ]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="min_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            kwargs={"exprs": exprs},
        )

    def max_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            init_series, *series = [s for _expr in exprs for s in _expr(df)]
            return [
                ArrowSeries(
                    native_series=reduce(
                        pc.max_element_wise,
                        [s._native_series for s in series],
                        init_series._native_series,
                    ),
                    name=init_series.name,
                    backend_version=self._backend_version,
                    version=self._version,
                )
            ]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="max_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            kwargs={"exprs": exprs},
        )

    def concat(
        self: Self,
        items: Iterable[ArrowDataFrame],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> ArrowDataFrame:
        dfs = [item._native_frame for item in items]

        if not dfs:
            msg = "No dataframes to concatenate"  # pragma: no cover
            raise AssertionError(msg)

        if how == "horizontal":
            result_table = horizontal_concat(dfs)
        elif how == "vertical":
            result_table = vertical_concat(dfs)
        elif how == "diagonal":
            result_table = diagonal_concat(dfs, self._backend_version)
        else:
            raise NotImplementedError

        return ArrowDataFrame(
            result_table,
            backend_version=self._backend_version,
            version=self._version,
            validate_column_names=True,
        )

    @property
    def selectors(self: Self) -> ArrowSelectorNamespace:
        return ArrowSelectorNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def when(self: Self, predicate: ArrowExpr) -> ArrowWhen:
        return ArrowWhen(predicate, self._backend_version, version=self._version)

    def concat_str(
        self: Self,
        *exprs: ArrowExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> ArrowExpr:
        dtypes = import_dtypes_module(self._version)

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            compliant_series_list = [
                s for _expr in exprs for s in _expr.cast(dtypes.String())(df)
            ]
            null_handling = "skip" if ignore_nulls else "emit_null"
            result_series = pc.binary_join_element_wise(
                *(s._native_series for s in compliant_series_list),
                separator,
                null_handling=null_handling,
            )
            return [
                ArrowSeries(
                    native_series=result_series,
                    name=compliant_series_list[0].name,
                    backend_version=self._backend_version,
                    version=self._version,
                )
            ]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="concat_str",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            kwargs={
                "exprs": exprs,
                "separator": separator,
                "ignore_nulls": ignore_nulls,
            },
        )


class ArrowWhen:
    def __init__(
        self: Self,
        condition: ArrowExpr,
        backend_version: tuple[int, ...],
        then_value: Any = None,
        otherwise_value: Any = None,
        *,
        version: Version,
    ) -> None:
        self._backend_version = backend_version
        self._condition = condition
        self._then_value = then_value
        self._otherwise_value = otherwise_value
        self._version = version

    def __call__(self: Self, df: ArrowDataFrame) -> Sequence[ArrowSeries]:
        condition = self._condition(df)[0]

        value_series = self._then_value(df)[0]
        condition_native, value_series_native = (
            condition._native_series,
            value_series._native_series,
        )

        if self._otherwise_value is None:
            otherwise_native = pa.repeat(
                pa.scalar(None, type=value_series_native.type), len(condition_native)
            )
            return [
                value_series._from_native_series(
                    pc.if_else(condition_native, value_series_native, otherwise_native)
                )
            ]
        otherwise_expr = self._otherwise_value
        otherwise_native = otherwise_expr(df)[0]._native_series
        return [
            value_series._from_native_series(
                pc.if_else(condition_native, value_series_native, otherwise_native)
            )
        ]

    def then(self: Self, value: ArrowExpr | ArrowSeries | Any) -> ArrowThen:
        self._then_value = value

        return ArrowThen(
            self,
            depth=0,
            function_name="whenthen",
            evaluate_output_names=getattr(
                value, "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(value, "_alias_output_names", None),
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"value": value},
        )


class ArrowThen(ArrowExpr):
    def __init__(
        self: Self,
        call: ArrowWhen,
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[ArrowDataFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None:
        self._backend_version = backend_version
        self._version = version
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._kwargs = kwargs

    def otherwise(self: Self, value: ArrowExpr | ArrowSeries | Any) -> ArrowExpr:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `PandasWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self
