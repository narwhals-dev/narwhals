from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import cast

from narwhals._arrow.dataframe import ArrowDataFrame
from narwhals._arrow.expr import ArrowExpr
from narwhals._arrow.selectors import ArrowSelectorNamespace
from narwhals._arrow.series import ArrowSeries
from narwhals._arrow.utils import broadcast_series
from narwhals._arrow.utils import horizontal_concat
from narwhals._arrow.utils import vertical_concat
from narwhals._expression_parsing import combine_root_names
from narwhals._expression_parsing import parse_into_exprs
from narwhals._expression_parsing import reduce_output_names
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from typing import Callable

    from narwhals._arrow.typing import IntoArrowExpr
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes


class ArrowNamespace:
    def _create_expr_from_callable(
        self,
        func: Callable[[ArrowDataFrame], list[ArrowSeries]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
    ) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr(
            func,
            depth=depth,
            function_name=function_name,
            root_names=root_names,
            output_names=output_names,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def _create_expr_from_series(self, series: ArrowSeries) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr(
            lambda _df: [series],
            depth=0,
            function_name="series",
            root_names=None,
            output_names=None,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def _create_series_from_scalar(self, value: Any, series: ArrowSeries) -> ArrowSeries:
        from narwhals._arrow.series import ArrowSeries

        if self._backend_version < (13,) and hasattr(value, "as_py"):  # pragma: no cover
            value = value.as_py()
        return ArrowSeries._from_iterable(
            [value],
            name=series.name,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def _create_compliant_series(self, value: Any) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()

        from narwhals._arrow.series import ArrowSeries

        return ArrowSeries(
            native_series=pa.chunked_array([value]),
            name="",
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    # --- not in spec ---
    def __init__(self, *, backend_version: tuple[int, ...], dtypes: DTypes) -> None:
        self._backend_version = backend_version
        self._implementation = Implementation.PYARROW
        self._dtypes = dtypes

    # --- selection ---
    def col(self, *column_names: str) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr.from_column_names(
            *column_names, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def nth(self, *column_indices: int) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr

        return ArrowExpr.from_column_indices(
            *column_indices, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def len(self) -> ArrowExpr:
        # coverage bug? this is definitely hit
        return ArrowExpr(  # pragma: no cover
            lambda df: [
                ArrowSeries._from_iterable(
                    [len(df._native_frame)],
                    name="len",
                    backend_version=self._backend_version,
                    dtypes=self._dtypes,
                )
            ],
            depth=0,
            function_name="len",
            root_names=None,
            output_names=["len"],
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def all(self) -> ArrowExpr:
        from narwhals._arrow.expr import ArrowExpr
        from narwhals._arrow.series import ArrowSeries

        return ArrowExpr(
            lambda df: [
                ArrowSeries(
                    df._native_frame[column_name],
                    name=column_name,
                    backend_version=df._backend_version,
                    dtypes=df._dtypes,
                )
                for column_name in df.columns
            ],
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def lit(self, value: Any, dtype: DType | None) -> ArrowExpr:
        def _lit_arrow_series(_: ArrowDataFrame) -> ArrowSeries:
            arrow_series = ArrowSeries._from_iterable(
                data=[value],
                name="lit",
                backend_version=self._backend_version,
                dtypes=self._dtypes,
            )
            if dtype:
                return arrow_series.cast(dtype)
            return arrow_series

        return ArrowExpr(
            lambda df: [_lit_arrow_series(df)],
            depth=0,
            function_name="lit",
            root_names=None,
            output_names=["lit"],
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def all_horizontal(self, *exprs: IntoArrowExpr) -> ArrowExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = (s for _expr in parsed_exprs for s in _expr._call(df))
            return [reduce(lambda x, y: x & y, series)]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="all_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
        )

    def any_horizontal(self, *exprs: IntoArrowExpr) -> ArrowExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = (s for _expr in parsed_exprs for s in _expr._call(df))
            return [reduce(lambda x, y: x | y, series)]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="any_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
        )

    def sum_horizontal(self, *exprs: IntoArrowExpr) -> ArrowExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = (s.fill_null(0) for _expr in parsed_exprs for s in _expr._call(df))
            return [reduce(lambda x, y: x + y, series)]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="sum_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
        )

    def mean_horizontal(self, *exprs: IntoArrowExpr) -> IntoArrowExpr:
        parsed_exprs = parse_into_exprs(*exprs, namespace=self)

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = (s.fill_null(0) for _expr in parsed_exprs for s in _expr._call(df))
            non_na = (
                1 - s.is_null().cast(self._dtypes.Int64())
                for _expr in parsed_exprs
                for s in _expr._call(df)
            )
            return [
                reduce(lambda x, y: x + y, series) / reduce(lambda x, y: x + y, non_na)
            ]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="mean_horizontal",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
        )

    def concat(
        self,
        items: Iterable[ArrowDataFrame],
        *,
        how: Literal["horizontal", "vertical"],
    ) -> ArrowDataFrame:
        dfs: list[Any] = [item._native_frame for item in items]

        if how == "horizontal":
            return ArrowDataFrame(
                horizontal_concat(dfs),
                backend_version=self._backend_version,
                dtypes=self._dtypes,
            )
        if how == "vertical":
            return ArrowDataFrame(
                vertical_concat(dfs),
                backend_version=self._backend_version,
                dtypes=self._dtypes,
            )
        raise NotImplementedError

    def sum(self, *column_names: str) -> ArrowExpr:
        return ArrowExpr.from_column_names(
            *column_names, backend_version=self._backend_version, dtypes=self._dtypes
        ).sum()

    def mean(self, *column_names: str) -> ArrowExpr:
        return ArrowExpr.from_column_names(
            *column_names, backend_version=self._backend_version, dtypes=self._dtypes
        ).mean()

    def max(self, *column_names: str) -> ArrowExpr:
        return ArrowExpr.from_column_names(
            *column_names, backend_version=self._backend_version, dtypes=self._dtypes
        ).max()

    def min(self, *column_names: str) -> ArrowExpr:
        return ArrowExpr.from_column_names(
            *column_names, backend_version=self._backend_version, dtypes=self._dtypes
        ).min()

    @property
    def selectors(self) -> ArrowSelectorNamespace:
        return ArrowSelectorNamespace(
            backend_version=self._backend_version, dtypes=self._dtypes
        )

    def when(
        self,
        *predicates: IntoArrowExpr,
    ) -> ArrowWhen:
        plx = self.__class__(backend_version=self._backend_version, dtypes=self._dtypes)
        if predicates:
            condition = plx.all_horizontal(*predicates)
        else:
            msg = "at least one predicate needs to be provided"
            raise TypeError(msg)

        return ArrowWhen(condition, self._backend_version, dtypes=self._dtypes)

    def concat_str(
        self,
        exprs: Iterable[IntoArrowExpr],
        *more_exprs: IntoArrowExpr,
        separator: str = "",
        ignore_nulls: bool = False,
    ) -> ArrowExpr:
        import pyarrow.compute as pc  # ignore-banned-import

        parsed_exprs: list[ArrowExpr] = [
            *parse_into_exprs(*exprs, namespace=self),
            *parse_into_exprs(*more_exprs, namespace=self),
        ]

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = (
                s._native_series
                for _expr in parsed_exprs
                for s in _expr.cast(self._dtypes.String())._call(df)
            )
            null_handling = "skip" if ignore_nulls else "emit_null"
            result_series = pc.binary_join_element_wise(
                *series, separator, null_handling=null_handling
            )
            return [
                ArrowSeries(
                    native_series=result_series,
                    name="",
                    backend_version=self._backend_version,
                    dtypes=self._dtypes,
                )
            ]

        return self._create_expr_from_callable(
            func=func,
            depth=max(x._depth for x in parsed_exprs) + 1,
            function_name="concat_str",
            root_names=combine_root_names(parsed_exprs),
            output_names=reduce_output_names(parsed_exprs),
        )


class ArrowWhen:
    def __init__(
        self,
        condition: ArrowExpr,
        backend_version: tuple[int, ...],
        then_value: Any = None,
        otherwise_value: Any = None,
        *,
        dtypes: DTypes,
    ) -> None:
        self._backend_version = backend_version
        self._condition = condition
        self._then_value = then_value
        self._otherwise_value = otherwise_value
        self._dtypes = dtypes

    def __call__(self, df: ArrowDataFrame) -> list[ArrowSeries]:
        import pyarrow as pa  # ignore-banned-import
        import pyarrow.compute as pc  # ignore-banned-import

        from narwhals._arrow.namespace import ArrowNamespace
        from narwhals._expression_parsing import parse_into_expr

        plx = ArrowNamespace(backend_version=self._backend_version, dtypes=self._dtypes)

        condition = parse_into_expr(self._condition, namespace=plx)._call(df)[0]  # type: ignore[arg-type]
        try:
            value_series = parse_into_expr(self._then_value, namespace=plx)._call(df)[0]  # type: ignore[arg-type]
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression
            value_series = condition.__class__._from_iterable(  # type: ignore[call-arg]
                [self._then_value] * len(condition),
                name="literal",
                backend_version=self._backend_version,
                dtypes=self._dtypes,
            )
        value_series = cast(ArrowSeries, value_series)

        value_series_native = value_series._native_series
        condition_native = condition._native_series.combine_chunks()

        if self._otherwise_value is None:
            otherwise_native = pa.array(
                [None] * len(condition_native), type=value_series_native.type
            )
            return [
                value_series._from_native_series(
                    pc.if_else(condition_native, value_series_native, otherwise_native)
                )
            ]
        try:
            otherwise_series = parse_into_expr(
                self._otherwise_value, namespace=plx
            )._call(df)[0]  # type: ignore[arg-type]
        except TypeError:
            # `self._otherwise_value` is a scalar and can't be converted to an expression.
            # Remark that string values _are_ converted into expressions!
            return [
                value_series._from_native_series(
                    pc.if_else(
                        condition_native, value_series_native, self._otherwise_value
                    )
                )
            ]
        else:
            otherwise_series = cast(ArrowSeries, otherwise_series)
            condition = cast(ArrowSeries, condition)
            condition_native, otherwise_native = broadcast_series(
                [condition, otherwise_series]
            )
            return [
                value_series._from_native_series(
                    pc.if_else(condition_native, value_series_native, otherwise_native)
                )
            ]

    def then(self, value: ArrowExpr | ArrowSeries | Any) -> ArrowThen:
        self._then_value = value

        return ArrowThen(
            self,
            depth=0,
            function_name="whenthen",
            root_names=None,
            output_names=None,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )


class ArrowThen(ArrowExpr):
    def __init__(
        self,
        call: ArrowWhen,
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> None:
        self._backend_version = backend_version
        self._dtypes = dtypes
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names

    def otherwise(self, value: ArrowExpr | ArrowSeries | Any) -> ArrowExpr:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `PandasWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self
