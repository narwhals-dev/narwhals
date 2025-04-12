from __future__ import annotations

import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

import pyarrow as pa
import pyarrow.compute as pc

from narwhals._arrow.dataframe import ArrowDataFrame
from narwhals._arrow.expr import ArrowExpr
from narwhals._arrow.selectors import ArrowSelectorNamespace
from narwhals._arrow.series import ArrowSeries
from narwhals._arrow.utils import align_series_full_broadcast
from narwhals._arrow.utils import cast_to_comparable_string_types
from narwhals._compliant import CompliantThen
from narwhals._compliant import EagerNamespace
from narwhals._compliant import EagerWhen
from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals.utils import Implementation
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.typing import ArrowChunkedArray
    from narwhals._arrow.typing import Incomplete
    from narwhals.dtypes import DType
    from narwhals.typing import ConcatMethod
    from narwhals.utils import Version


class ArrowNamespace(
    EagerNamespace[
        ArrowDataFrame, ArrowSeries, ArrowExpr, "pa.Table", "ArrowChunkedArray"
    ]
):
    @property
    def _dataframe(self) -> type[ArrowDataFrame]:
        return ArrowDataFrame

    @property
    def _expr(self) -> type[ArrowExpr]:
        return ArrowExpr

    @property
    def _series(self) -> type[ArrowSeries]:
        return ArrowSeries

    # --- not in spec ---
    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._implementation = Implementation.PYARROW
        self._version = version

    def len(self: Self) -> ArrowExpr:
        # coverage bug? this is definitely hit
        return self._expr(  # pragma: no cover
            lambda df: [
                ArrowSeries.from_iterable([len(df.native)], name="len", context=self)
            ],
            depth=0,
            function_name="len",
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )

    def lit(self: Self, value: Any, dtype: DType | type[DType] | None) -> ArrowExpr:
        def _lit_arrow_series(_: ArrowDataFrame) -> ArrowSeries:
            arrow_series = ArrowSeries.from_iterable(
                data=[value], name="literal", context=self
            )
            if dtype:
                return arrow_series.cast(dtype)
            return arrow_series

        return self._expr(
            lambda df: [_lit_arrow_series(df)],
            depth=0,
            function_name="lit",
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )

    def all_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = chain.from_iterable(expr(df) for expr in exprs)
            return [reduce(operator.and_, align_series_full_broadcast(*series))]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="all_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def any_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            series = chain.from_iterable(expr(df) for expr in exprs)
            return [reduce(operator.or_, align_series_full_broadcast(*series))]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="any_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def sum_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            it = chain.from_iterable(expr(df) for expr in exprs)
            series = (s.fill_null(0, strategy=None, limit=None) for s in it)
            return [reduce(operator.add, align_series_full_broadcast(*series))]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="sum_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def mean_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        dtypes = import_dtypes_module(self._version)

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            expr_results = list(chain.from_iterable(expr(df) for expr in exprs))
            series = align_series_full_broadcast(
                *(s.fill_null(0, strategy=None, limit=None) for s in expr_results)
            )
            non_na = align_series_full_broadcast(
                *(1 - s.is_null().cast(dtypes.Int64()) for s in expr_results)
            )
            return [reduce(operator.add, series) / reduce(operator.add, non_na)]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="mean_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def min_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            init_series, *series = list(chain.from_iterable(expr(df) for expr in exprs))
            init_series, *series = align_series_full_broadcast(init_series, *series)
            native_series = reduce(
                pc.min_element_wise, [s.native for s in series], init_series.native
            )
            return [
                ArrowSeries(
                    native_series,
                    name=init_series.name,
                    backend_version=self._backend_version,
                    version=self._version,
                )
            ]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="min_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    def max_horizontal(self: Self, *exprs: ArrowExpr) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            init_series, *series = list(chain.from_iterable(expr(df) for expr in exprs))
            init_series, *series = align_series_full_broadcast(init_series, *series)
            native_series = reduce(
                pc.max_element_wise, [s.native for s in series], init_series.native
            )
            return [
                ArrowSeries(
                    native_series,
                    name=init_series.name,
                    backend_version=self._backend_version,
                    version=self._version,
                )
            ]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="max_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )

    # NOTE: Stub issue fixed in https://github.com/zen-xu/pyarrow-stubs/pull/203
    def _concat_diagonal(self, dfs: Sequence[pa.Table], /) -> pa.Table:
        if self._backend_version >= (14,):
            return pa.concat_tables(dfs, promote_options="default")  # type: ignore[arg-type]
        return pa.concat_tables(dfs, promote=True)  # type: ignore[arg-type] # pragma: no cover

    def _concat_horizontal(self, dfs: Sequence[pa.Table], /) -> pa.Table:
        names = [name for df in dfs for name in df.column_names]

        # NOTE: Think this is redundant, since we have `validate_column_names=True` later
        if len(set(names)) < len(names):  # pragma: no cover
            msg = "Expected unique column names"
            raise ValueError(msg)
        arrays = list(chain.from_iterable(df.itercolumns() for df in dfs))
        return pa.Table.from_arrays(arrays, names=names)

    def _concat_vertical(self, dfs: Sequence[pa.Table], /) -> pa.Table:
        cols_0 = dfs[0].column_names
        for i, df in enumerate(dfs[1:], start=1):
            cols_current = df.column_names
            if cols_current != cols_0:
                msg = (
                    "unable to vstack, column names don't match:\n"
                    f"   - dataframe 0: {cols_0}\n"
                    f"   - dataframe {i}: {cols_current}\n"
                )
                raise TypeError(msg)
        return pa.concat_tables(dfs)  # type: ignore[arg-type]

    def concat(
        self, items: Iterable[ArrowDataFrame], *, how: ConcatMethod
    ) -> ArrowDataFrame:
        dfs = [item.native for item in items]
        if how == "horizontal":
            native = self._concat_horizontal(dfs)
        elif how == "vertical":
            native = self._concat_vertical(dfs)
        elif how == "diagonal":
            native = self._concat_diagonal(dfs)
        else:
            raise NotImplementedError
        return self._dataframe.from_native(native, context=self)

    @property
    def selectors(self: Self) -> ArrowSelectorNamespace:
        return ArrowSelectorNamespace.from_namespace(self)

    def when(self: Self, predicate: ArrowExpr) -> ArrowWhen:
        return ArrowWhen.from_expr(predicate, context=self)

    def concat_str(
        self: Self,
        *exprs: ArrowExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> ArrowExpr:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            compliant_series_list = align_series_full_broadcast(
                *(chain.from_iterable(expr(df) for expr in exprs))
            )
            name = compliant_series_list[0].name
            null_handling: Literal["skip", "emit_null"] = (
                "skip" if ignore_nulls else "emit_null"
            )
            it, separator_scalar = cast_to_comparable_string_types(
                *(s.native for s in compliant_series_list), separator=separator
            )
            # NOTE: stubs indicate `separator` must also be a `ChunkedArray`
            # Reality: `str` is fine
            concat_str: Incomplete = pc.binary_join_element_wise
            compliant = self._series(
                concat_str(*it, separator_scalar, null_handling=null_handling),
                name=name,
                backend_version=self._backend_version,
                version=self._version,
            )
            return [compliant]

        return self._expr._from_callable(
            func=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="concat_str",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            context=self,
        )


class ArrowWhen(EagerWhen[ArrowDataFrame, ArrowSeries, ArrowExpr, "ArrowChunkedArray"]):
    @property
    def _then(self) -> type[ArrowThen]:
        return ArrowThen

    def _if_then_else(
        self, when: ArrowChunkedArray, then: ArrowChunkedArray, otherwise: Any, /
    ) -> ArrowChunkedArray:
        otherwise = pa.nulls(len(when), then.type) if otherwise is None else otherwise
        return pc.if_else(when, then, otherwise)


class ArrowThen(CompliantThen[ArrowDataFrame, ArrowSeries, ArrowExpr], ArrowExpr): ...
