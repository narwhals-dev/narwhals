from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import narwhals_to_native_dtype
from narwhals._plan.literal import is_literal_scalar
from narwhals._plan.protocols import EagerNamespace
from narwhals._utils import Version
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from narwhals._arrow.typing import ChunkedArrayAny
    from narwhals._plan import expr, functions as F  # noqa: N812
    from narwhals._plan.arrow.dataframe import ArrowDataFrame
    from narwhals._plan.arrow.expr import ArrowExpr, ArrowScalar
    from narwhals._plan.arrow.series import ArrowSeries
    from narwhals._plan.boolean import AllHorizontal, AnyHorizontal
    from narwhals._plan.dummy import DummySeries
    from narwhals._plan.expr import FunctionExpr, RangeExpr
    from narwhals._plan.ranges import IntRange
    from narwhals._plan.strings import ConcatHorizontal
    from narwhals.typing import NonNestedLiteral, PythonLiteral


class ArrowNamespace(
    EagerNamespace["ArrowDataFrame", "ArrowSeries", "ArrowExpr", "ArrowScalar"]
):
    def __init__(self, version: Version = Version.MAIN) -> None:
        self._version = version

    @property
    def _expr(self) -> type[ArrowExpr]:
        from narwhals._plan.arrow.expr import ArrowExpr

        return ArrowExpr

    @property
    def _scalar(self) -> type[ArrowScalar]:
        from narwhals._plan.arrow.expr import ArrowScalar

        return ArrowScalar

    @property
    def _series(self) -> type[ArrowSeries]:
        from narwhals._plan.arrow.series import ArrowSeries

        return ArrowSeries

    @property
    def _dataframe(self) -> type[ArrowDataFrame]:
        from narwhals._plan.arrow.dataframe import ArrowDataFrame

        return ArrowDataFrame

    def col(self, node: expr.Column, frame: ArrowDataFrame, name: str) -> ArrowExpr:
        return self._expr.from_native(
            frame.native.column(node.name), name, version=frame.version
        )

    @overload
    def lit(
        self, node: expr.Literal[NonNestedLiteral], frame: ArrowDataFrame, name: str
    ) -> ArrowScalar: ...

    @overload
    def lit(
        self,
        node: expr.Literal[DummySeries[ChunkedArrayAny]],
        frame: ArrowDataFrame,
        name: str,
    ) -> ArrowExpr: ...

    @overload
    def lit(
        self,
        node: expr.Literal[NonNestedLiteral] | expr.Literal[DummySeries[ChunkedArrayAny]],
        frame: ArrowDataFrame,
        name: str,
    ) -> ArrowExpr | ArrowScalar: ...

    def lit(
        self,
        node: expr.Literal[NonNestedLiteral] | expr.Literal[DummySeries[ChunkedArrayAny]],
        frame: ArrowDataFrame,
        name: str,
    ) -> ArrowExpr | ArrowScalar:
        if is_literal_scalar(node):
            return self._scalar.from_python(
                node.unwrap(), name, dtype=node.dtype, version=frame.version
            )
        nw_ser = node.unwrap()
        return self._expr.from_native(
            nw_ser.to_native(), name or node.name, nw_ser.version
        )

    # NOTE: Update with `ignore_nulls`/`fill_null` behavior once added to each `Function`
    # https://github.com/narwhals-dev/narwhals/pull/2719
    def _horizontal_function(
        self, fn: Callable[[Any, Any], Any], /
    ) -> Callable[[FunctionExpr[Any], ArrowDataFrame, str], ArrowExpr | ArrowScalar]:
        def func(
            node: FunctionExpr[Any], frame: ArrowDataFrame, name: str
        ) -> ArrowExpr | ArrowScalar:
            it = (self._expr.from_ir(e, frame, name).native for e in node.input)
            result = reduce(fn, it)
            if isinstance(result, pa.Scalar):
                return self._scalar.from_native(result, name, self.version)
            return self._expr.from_native(result, name, self.version)

        return func

    def any_horizontal(
        self, node: FunctionExpr[AnyHorizontal], frame: ArrowDataFrame, name: str
    ) -> ArrowExpr | ArrowScalar:
        return self._horizontal_function(pc.or_kleene)(node, frame, name)

    def all_horizontal(
        self, node: FunctionExpr[AllHorizontal], frame: ArrowDataFrame, name: str
    ) -> ArrowExpr | ArrowScalar:
        return self._horizontal_function(pc.and_kleene)(node, frame, name)

    def sum_horizontal(
        self, node: FunctionExpr[F.SumHorizontal], frame: ArrowDataFrame, name: str
    ) -> ArrowExpr | ArrowScalar:
        return self._horizontal_function(pc.add)(node, frame, name)

    def min_horizontal(
        self, node: FunctionExpr[F.MinHorizontal], frame: ArrowDataFrame, name: str
    ) -> ArrowExpr | ArrowScalar:
        return self._horizontal_function(pc.min_element_wise)(node, frame, name)

    def max_horizontal(
        self, node: FunctionExpr[F.MaxHorizontal], frame: ArrowDataFrame, name: str
    ) -> ArrowExpr | ArrowScalar:
        return self._horizontal_function(pc.max_element_wise)(node, frame, name)

    # TODO @dangotbanned: Impl `mean_horizontal`
    def mean_horizontal(
        self, node: FunctionExpr[F.MeanHorizontal], frame: ArrowDataFrame, name: str
    ) -> ArrowExpr | ArrowScalar:
        raise NotImplementedError

    # TODO @dangotbanned: Impl `concat_str`
    def concat_str(
        self, node: FunctionExpr[ConcatHorizontal], frame: ArrowDataFrame, name: str
    ) -> ArrowExpr | ArrowScalar:
        raise NotImplementedError

    def int_range(
        self, node: RangeExpr[IntRange], frame: ArrowDataFrame, name: str
    ) -> ArrowExpr:
        start_: PythonLiteral
        end_: PythonLiteral
        start, end = node.function.unwrap_input(node)
        step = node.function.step
        dtype = node.function.dtype
        if is_literal_scalar(start) and is_literal_scalar(end):
            start_, end_ = start.unwrap(), end.unwrap()
        else:
            scalar_start = self._expr.from_ir(start, frame, "start")
            scalar_end = self._expr.from_ir(end, frame, "end")
            if isinstance(scalar_start, self._scalar) and isinstance(
                scalar_end, self._scalar
            ):
                start_, end_ = scalar_start.to_python(), scalar_end.to_python()
            else:
                msg = (
                    f"All inputs for `int_range()` must be scalar or aggregations, but got \n"
                    f"{scalar_start.native!r}\n{scalar_end.native!r}"
                )
                raise InvalidOperationError(msg)
        if isinstance(start_, int) and isinstance(end_, int):
            import numpy as np  # ignore-banned-import

            from narwhals._plan.arrow.expr import chunked_array

            pa_dtype = narwhals_to_native_dtype(dtype, self.version)
            native = chunked_array(pa.array(np.arange(start_, end_, step), pa_dtype))
            return self._expr.from_native(native, name, self.version)

        else:
            msg = (
                f"All inputs for `int_range()` resolve to int, but got \n"
                f"{start_!r}\n{end_!r}"
            )
            raise InvalidOperationError(msg)
