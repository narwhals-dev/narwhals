from __future__ import annotations

from typing import TYPE_CHECKING, overload

from narwhals._plan.literal import is_literal_scalar
from narwhals._plan.protocols import EagerNamespace
from narwhals._utils import Version

if TYPE_CHECKING:
    from typing_extensions import Self

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
    from narwhals.typing import NonNestedLiteral


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

    def any_horizontal(
        self, node: FunctionExpr[AnyHorizontal], frame: ArrowDataFrame, name: str
    ) -> Self:
        raise NotImplementedError

    def all_horizontal(
        self, node: FunctionExpr[AllHorizontal], frame: ArrowDataFrame, name: str
    ) -> Self:
        raise NotImplementedError

    def sum_horizontal(
        self, node: FunctionExpr[F.SumHorizontal], frame: ArrowDataFrame, name: str
    ) -> Self:
        raise NotImplementedError

    def min_horizontal(
        self, node: FunctionExpr[F.MinHorizontal], frame: ArrowDataFrame, name: str
    ) -> Self:
        raise NotImplementedError

    def max_horizontal(
        self, node: FunctionExpr[F.MaxHorizontal], frame: ArrowDataFrame, name: str
    ) -> Self:
        raise NotImplementedError

    def mean_horizontal(
        self, node: FunctionExpr[F.MeanHorizontal], frame: ArrowDataFrame, name: str
    ) -> Self:
        raise NotImplementedError

    def concat_str(
        self, node: FunctionExpr[ConcatHorizontal], frame: ArrowDataFrame, name: str
    ) -> Self:
        raise NotImplementedError

    def int_range(
        self, node: RangeExpr[IntRange], frame: ArrowDataFrame, name: str
    ) -> Self:
        raise NotImplementedError
