from __future__ import annotations

from typing import Generic, Sequence

from narwhals._compliant.typing import NativeExprT_co


class WindowInputs(Generic[NativeExprT_co]):
    __slots__ = ("expr", "order_by", "partition_by")

    def __init__(
        self, expr: NativeExprT_co, partition_by: Sequence[str], order_by: Sequence[str]
    ) -> None:
        self.expr = expr
        self.partition_by = partition_by
        self.order_by = order_by


class UnorderableWindowInputs(Generic[NativeExprT_co]):
    __slots__ = ("expr", "partition_by")

    def __init__(self, expr: NativeExprT_co, partition_by: Sequence[str]) -> None:
        self.expr = expr
        self.partition_by = partition_by
