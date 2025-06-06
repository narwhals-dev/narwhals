from __future__ import annotations

from typing import Sequence


class WindowInputs:
    __slots__ = ("order_by", "partition_by")

    def __init__(self, partition_by: Sequence[str], order_by: Sequence[str]) -> None:
        self.partition_by = partition_by
        self.order_by = order_by
