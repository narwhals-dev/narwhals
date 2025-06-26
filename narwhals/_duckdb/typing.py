from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import Expression


class WindowExpressionKwargs(TypedDict, total=False):
    partition_by: Sequence[str | Expression]
    order_by: Sequence[str | Expression]
    rows_start: str
    rows_end: str
    descending: bool
    nulls_last: bool
    ignore_nulls: bool
