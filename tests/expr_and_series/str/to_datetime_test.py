from datetime import datetime
from typing import Any

import narwhals.stable.v1 as nw

data = {"a": ["2020-01-01T12:34:56"]}


def test_to_datetime(constructor: Any) -> None:
    result = (
        nw.from_native(constructor(data), eager_only=True)
        .select(b=nw.col("a").str.to_datetime(format="%Y-%m-%dT%H:%M:%S"))
        .item(row=0, column="b")
    )
    assert result == datetime(2020, 1, 1, 12, 34, 56)
