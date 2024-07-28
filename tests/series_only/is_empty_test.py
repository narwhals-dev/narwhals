from typing import Any

import narwhals.stable.v1 as nw


def test_is_empty(constructor: Any) -> None:
    series = nw.from_native(constructor({"a": [1, 2, 3]}), eager_only=True)["a"]
    assert not series.is_empty()
    assert not series[:1].is_empty()
    assert len(series[:1]) == 1
    assert series[:0].is_empty()
