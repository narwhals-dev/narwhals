from typing import Any

import narwhals.stable.v1 as nw


def test_is_empty(constructor_series_with_pyarrow: Any) -> None:
    series = nw.from_native(constructor_series_with_pyarrow([1, 2, 3]), series_only=True)
    assert not series.is_empty()
    assert not series[:1].is_empty()
    assert len(series[:1]) == 1
    assert series[:0].is_empty()
