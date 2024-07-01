from typing import Any

import narwhals as nw

data = [1, 2, 3]


def test_to_list(constructor_series: Any) -> None:
    s = nw.from_native(constructor_series(data), series_only=True)
    assert s.to_list() == [1, 2, 3]
