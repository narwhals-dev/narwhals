from typing import Any

import narwhals.stable.v1 as nw

data = [1, 2, 3]


def test_to_list(constructor_series_with_pyarrow: Any) -> None:
    s = nw.from_native(constructor_series_with_pyarrow(data), series_only=True)
    assert s.to_list() == [1, 2, 3]
