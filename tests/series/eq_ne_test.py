from typing import Any

import narwhals.stable.v1 as nw

data = [1, 2, 3]


def test_eq_ne(constructor_series: Any) -> None:
    s = nw.from_native(constructor_series(data), series_only=True)
    assert (s == 1).to_numpy().tolist() == [True, False, False]
    assert (s != 1).to_numpy().tolist() == [False, True, True]
