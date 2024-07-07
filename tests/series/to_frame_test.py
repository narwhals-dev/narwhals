from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = [1, 2, 3]


def test_eq_ne(constructor_series: Any) -> None:
    df = nw.from_native(constructor_series(data), series_only=True).alias("").to_frame()
    compare_dicts(df, {"": [1, 2, 3]})
