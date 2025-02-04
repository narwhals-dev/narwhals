from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_is_unique(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)
    result = nw.concat([df, df.head(1)]).is_unique()
    expected = {"is_unique": [False, True, True, False]}
    assert_equal_data({"is_unique": result}, expected)
