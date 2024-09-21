from typing import Any

import narwhals.stable.v1 as nw


def test_to_native(constructor_eager: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)

    assert isinstance(df.to_native(), df_raw.__class__)
