from __future__ import annotations

from tests.utils import ConstructorEager, assert_equal_data

import narwhals as nw

data = [1, 2, 3]


def test_to_frame(constructor_eager: ConstructorEager) -> None:
    df = (
        nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
        .alias("")
        .to_frame()
    )
    assert_equal_data(df, {"": [1, 2, 3]})
