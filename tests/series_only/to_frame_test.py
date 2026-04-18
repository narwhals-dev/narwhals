from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = [1, 2, 3]


def test_to_frame(nw_eager_constructor: ConstructorEager) -> None:
    df = (
        nw.from_native(nw_eager_constructor({"a": data}), eager_only=True)["a"]
        .alias("")
        .to_frame()
    )
    assert_equal_data(df, {"": [1, 2, 3]})
