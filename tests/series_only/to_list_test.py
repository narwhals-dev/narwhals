from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = [1, 2, 3]


def test_to_list(nw_eager_constructor: ConstructorEager) -> None:
    s = nw.from_native(nw_eager_constructor({"a": data}), eager_only=True)["a"]
    assert_equal_data({"a": s.to_list()}, {"a": [1, 2, 3]})
