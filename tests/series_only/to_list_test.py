from typing import Any

import narwhals.stable.v1 as nw

data = [1, 2, 3]


def test_to_list(constructor_eager: Any) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    assert s.to_list() == [1, 2, 3]
