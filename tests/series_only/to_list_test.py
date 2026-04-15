from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.testing.typing import ConstructorEager

data = [1, 2, 3]


def test_to_list(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    assert_equal_data({"a": s.to_list()}, {"a": [1, 2, 3]})
