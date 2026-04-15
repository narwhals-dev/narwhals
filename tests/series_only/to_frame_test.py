from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.testing.typing import ConstructorEager

data = [1, 2, 3]


def test_to_frame(constructor_eager: ConstructorEager) -> None:
    df = (
        nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
        .alias("")
        .to_frame()
    )
    assert_equal_data(df, {"": [1, 2, 3]})
