from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = [1, 2, 3]


def test_to_frame(constructor: Any) -> None:
    df = (
        nw.from_native(constructor({"a": data}), eager_only=True)["a"]
        .alias("")
        .to_frame()
    )
    compare_dicts(df, {"": [1, 2, 3]})
