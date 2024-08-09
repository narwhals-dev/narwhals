from typing import Any

import narwhals.stable.v1 as nw


def test_lazy(constructor_eager: Any) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)
