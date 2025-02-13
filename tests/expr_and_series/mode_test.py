from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": [1, 1, 2, 2, 3],
    "b": [1, 2, 3, 3, 4],
}


def test_mode_single_expr(constructor_eager: ConstructorEager) -> None:
    if "pyarrow" in str(constructor_eager):
        # TODO(unassigned): reimplement mode for pyarrow
        return
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").mode()).sort("a")
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_mode_series(constructor_eager: ConstructorEager) -> None:
    if "pyarrow" in str(constructor_eager):
        # TODO(unassigned): reimplement mode for pyarrow
        return
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.mode().sort()
    expected = {"a": [1, 2]}
    assert_equal_data({"a": result}, expected)
