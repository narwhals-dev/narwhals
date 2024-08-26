from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_null(constructor: Any) -> None:
    data_na = {"a": [None, 3, 2], "z": [7.0, None, None]}
    expected = {"a": [True, False, False], "z": [True, False, False]}
    df = nw.from_native(constructor(data_na))
    result = df.select(nw.col("a").is_null(), ~nw.col("z").is_null())

    compare_dicts(result, expected)


def test_null_series(constructor_eager: Any) -> None:
    data_na = {"a": [None, 3, 2], "z": [7.0, None, None]}
    expected = {"a": [True, False, False], "z": [True, False, False]}
    df = nw.from_native(constructor_eager(data_na), eager_only=True)
    result = {"a": df["a"].is_null(), "z": ~df["z"].is_null()}

    compare_dicts(result, expected)
