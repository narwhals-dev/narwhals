from typing import Any

import numpy as np
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


@pytest.mark.parametrize("col_expr", [np.array([False, False, True]), nw.col("a"), "a"])
def test_allh(constructor_with_pyarrow: Any, col_expr: Any) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor_with_pyarrow(data), eager_only=True)
    result = df.select(all=nw.all_horizontal(col_expr, nw.col("b")))

    expected = {"all": [False, False, True]}
    compare_dicts(result, expected)
