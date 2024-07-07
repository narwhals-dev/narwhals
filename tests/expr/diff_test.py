from typing import Any

import pyarrow as pa
import pytest

import narwhals as nw
from narwhals.utils import parse_version
from tests.utils import compare_dicts

data = {
    "i": [0, 1, 2, 3, 4],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_diff(constructor_with_pyarrow: Any, request: Any) -> None:
    if "table" in str(constructor_with_pyarrow) and parse_version(pa.__version__) < (13,):
        # pc.pairwisediff is available since pyarrow 13.0.0
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_with_pyarrow(data), eager_only=True)
    result = df.with_columns(c_diff=nw.col("c").diff())[1:]
    expected = {
        "i": [1, 2, 3, 4],
        "b": [2, 3, 5, 3],
        "c": [4, 3, 2, 1],
        "c_diff": [-1, -1, -1, -1],
    }
    compare_dicts(result, expected)
    result = df.with_columns(c_diff=df["c"].diff())[1:]
    compare_dicts(result, expected)
