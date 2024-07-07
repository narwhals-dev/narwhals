from typing import Any

import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import compare_dicts


def test_any_all(constructor_with_pyarrow: Any, request: Any) -> None:
    if "table" in str(constructor_with_pyarrow) and parse_version(
        pa.__version__
    ) < parse_version("12.0.0"):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(
        constructor_with_pyarrow(
            {
                "a": [True, False, True],
                "b": [True, True, True],
                "c": [False, False, False],
            }
        )
    )
    result = df.select(nw.all().all())
    expected = {"a": [False], "b": [True], "c": [False]}
    compare_dicts(result, expected)
    result = df.select(nw.all().any())
    expected = {"a": [True], "b": [True], "c": [False]}
    compare_dicts(result, expected)
