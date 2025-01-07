from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import PYARROW_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.stable.v1.dtypes import DType

data = {
    "a": ["x", "y", "z"],
    "b": [1, 3, 5],
    "c": [2, 4, 6],
}

expected_b_only = {
    "a": ["x", "y", "z"],
    "variable": ["b", "b", "b"],
    "value": [1, 3, 5],
}

expected_b_c = {
    "a": ["x", "y", "z", "x", "y", "z"],
    "variable": ["b", "b", "b", "c", "c", "c"],
    "value": [1, 3, 5, 2, 4, 6],
}


@pytest.mark.parametrize(
    ("on", "expected"),
    [("b", expected_b_only), (["b", "c"], expected_b_c), (None, expected_b_c)],
)
def test_unpivot_on(
    constructor: Constructor, on: str | list[str] | None, expected: dict[str, list[float]]
) -> None:
    df = nw.from_native(constructor(data))
    result = df.unpivot(on=on, index=["a"]).sort("variable", "a")
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("variable_name", "value_name"),
    [
        ("", "custom_value_name"),
        ("custom_variable_name", ""),
        ("custom_variable_name", "custom_value_name"),
    ],
)
def test_unpivot_var_value_names(
    constructor: Constructor,
    variable_name: str | None,
    value_name: str | None,
) -> None:
    df = nw.from_native(constructor(data))
    result = df.unpivot(
        on=["b", "c"], index=["a"], variable_name=variable_name, value_name=value_name
    )

    assert result.collect_schema().names()[-2:] == [variable_name, value_name]


def test_unpivot_default_var_value_names(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.unpivot(on=["b", "c"], index=["a"])

    assert result.collect_schema().names()[-2:] == ["variable", "value"]


@pytest.mark.parametrize(
    ("data", "expected_dtypes"),
    [
        (
            {"idx": [0, 1], "a": [1, 2], "b": [1.5, 2.5]},
            [nw.Int64(), nw.String(), nw.Float64()],
        ),
    ],
)
def test_unpivot_mixed_types(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: dict[str, Any],
    expected_dtypes: list[DType],
) -> None:
    if "cudf" in str(constructor) or (
        "pyarrow_table" in str(constructor) and PYARROW_VERSION < (14, 0, 0)
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.unpivot(on=["a", "b"], index="idx")

    assert result.collect_schema().dtypes() == expected_dtypes
