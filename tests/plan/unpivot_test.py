from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.plan.utils import assert_equal_data, dataframe
from tests.utils import PYARROW_VERSION

if TYPE_CHECKING:
    from narwhals.dtypes import DType


data = {"a": [7, 8, 9], "b": [1, 3, 5], "c": [2, 4, 6]}

expected_on_b_idx_a = {"a": [7, 8, 9], "variable": ["b", "b", "b"], "value": [1, 3, 5]}

expected_on_b_c_idx_a = {
    "a": [7, 8, 9, 7, 8, 9],
    "variable": ["b", "b", "b", "c", "c", "c"],
    "value": [1, 3, 5, 2, 4, 6],
}

expected_on_none_idx_a = {
    "a": [7, 8, 9, 7, 8, 9],
    "variable": ["b", "b", "b", "c", "c", "c"],
    "value": [1, 3, 5, 2, 4, 6],
}

expected_on_b_c_idx_none = {
    "variable": ["b", "b", "b", "c", "c", "c"],
    "value": [1, 3, 5, 2, 4, 6],
}

expected_on_none_idx_none = {
    "variable": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
    "value": [7, 8, 9, 1, 3, 5, 2, 4, 6],
}


@pytest.mark.parametrize(
    ("on", "index", "expected"),
    [
        ("b", ["a"], expected_on_b_idx_a),
        (["b", "c"], ["a"], expected_on_b_c_idx_a),
        (None, ["a"], expected_on_none_idx_a),
        (["b", "c"], None, expected_on_b_c_idx_none),
        (None, None, expected_on_none_idx_none),
    ],
)
def test_unpivot(
    on: str | list[str] | None, index: list[str] | None, expected: dict[str, list[float]]
) -> None:
    df = dataframe(data)
    sort_columns = ["variable"] if index is None else ["variable", "a"]
    result = df.unpivot(on=on, index=index).sort(by=sort_columns)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("variable_name", "value_name"),
    [
        ("", "custom_value_name"),
        ("custom_variable_name", ""),
        ("custom_variable_name", "custom_value_name"),
    ],
)
def test_unpivot_var_value_names(variable_name: str, value_name: str) -> None:
    df = dataframe(data)
    result = df.unpivot(
        on=["b", "c"], index=["a"], variable_name=variable_name, value_name=value_name
    )
    assert result.collect_schema().names()[-2:] == [variable_name, value_name]


def test_unpivot_default_var_value_names() -> None:
    df = dataframe(data)
    result = df.unpivot(on=["b", "c"], index=["a"])
    assert result.collect_schema().names()[-2:] == ["variable", "value"]


@pytest.mark.xfail(PYARROW_VERSION < (14, 0, 0), reason="pyarrow<14")
@pytest.mark.parametrize(
    ("data", "expected_dtypes"),
    [
        (
            {"idx": [0, 1], "a": [1, 2], "b": [1.5, 2.5]},
            [nw.Int64(), nw.String(), nw.Float64()],
        )
    ],
)
def test_unpivot_mixed_types(data: dict[str, Any], expected_dtypes: list[DType]) -> None:
    df = dataframe(data)
    result = df.unpivot(on=["a", "b"], index="idx")
    assert result.collect_schema().dtypes() == expected_dtypes
