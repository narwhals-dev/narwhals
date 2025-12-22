from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.plan.utils import assert_equal_data, dataframe
from tests.utils import PYARROW_VERSION

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [7, 8, 9], "b": [1, 3, 5], "c": [2, 4, 6]}


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
    data: Data, on: str | list[str] | None, index: list[str] | None, expected: Data
) -> None:
    sort_columns = ["variable"] if index is None else ["variable", "a"]
    result = dataframe(data).unpivot(on, index=index).sort(sort_columns)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("variable_name", "value_name"),
    [
        ("", "custom_value_name"),
        ("custom_variable_name", ""),
        ("custom_variable_name", "custom_value_name"),
    ],
)
def test_unpivot_var_value_names(data: Data, variable_name: str, value_name: str) -> None:
    result = dataframe(data).unpivot(
        ["b", "c"], index=["a"], variable_name=variable_name, value_name=value_name
    )
    assert result.collect_schema().names()[-2:] == [variable_name, value_name]


def test_unpivot_default_var_value_names(data: Data) -> None:
    result = dataframe(data).unpivot(["b", "c"], index=["a"])
    assert result.collect_schema().names()[-2:] == ["variable", "value"]


@pytest.mark.xfail(PYARROW_VERSION < (14, 0, 0), reason="pyarrow<14")
def test_unpivot_mixed_types() -> None:
    df = dataframe({"idx": [0, 1], "a": [1, 2], "b": [1.5, 2.5]})
    result = df.unpivot(["a", "b"], index="idx")
    assert result.collect_schema().dtypes() == [nw.Int64(), nw.String(), nw.Float64()]
