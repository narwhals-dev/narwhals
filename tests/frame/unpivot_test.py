from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

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
    constructor: Constructor,
    on: str | list[str] | None,
    expected: dict[str, list[float]],
) -> None:
    df = nw.from_native(constructor(data))
    result = df.unpivot(on=on, index=["a"]).sort("variable", "a")
    compare_dicts(result, expected)


@pytest.mark.parametrize("variable_name", ["", "custom_var_name", None])
@pytest.mark.parametrize("value_name", ["", "custom_value_name", None])
def test_unpivot_var_value_names(
    constructor: Constructor,
    variable_name: str | None,
    value_name: str | None,
) -> None:
    if variable_name == "" and value_name == "":
        pytest.skip()

    df = nw.from_native(constructor(data))
    result = df.unpivot(
        on=["b", "c"], index=["a"], variable_name=variable_name, value_name=value_name
    )

    out_variable_name = variable_name if variable_name is not None else "variable"
    out_value_name = value_name if value_name is not None else "value"
    assert result.collect_schema().names()[-2:] == [out_variable_name, out_value_name]
