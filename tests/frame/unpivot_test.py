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


@pytest.mark.parametrize("variable_name", ["", "custom_var_name"])
@pytest.mark.parametrize("value_name", ["", "custom_value_name"])
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

    assert result.collect_schema().names()[-2:] == [variable_name, value_name]


def test_unpivot_default_var_value_names(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.unpivot(on=["b", "c"], index=["a"])

    assert result.collect_schema().names()[-2:] == ["variable", "value"]
