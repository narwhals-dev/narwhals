from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, ConstructorEager, assert_equal_data

data = [4, 4, 4, 1, 6, 6, 4, 4, 1, 1]


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("name", [None, "count_name"])
def test_value_counts(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    normalize: Any,
    name: str | None,
) -> None:
    if "pandas_nullable_constructor" in str(constructor_eager) and PANDAS_VERSION < (
        2,
        2,
    ):
        # bug in old pandas
        request.applymarker(pytest.mark.xfail)

    expected_count = [5, 3, 2]
    expected_index = [4, 1, 6]

    if normalize:
        expected_count = [v / len(data) for v in expected_count]  # type: ignore[misc]

    expected_name = name or ("proportion" if normalize else "count")
    expected = {"a": expected_index, expected_name: expected_count}

    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias(
        "a"
    )

    sorted_result = series.value_counts(sort=True, name=name, normalize=normalize)
    assert_equal_data(sorted_result, expected)

    unsorted_result = series.value_counts(
        sort=False, name=name, normalize=normalize
    ).sort(expected_name, descending=True)
    assert_equal_data(unsorted_result, expected)
