from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
data_na = {"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]}


@pytest.mark.parametrize(
    ("named", "expected"),
    [
        (False, [(1, 4, 7.0, 5), (3, 4, 8.0, 6), (2, 6, 9.0, 7)]),
        (
            True,
            [
                {"a": 1, "_b": 4, "z": 7.0, "1": 5},
                {"a": 3, "_b": 4, "z": 8.0, "1": 6},
                {"a": 2, "_b": 6, "z": 9.0, "1": 7},
            ],
        ),
    ],
)
def test_iter_rows(
    request: Any,
    constructor_eager: ConstructorEager,
    named: bool,  # noqa: FBT001
    expected: list[tuple[Any, ...]] | list[dict[str, Any]],
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "_b": [4, 4, 6], "z": [7.0, 8.0, 9.0], "1": [5, 6, 7]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = list(df.iter_rows(named=named))
    assert result == expected


@pytest.mark.filterwarnings(
    "ignore:.*all arguments of to_dict except for the argument:FutureWarning"
)
@pytest.mark.parametrize(
    ("named", "expected"),
    [
        (False, [(1, 4, 7.0), (3, 4, 8.0), (2, 6, 9.0)]),
        (
            True,
            [
                {"a": 1, "b": 4, "z": 7.0},
                {"a": 3, "b": 4, "z": 8.0},
                {"a": 2, "b": 6, "z": 9.0},
            ],
        ),
    ],
)
def test_rows(
    constructor_eager: ConstructorEager,
    named: bool,  # noqa: FBT001
    expected: list[tuple[Any, ...]] | list[dict[str, Any]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.rows(named=named)
    assert result == expected


@pytest.mark.filterwarnings(
    r"ignore:.*Starting with pandas version 3\.0 all arguments of to_dict"
)
@pytest.mark.parametrize(
    ("named", "expected"),
    [
        (False, [(1, 4, 7.0, 5), (3, 4, 8.0, 6), (2, 6, 9.0, 7)]),
        (
            True,
            [
                {"a": 1, "_b": 4, "z": 7.0, "1": 5},
                {"a": 3, "_b": 4, "z": 8.0, "1": 6},
                {"a": 2, "_b": 6, "z": 9.0, "1": 7},
            ],
        ),
    ],
)
def test_rows_eager(
    constructor_eager: Any,
    named: bool,  # noqa: FBT001
    expected: list[tuple[Any, ...]] | list[dict[str, Any]],
) -> None:
    # posit-dev/py-shiny relies on `.rows(named=False)` to return unnamed rows
    data = {"a": [1, 3, 2], "_b": [4, 4, 6], "z": [7.0, 8.0, 9.0], "1": [5, 6, 7]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.rows(named=named)
    assert result == expected


def test_rows_with_nulls_unnamed(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        # cudf intentionally doesn't support itertuples / iter_rows
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data_na), eager_only=True)
    result = list(df.iter_rows(named=False))
    expected = [(None, 4, 7.0), (3, 4, None), (2, 6, 9.0)]
    for i, row in enumerate(expected):
        for j, value in enumerate(row):
            value_in_result = result[i][j]
            if value is None:
                assert pd.isna(value_in_result)  # because float('nan') != float('nan')
            else:
                assert value_in_result == value


def test_rows_with_nulls_named(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        # cudf intentionally doesn't support itertuples / iter_rows
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data_na), eager_only=True)
    result = list(df.iter_rows(named=True))
    expected: list[dict[str, Any]] = [
        {"a": None, "b": 4, "z": 7.0},
        {"a": 3, "b": 4, "z": None},
        {"a": 2, "b": 6, "z": 9.0},
    ]
    for i, row in enumerate(expected):
        for col, value in row.items():
            value_in_result = result[i][col]
            if value is None:
                assert pd.isna(value_in_result)  # because float('nan') != float('nan')
            else:
                assert value_in_result == value
