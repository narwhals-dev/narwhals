from __future__ import annotations

from typing import Literal

import pytest

import narwhals as nw
from narwhals.exceptions import ColumnNotFoundError, InvalidOperationError
from tests.utils import DUCKDB_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


@pytest.mark.parametrize("subset", ["b", ["b"]])
@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("last", {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}),
    ],
)
def test_unique_eager(
    constructor_eager: ConstructorEager,
    subset: str | list[str] | None,
    keep: Literal["first", "last"],
    expected: dict[str, list[float]],
) -> None:
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw)
    result = df.unique(subset, keep=keep).sort("z")
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", {"i": [None, 2], "a": [2, 1], "b": [4, 6]}),
        ("last", {"i": [1, 2], "a": [3, 1], "b": [4, 6]}),
    ],
)
def test_unique_first_last(
    constructor: Constructor,
    keep: Literal["first", "last"],
    expected: dict[str, list[float]],
    request: pytest.FixtureRequest,
) -> None:
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/12073
        request.applymarker(pytest.mark.xfail)
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {"i": [0, 1, None, 2], "a": [1, 3, 2, 1], "b": [4, 4, 4, 6]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.unique("b", keep=keep, order_by="i").sort("i")
    assert_equal_data(result, expected)

    if isinstance(df, nw.DataFrame):
        result = df.unique("b", keep=keep, order_by="i", maintain_order=True)
        assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", {"i": [0, 1, 2], "b": [4, 4, 6]}),
        ("last", {"i": [0, 1, 2], "b": [4, 4, 6]}),
    ],
)
def test_unique_first_last_no_subset(
    constructor: Constructor,
    keep: Literal["first", "last"],
    expected: dict[str, list[float]],
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {"i": [0, 1, 1, 2], "b": [4, 4, 4, 6]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.unique(keep=keep, order_by="i").sort("i")
    assert_equal_data(result, expected)

    if isinstance(df, nw.DataFrame):
        result = df.unique(keep=keep, order_by="i", maintain_order=True)
        assert_equal_data(result, expected)


def test_unique_invalid(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    with pytest.raises(ColumnNotFoundError):
        df.lazy().unique(["fdssfad"]).collect()
    with pytest.raises(InvalidOperationError):
        df.lazy().unique(keep="first").collect()


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("any", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("none", {"a": [2], "b": [6], "z": [9]}),
    ],
)
def test_unique(
    constructor: Constructor,
    keep: Literal["any", "none"],
    expected: dict[str, list[float]],
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.unique(["b"], keep=keep).sort("z")
    assert_equal_data(result, expected)


@pytest.mark.parametrize("subset", [None, ["a", "b"]])
@pytest.mark.parametrize(
    ("keep", "expected"),
    [("any", {"a": [1, 1, 2], "b": [3, 4, 4]}), ("none", {"a": [1, 2], "b": [4, 4]})],
)
def test_unique_full_subset(
    constructor: Constructor,
    subset: list[str] | None,
    keep: Literal["any", "none"],
    expected: dict[str, list[float]],
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {"a": [1, 1, 1, 2], "b": [3, 3, 4, 4]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.unique(subset, keep=keep).sort("a", "b")
    assert_equal_data(result, expected)


def test_unique_invalid_keep(constructor: Constructor) -> None:
    with pytest.raises(ValueError, match=r"(Got|got): cabbage"):
        nw.from_native(constructor(data)).unique(keep="cabbage")  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore:.*backwards-compatibility:UserWarning")
def test_unique_none(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    result = df.unique().sort("z")
    assert_equal_data(result, data)

    if not isinstance(df, nw.LazyFrame):
        result = df.unique(maintain_order=True)
        assert_equal_data(result, data)


def test_unique_3069(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {"name": ["a", "b", "c"], "group": ["d", "e", "f"], "value": [1, 2, 3]}
    df = nw.from_native(constructor(data))
    unique_to_get = "group"
    result = df.select(nw.col(unique_to_get)).unique().sort(unique_to_get)
    expected = {"group": ["d", "e", "f"]}
    assert_equal_data(result, expected)
