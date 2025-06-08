from __future__ import annotations

from typing import Literal

import pytest

# We use nw instead of nw.stable.v1 to ensure that DuckDBPyRelation
# becomes LazyFrame instead of DataFrame
import narwhals as nw
from narwhals.exceptions import ColumnNotFoundError
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


def test_unique_invalid_subset(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    with pytest.raises(ColumnNotFoundError):
        df.lazy().unique(["fdssfad"]).collect()


@pytest.mark.parametrize("subset", ["b", ["b"]])
@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("any", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("none", {"a": [2], "b": [6], "z": [9]}),
    ],
)
def test_unique(
    constructor: Constructor,
    subset: str | list[str] | None,
    keep: Literal["any", "none"],
    expected: dict[str, list[float]],
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.unique(subset, keep=keep).sort("z")
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
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    result = df.unique().sort("z")
    assert_equal_data(result, data)

    if not isinstance(df, nw.LazyFrame):
        result = df.unique(maintain_order=True)
        assert_equal_data(result, data)
