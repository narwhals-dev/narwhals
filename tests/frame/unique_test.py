from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest

# We use nw instead of nw.stable.v1 to ensure that DuckDBPyRelation
# becomes LazyFrame instead of DataFrame
import narwhals as nw
from narwhals.exceptions import ColumnNotFoundError
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}


@pytest.mark.parametrize("subset", ["b", ["b"]])
@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("first", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("last", {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}),
        ("any", {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}),
        ("none", {"a": [2], "b": [6], "z": [9]}),
        ("foo", {"a": [2], "b": [6], "z": [9]}),
    ],
)
def test_unique(
    constructor: Constructor,
    subset: str | list[str] | None,
    keep: str,
    expected: dict[str, list[float]],
) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    if isinstance(df, nw.LazyFrame) and keep in {
        "first",
        "last",
    }:
        context: Any = pytest.raises(ValueError, match="row order")
    elif keep == "foo":
        context = pytest.raises(ValueError, match=": foo")
    else:
        context = does_not_raise()

    with context:
        result = df.unique(subset, keep=keep).sort("z")  # type: ignore[arg-type]
        assert_equal_data(result, expected)


def test_unique_invalid_subset(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    with pytest.raises(ColumnNotFoundError):
        df.lazy().unique(["fdssfad"]).collect()


@pytest.mark.filterwarnings("ignore:.*backwards-compatibility:UserWarning")
def test_unique_none(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    result = df.unique(maintain_order=False).sort("z")
    assert_equal_data(result, data)

    if isinstance(df, nw.LazyFrame):
        with pytest.raises(ValueError, match="not supported"):
            result = df.unique(maintain_order=True).sort("z")
    else:
        result = df.unique(maintain_order=True)
        assert_equal_data(result, data)
