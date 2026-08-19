from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import qualified_type_name
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from narwhals._typing import EagerAllowed


@pytest.mark.parametrize(
    "into_mapping", [dict, types.MappingProxyType], ids=qualified_type_name
)
def test_from_dicts(
    eager_backend: EagerAllowed, into_mapping: Callable[..., Mapping[str, Any]]
) -> None:
    rows = {"c": 1, "d": 5}, {"c": 2, "d": 6}
    data = [into_mapping(row) for row in rows]
    result = nw.DataFrame.from_dicts(data, backend=eager_backend)
    expected = {"c": [1, 2], "d": [5, 6]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw.DataFrame)


def test_from_dicts_schema(eager_backend: EagerAllowed) -> None:
    schema = {"c": nw.Int16(), "d": nw.Float32()}
    result = nw.DataFrame.from_dicts(
        [{"c": 1, "d": 5}, {"c": 2, "d": 6}], backend=eager_backend, schema=schema
    )
    assert result.collect_schema() == schema


@pytest.mark.parametrize(
    ("data", "schema", "expected"),
    [
        (
            [{"b": 5, "a": 1}, {"b": 6, "a": 2}],
            {"a": nw.Int16(), "b": nw.Float32()},
            {"a": [1, 2], "b": [5.0, 6.0]},
        ),
        (
            [{"a": 1, "b": 5, "c": 100}, {"a": 2, "b": 6, "c": 200}],
            {"a": nw.Int16(), "b": nw.Float32()},
            {"a": [1, 2], "b": [5.0, 6.0]},
        ),
        (
            [{"a": 1}, {"a": 2}],
            {"a": nw.Int16(), "b": nw.Float32()},
            {"a": [1, 2], "b": [None, None]},
        ),
        ([{}, {}], {"a": nw.Int64()}, {"a": [None, None]}),
        (
            [{"a": 1, "b": True}, {"a": 2}],
            {"a": nw.Int64(), "b": nw.Boolean()},
            {"a": [1, 2], "b": [True, None]},
        ),
    ],
    ids=["reordered-keys", "extra-key", "missing-key", "empty-rows", "partial-rows"],
)
def test_from_dicts_schema_mismatched_keys(
    eager_backend: EagerAllowed,
    request: pytest.FixtureRequest,
    data: list[dict[str, Any]],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3837
    if "pandas" in str(eager_backend):
        if not data[0]:
            # `DataFrame.from_records` returns 0 rows for all-empty row dicts
            request.applymarker(pytest.mark.xfail)
        if nw.Boolean() in schema.values():
            # NaN -> bool astype coerces missing values to True
            request.applymarker(pytest.mark.xfail)
    result = nw.DataFrame.from_dicts(data, backend=eager_backend, schema=schema)
    assert result.columns == list(schema)
    assert result.collect_schema() == schema
    assert_equal_data(result, expected)


def test_from_dicts_non_eager() -> None:
    pytest.importorskip("duckdb")
    with pytest.raises(ValueError, match="lazy-only"):
        nw.DataFrame.from_dicts([{"c": 1, "d": 5}, {"c": 2, "d": 6}], backend="duckdb")  # type: ignore[arg-type]


def test_from_dicts_empty(eager_backend: EagerAllowed) -> None:
    result = nw.DataFrame.from_dicts([], backend=eager_backend)
    assert result.shape == (0, 0)


def test_from_dicts_empty_with_schema(eager_backend: EagerAllowed) -> None:
    schema = nw.Schema({"a": nw.String(), "b": nw.Int8()})
    result = nw.DataFrame.from_dicts([], schema, backend=eager_backend)
    assert result.schema == schema


@pytest.mark.parametrize("n_first_schema", [0, 1, 99, 100])
def test_from_dicts_inconsistent_keys(
    eager_implementation: EagerAllowed,
    request: pytest.FixtureRequest,
    n_first_schema: int,
) -> None:
    # pyarrow only checks 1 row
    if "pyarrow" in str(eager_implementation) and n_first_schema >= 1:
        request.applymarker(pytest.mark.xfail)
    # polars checks 100 rows
    if "polars" in str(eager_implementation) and n_first_schema >= 100:
        request.applymarker(pytest.mark.xfail)
    # no xfail for pandas as it always scans all rows

    incomplete = ({"a": i} for i in range(n_first_schema))
    complete = {"a": n_first_schema, "b": 0}
    data = (*incomplete, complete)

    result = nw.DataFrame.from_dicts(data, backend=eager_implementation)
    assert result.columns == ["a", "b"]
