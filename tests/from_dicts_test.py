from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import qualified_type_name

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
    result = nw.from_dicts(data, backend=eager_backend)
    expected = [{"c": 1, "d": 5}, {"c": 2, "d": 6}]
    assert result.rows(named=True) == expected
    assert isinstance(result, nw.DataFrame)


def test_from_dicts_schema(eager_backend: EagerAllowed) -> None:
    schema = {"c": nw.Int16(), "d": nw.Float32()}
    result = nw.from_dicts(
        [{"c": 1, "d": 5}, {"c": 2, "d": 6}], backend=eager_backend, schema=schema
    )
    assert result.collect_schema() == schema


def test_from_dicts_non_eager() -> None:
    pytest.importorskip("duckdb")
    with pytest.raises(ValueError, match="lazy-only"):
        nw.from_dicts([{"c": 1, "d": 5}, {"c": 2, "d": 6}], backend="duckdb")  # type: ignore[arg-type]


def test_from_dicts_empty(eager_backend: EagerAllowed) -> None:
    result = nw.from_dicts([], backend=eager_backend)
    assert result.shape == (0, 0)


def test_from_dicts_empty_with_schema(eager_backend: EagerAllowed) -> None:
    schema = nw.Schema({"a": nw.String(), "b": nw.Int8()})
    result = nw.from_dicts([], schema, backend=eager_backend)
    assert result.schema == schema
