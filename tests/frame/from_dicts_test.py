from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._typing import EagerAllowed


def test_from_dicts(eager_backend: EagerAllowed) -> None:
    result = nw.DataFrame.from_dicts(
        [{"c": 1, "d": 5}, {"c": 2, "d": 6}], backend=eager_backend
    )
    expected = [{"c": 1, "d": 5}, {"c": 2, "d": 6}]
    assert result.rows(named=True) == expected
    assert isinstance(result, nw.DataFrame)


def test_from_dicts_schema(eager_backend: EagerAllowed) -> None:
    schema = {"c": nw.Int16(), "d": nw.Float32()}
    result = nw.DataFrame.from_dicts(
        [{"c": 1, "d": 5}, {"c": 2, "d": 6}], backend=eager_backend, schema=schema
    )
    assert result.collect_schema() == schema


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


class NonDictMapping(Mapping[str, Any]):
    """A mapping which is not a dictionary."""

    def __init__(self, *keys: str) -> None:
        self._keys = set(keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        yield from self._keys

    def __getitem__(self, key: str) -> Any:
        return len(key)


def test_from_dicts_other_mapping(eager_backend: EagerAllowed) -> None:
    # test the function works with non-dict mappings
    data = [NonDictMapping("c", "d", "hello"), NonDictMapping("c", "d", "hello")]
    expected = [{"c": 1, "d": 1, "hello": 5}, {"c": 1, "d": 1, "hello": 5}]

    result = nw.from_dicts(data, backend=eager_backend)
    assert result.rows(named=True) == expected
