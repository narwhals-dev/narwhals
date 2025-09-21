from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from narwhals._typing import EagerAllowed


def test_from_dicts(eager_backend: EagerAllowed) -> None:
    result = nw.from_dicts([{"c": 1, "d": 5}, {"c": 2, "d": 6}], backend=eager_backend)
    expected = [{"c": 1, "d": 5}, {"c": 2, "d": 6}]
    assert result.rows(named=True) == expected
    assert isinstance(result, nw.DataFrame)


def test_from_dicts_schema(eager_backend: EagerAllowed) -> None:
    schema = {"c": nw.Int16(), "d": nw.Float32()}
    result = nw.from_dicts(
        [{"c": 1, "d": 5}, {"c": 2, "d": 6}], backend=eager_backend, schema=schema
    )
    assert result.collect_schema() == schema
    with pytest.deprecated_call():
        result = nw.from_dicts(
            [{"c": 1, "d": 5}, {"c": 2, "d": 6}],
            native_namespace=eager_backend,  # type: ignore[arg-type]
            schema=schema,
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
