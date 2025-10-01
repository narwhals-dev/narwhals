from __future__ import annotations

import random
import re
import string

# ruff: noqa: S311
from collections import deque
from itertools import islice, product, repeat
from typing import TYPE_CHECKING, NamedTuple

import hypothesis.strategies as st
import pytest
from hypothesis import given

import narwhals as nw
from narwhals._plan.common import temp
from narwhals._utils import qualified_type_name
from narwhals.exceptions import NarwhalsError

pytest.importorskip("pyarrow")
pytest.importorskip("polars")


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from narwhals._utils import _StoresColumns


class MockStoresColumns(NamedTuple):
    columns: Sequence[str]


_COLUMNS = ("abc", "XYZ", "nw2929023", "column", string.hexdigits)
_EMPTY_SCHEMA = nw.Schema((name, nw.Int64()) for name in _COLUMNS)


sources = pytest.mark.parametrize(
    "source",
    [
        _COLUMNS,
        MockStoresColumns(columns=_COLUMNS),
        deque(_COLUMNS),
        nw.from_dict({}, _EMPTY_SCHEMA, backend="pyarrow"),
        dict.fromkeys(_COLUMNS),
        set(_COLUMNS),
        nw.from_dict({}, _EMPTY_SCHEMA, backend="polars").to_native(),
    ],
    ids=qualified_type_name,
)


@sources
def test_temp_column_name_sources(source: _StoresColumns | Iterable[str]) -> None:
    name = temp.column_name(source)
    assert name not in _COLUMNS


@sources
def test_temp_column_names_sources(source: _StoresColumns | Iterable[str]) -> None:
    it = temp.column_names(source)
    name = next(it)
    assert name not in _COLUMNS


@given(n_chars=st.integers(6, 106))
@pytest.mark.slow
def test_temp_column_name_n_chars(n_chars: int) -> None:
    name = temp.column_name(_COLUMNS, n_chars=n_chars)
    assert name not in _COLUMNS


@given(n_new_names=st.integers(10_000, 100_000))
@pytest.mark.slow
def test_temp_column_names_always_new_names(n_new_names: int) -> None:
    it = temp.column_names(_COLUMNS)
    new_names = set(islice(it, n_new_names))
    assert len(new_names) == n_new_names
    assert new_names.isdisjoint(_COLUMNS)


@pytest.mark.parametrize(
    ("prefix", "n_chars"),
    [
        ("nw", random.randint(0, 5)),
        ("col", random.randint(0, 4)),
        ("NW_", random.randint(0, 3)),
        ("join", random.randint(0, 2)),
        ("__tmp", random.randint(0, 1)),
        ("longer", random.randint(-5, 0)),
        ("", random.randint(0, 5)),
    ],
)
def test_temp_column_name_requires_more_characters(prefix: str, n_chars: int) -> None:
    pattern = re.compile(
        rf"temp.+column.+name.+requires.+try.+shorter.+{prefix}.+higher.+{n_chars}",
        re.IGNORECASE | re.DOTALL,
    )
    with pytest.raises(NarwhalsError, match=pattern):
        temp.column_name(_COLUMNS, prefix=prefix, n_chars=n_chars)


def test_temp_column_name_failed_unique() -> None:
    hex_lower = string.hexdigits.strip(string.ascii_uppercase)
    every_possible_name_65k = [
        f"nw{e1}{e2}{e3}{e4}" for e1, e2, e3, e4 in product(*repeat(hex_lower, 4))
    ]
    n_many_columns = len(every_possible_name_65k)

    pattern = re.compile(
        rf"unable.+generate.+name.+n_chars=6.+within.+existing.+{n_many_columns}.+columns",
        re.DOTALL,
    )
    with pytest.raises(NarwhalsError, match=pattern):
        temp.column_name(every_possible_name_65k, prefix="nw", n_chars=6)


def test_temp_column_names_failed_unique() -> None:
    it = temp.column_names(["a", "b", "c"], prefix="long_prefix", n_chars=16)
    pattern = re.compile(
        r"unable.+generate.+name.+n_chars=16.+within.+existing.+.+columns.+\.\.\.",
        re.DOTALL,
    )
    with pytest.raises(NarwhalsError, match=pattern):
        list(islice(it, 100_000))
