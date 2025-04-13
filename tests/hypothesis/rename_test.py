from __future__ import annotations

import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

import narwhals as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

strategy = st.lists(
    st.text(
        alphabet=st.characters(
            codec="utf-8", exclude_characters=['"', ".", "'", "\x00", *string.whitespace]
        ),
        min_size=2,
        max_size=10,
    ),
    min_size=3,
    max_size=3,
    unique=True,
)


@given(strategy, strategy)
@pytest.mark.slow
def test_rename(  # pragma: no cover
    constructor: Constructor,
    from_names: list[str],
    to_names: list[str],
) -> None:
    mapping = dict(zip(from_names, to_names))
    values = [1, 2, 3]
    data = dict.fromkeys(from_names, values)
    expected = dict.fromkeys(to_names, values)

    df = nw.from_native(constructor(data))
    result = df.rename(mapping)

    assert_equal_data(result, expected)
