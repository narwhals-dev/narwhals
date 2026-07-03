from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
import narwhals.stable.v2 as nw_v2
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.interchange.conftest import MockDf


def test_main_reject(mockdf: MockDf) -> None:
    result = nw.from_native(mockdf, pass_through=True)
    assert result is mockdf
    with pytest.raises(TypeError):
        nw.from_native(mockdf)  # type: ignore[call-overload]


def test_v1_non_strict(mockdf: MockDf) -> None:
    result = nw_v1.from_native(mockdf, eager_only=True, strict=False)
    # mypy issue?
    assert result is mockdf  # type: ignore[comparison-overlap]


def test_v2_reject(mockdf: MockDf) -> None:
    with pytest.raises(TypeError, match="Unsupported dataframe type"):
        # Typing rejection **is** expected in v2, since IntoDataFrame excludes
        # DataFrameLike objects!
        nw_v2.from_native(mockdf)  # type: ignore[call-overload]
    assert nw_v2.from_native(mockdf, pass_through=True) is mockdf


@pytest.mark.filterwarnings("ignore:.*Interchange Protocol:DeprecationWarning")
@pytest.mark.skipif(PANDAS_VERSION < (2, 0), reason="requires interchange protocol")
def test_is_ordered_categorical() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    data = {"a": ["a", "b"]}
    native = pd.DataFrame(data, dtype=pd.CategoricalDtype(ordered=True))
    interchange = native.__dataframe__()
    df = nw_v1.from_native(interchange, eager_or_interchange_only=True)
    series = df["a"]
    assert nw_v1.is_ordered_categorical(series)
