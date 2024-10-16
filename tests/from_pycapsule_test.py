import sys

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import compare_dicts


@pytest.mark.xfail(parse_version(pa.__version__) < (14,), reason="too old")
def test_from_pycapsule_to_arrow() -> None:
    df = nw.from_native(pl.DataFrame({"ab": [1, 2, 3], "ba": [4, 5, 6]}), eager_only=True)
    result = nw.from_pycapsule(df, native_namespace=pa)
    assert isinstance(result.to_native(), pa.Table)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    compare_dicts(result, expected)


@pytest.mark.xfail(parse_version(pa.__version__) < (14,), reason="too old")
def test_from_pycapsule_to_polars(monkeypatch: pytest.MonkeyPatch) -> None:
    tbl = pa.table({"ab": [1, 2, 3], "ba": [4, 5, 6]})
    monkeypatch.delitem(sys.modules, "pandas")
    df = nw.from_native(tbl, eager_only=True)
    result = nw.from_pycapsule(df, native_namespace=pl)
    assert isinstance(result.to_native(), pl.DataFrame)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    compare_dicts(result, expected)
    assert "pandas" not in sys.modules


@pytest.mark.xfail(parse_version(pa.__version__) < (14,), reason="too old")
def test_from_pycapsule_to_pandas() -> None:
    df = nw.from_native(pa.table({"ab": [1, 2, 3], "ba": [4, 5, 6]}), eager_only=True)
    result = nw.from_pycapsule(df, native_namespace=pd)
    assert isinstance(result.to_native(), pd.DataFrame)
    expected = {"ab": [1, 2, 3], "ba": [4, 5, 6]}
    compare_dicts(result, expected)


def test_from_pycapsule_invalid() -> None:
    with pytest.raises(TypeError, match="PyCapsule"):
        nw.from_pycapsule({"a": [1]}, native_namespace=pa)  # type: ignore[arg-type]
