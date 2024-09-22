import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import Constructor

data = {
    "a": [1],
    "b": [1],
    "c": [1],
    "d": [1],
    "e": [1],
    "f": [1],
    "g": [1],
    "h": [1],
    "i": [1],
    "j": [1],
    "k": ["1"],
    "l": [1],
    "m": [True],
    "n": [True],
    "o": ["a"],
    "p": [1],
}
schema = {
    "a": nw.Int64,
    "b": nw.Int32,
    "c": nw.Int16,
    "d": nw.Int8,
    "e": nw.UInt64,
    "f": nw.UInt32,
    "g": nw.UInt16,
    "h": nw.UInt8,
    "i": nw.Float64,
    "j": nw.Float32,
    "k": nw.String,
    "l": nw.Datetime,
    "m": nw.Boolean,
    "n": nw.Boolean,
    "o": nw.Categorical,
    "p": nw.Int64,
}


@pytest.mark.filterwarnings("ignore:casting period[M] values to int64:FutureWarning")
def test_cast(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "pyarrow_table_constructor" in str(constructor) and parse_version(
        pa.__version__
    ) <= (15,):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        # TODO(unassigned): in modin, we end up with `'<U0'` dtype
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data)).cast(schema)

    expected = {
        "a": nw.Int32,
        "b": nw.Int16,
        "c": nw.Int8,
        "d": nw.Int64,
        "e": nw.UInt32,
        "f": nw.UInt16,
        "g": nw.UInt8,
        "h": nw.UInt64,
        "i": nw.Float32,
        "j": nw.Float64,
        "k": nw.String,
        "l": nw.Datetime,
        "m": nw.Int8,
        "n": nw.Int8,
        "o": nw.String,
        "p": nw.Duration,
    }
    result = df.cast(expected)

    assert dict(result.collect_schema()) == expected


@pytest.mark.filterwarnings("ignore:casting period[M] values to int64:FutureWarning")
def test_cast_all_str(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "pyarrow_table_constructor" in str(constructor) and parse_version(
        pa.__version__
    ) <= (15,):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        # TODO(unassigned): in modin, we end up with `'<U0'` dtype
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data)).cast(schema)

    expected = {
        "a": nw.String,
        "b": nw.String,
        "c": nw.String,
        "d": nw.String,
        "e": nw.String,
        "f": nw.String,
        "g": nw.String,
        "h": nw.String,
        "i": nw.String,
        "j": nw.String,
        "k": nw.String,
        "l": nw.String,
        "m": nw.String,
        "n": nw.String,
        "o": nw.String,
        "p": nw.String,
    }
    result = df.cast(nw.String)

    assert dict(result.collect_schema()) == expected
