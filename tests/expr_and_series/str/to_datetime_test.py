import narwhals.stable.v1 as nw
from tests.utils import Constructor

data = {"a": ["2020-01-01T12:34:56"]}


def test_to_datetime(constructor: Constructor) -> None:
    if "cudf" in str(constructor):
        expected = "2020-01-01T12:34:56.000000000"
    else:
        expected = "2020-01-01 12:34:56"

    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_datetime(format="%Y-%m-%dT%H:%M:%S"))
        .collect()
        .item(row=0, column="b")
    )
    assert str(result) == expected
