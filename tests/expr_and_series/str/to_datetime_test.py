import narwhals.stable.v1 as nw
from tests.utils import Constructor

data = {"a": ["2020-01-01T12:34:56"]}


def test_to_datetime(constructor: Constructor) -> None:
    result = (
        nw.from_native(constructor(data))
        .lazy()
        .select(b=nw.col("a").str.to_datetime(format="%Y-%m-%dT%H:%M:%S"))
        .collect()
        .item(row=0, column="b")
    )
    assert str(result) == "2020-01-01 12:34:56"
