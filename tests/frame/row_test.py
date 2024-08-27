from typing import Any

import narwhals.stable.v1 as nw


def test_row_column(constructor_eager: Any) -> None:
    data = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [11, 12, 13, 14, 15, 16],
    }
    result = nw.from_native(constructor_eager(data), eager_only=True).row(2)
    if "pyarrow_table" in str(constructor_eager):
        result = tuple(x.as_py() for x in result)
    assert result == (3.0, 13)
