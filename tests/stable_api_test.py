import pytest

import narwhals
from narwhals import StableAPI


def test_invalid_version() -> None:
    with pytest.raises(ValueError, match="Expected one of"):
        StableAPI("0.19")  # type: ignore[arg-type]


def test_completeness() -> None:
    left = [i for i in narwhals.__dir__() if not i.startswith("_")]
    right = [i for i in narwhals.StableAPI("0.20").__dir__() if not i.startswith("_")]
    missing = [i for i in left if i not in right]
    assert missing == ["StableAPI"]
