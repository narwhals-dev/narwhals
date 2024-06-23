import pytest

from narwhals import StableAPI


def test_invalid_version() -> None:
    with pytest.raises(ValueError, match="Expected one of"):
        StableAPI("0.19")  # type: ignore[arg-type]
