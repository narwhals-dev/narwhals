import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager


def test_shape(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager({"a": [1, 2]}), eager_only=True)["a"].shape
    expected = (2,)
    assert result == expected
