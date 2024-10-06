import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager

data = {
    "a": [1.0, 2.0, None, 4.0],
    "b": [None, 3.0, None, 5.0],
}


def test_len(constructor_eager: ConstructorEager) -> None:
    result = len(nw.from_native(constructor_eager(data), eager_only=True))

    assert result == 4
