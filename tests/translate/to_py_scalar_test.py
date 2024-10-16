import narwhals as nw
from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_cupy
from tests.utils import Constructor


def test_to_py_scalar(constructor_eager: Constructor) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    assert nw.to_py_scalar(df["a"].item(0)) == 1


def test_to_py_scalar_cudf_array() -> None:
    if cudf := get_cudf():
        cupy = get_cupy()
        df = nw.from_native(cudf.DataFrame({"a": [1, 2, 3]}))
        assert isinstance(nw.to_py_scalar(df["a"]), cupy.ndarray)
