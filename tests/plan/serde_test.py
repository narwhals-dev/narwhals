"""Adapted from upstream tests.

- https://github.com/pola-rs/polars/blob/1c11555550f8772dd4378b729069fd8c19e2d2dc/py-polars/tests/unit/expr/test_serde.py
- https://github.com//pola-rs/polars/blob/1c11555550f8772dd4378b729069fd8c19e2d2dc/py-polars/tests/unit/io/test_serde.py
"""

from __future__ import annotations

import io
import pickle

# ruff: noqa: S301
from io import BytesIO
from typing import TYPE_CHECKING, Literal, TypeAlias

import pytest
from pytest import FixtureRequest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from narwhals._typing_compat import assert_never
from narwhals.exceptions import ComputeError
from tests.plan.utils import assert_expr_ir_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from _typeshed import ReadableBuffer


class MockPickleRW:
    """Exposes the interface required to roundtrip `pickle.dump` -> `pickle.load`.

    Intersection of:

        SupportsRead[bytes] & SupportsNoArgReadline[bytes] & Writer[bytes]
    """

    def __init__(self, initial_bytes: ReadableBuffer = b"") -> None:
        self._dont_look: BytesIO = BytesIO(initial_bytes)

    def read(self, n: int) -> bytes:
        return self._dont_look.read(n)

    def readline(self) -> bytes:
        return self._dont_look.readline()

    def write(self, data: bytes) -> int:
        return self._dont_look.write(data)

    def getbuffer(self) -> memoryview:
        # Not required by either `dump` or `load`, but is an easy way to get back to the start
        return self._dont_look.getbuffer()


class MockPathLike:
    """Duplicated from `frame_{scan_read,sink_write}_test.py`.

    Trying to keep the diff self-containedl, will need to leave refactoring for now.
    """

    def __init__(self, path: Path) -> None:
        self._super_secret: Path = path

    def __fspath__(self) -> str:
        return self._super_secret.__fspath__()


FileSourceKind: TypeAlias = Literal["str", "Path", "PathLike"]
BufferSourceKind: TypeAlias = Literal["BytesIO", "PickleRW"]


@pytest.fixture(params=[BytesIO, MockPickleRW])
def buffer(request: FixtureRequest) -> BytesIO | MockPickleRW:
    param: type[BytesIO | MockPickleRW] = request.param
    return param()


@pytest.fixture(params=["str", "Path", "PathLike"])
def file(
    request: FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> MockPathLike | str | Path:
    param: FileSourceKind = request.param
    path = tmp_path_factory.mktemp(param) / "source.tmp"
    path.touch()
    match param:
        case "Path":
            return path
        case "PathLike":
            return MockPathLike(path)
        case "str":
            return path.as_posix()
        case _:
            assert_never(param)


@pytest.fixture
def expr_complex() -> nwp.Expr:
    return (
        nwp.when(a=5)
        .then(ncs.integer())
        .otherwise(nwp.max("b"))
        .name.suffix("_extra")
        .rank("dense", descending=True)
        * 99
    ).name.to_uppercase()


XFAIL_NOT_IMPL_JSON = pytest.mark.xfail(
    reason="`deserialize(format='json')` is not yet implemented",
    raises=(NotImplementedError),
)


def cases() -> pytest.MarkDecorator:
    """Split out to get unique instances for the 2x shared tests.

    Same cases are duplicated upstream
    - https://github.com/pola-rs/polars/blob/1c11555550f8772dd4378b729069fd8c19e2d2dc/py-polars/tests/unit/expr/test_serde.py#L9-L18
    - https://github.com/pola-rs/polars/blob/1c11555550f8772dd4378b729069fd8c19e2d2dc/py-polars/tests/unit/expr/test_serde.py#L25-L34
    """
    return pytest.mark.parametrize(
        "expr",
        [
            nwp.col("foo").sum().over("bar"),
            nwp.col("foo").sum().over("bar", order_by=ncs.string()),
            nwp.col("foo").replace_strict({3: 1}, return_dtype=nw.UInt32),
            nwp.col("foo").replace_strict(
                [5, 6, 7], [8, 9, 10], default=nwp.nth(-1).last()
            ),
            nwp.col("foo").rolling_var(window_size=4, ddof=2),
            nwp.col("foo").rolling_mean(window_size=2),
            nwp.col("foo").sort_by(
                "bar", descending=[False, True], nulls_last=[True, False]
            ),
            nwp.col("foo").ewm_mean(alpha=0.5, min_samples=2, ignore_nulls=True),
        ],
    )


# TODO @dangotbanned: Add `Expr.meta.__eq__`
# See https://github.com/narwhals-dev/narwhals/blob/c3f00c85945230c945ac2eb90e4b9049949a0313/src/narwhals/_plan/meta.py#L124-L141
def meta_eq(actual: nwp.Expr, expected: nwp.Expr) -> None:
    assert_expr_ir_equal(actual._ir, expected._ir)


@cases()
@pytest.mark.parametrize("buffer_read", [BytesIO, memoryview, bytes])
def test_expr_serde_roundtrip_binary(
    expr: nwp.Expr, buffer_read: Callable[[bytes], ReadableBuffer]
) -> None:
    buf = expr.meta.serialize()
    round_tripped = nwp.Expr.deserialize(buffer_read(buf), format="binary")
    meta_eq(round_tripped, expr)


@cases()
@XFAIL_NOT_IMPL_JSON
def test_expr_serde_roundtrip_json(expr: nwp.Expr) -> None:  # pragma: no cover
    expr = nwp.col("foo").sum().over("bar")
    json = expr.meta.serialize(format="json")
    round_tripped = nwp.Expr.deserialize(io.StringIO(json), format="json")  # type: ignore[arg-type]
    meta_eq(round_tripped, expr)


def test_expr_serde_file(expr_complex: nwp.Expr, file: MockPathLike | str | Path) -> None:
    out = expr_complex.meta.serialize(file)
    assert out is None
    round_tripped = nwp.Expr.deserialize(file)
    meta_eq(round_tripped, expr_complex)


def test_expr_serde_buffer(
    expr_complex: nwp.Expr, buffer: BytesIO | MockPickleRW
) -> None:
    out = expr_complex.meta.serialize(buffer)
    assert out is None
    rewind = type(buffer)(buffer.getbuffer())
    round_tripped = nwp.Expr.deserialize(rewind)
    meta_eq(round_tripped, expr_complex)


# TODO @dangotbanned: Cover more invalid `source` types
def test_expr_deserialize_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        nwp.Expr.deserialize("abcdef")


@XFAIL_NOT_IMPL_JSON
def test_expr_deserialize_invalid_json() -> None:
    with pytest.raises(
        ComputeError, match="could not deserialize input into an expression"
    ):
        nwp.Expr.deserialize(io.StringIO("abcdef"), format="json")  # type: ignore[arg-type]


@XFAIL_NOT_IMPL_JSON
def test_expression_json_13991() -> None:  # pragma: no cover
    expr = nwp.col("foo").cast(nw.Decimal(38, 10))
    json = expr.meta.serialize(format="json")

    round_tripped = nwp.Expr.deserialize(io.StringIO(json), format="json")  # type: ignore[arg-type]
    meta_eq(round_tripped, expr)


def test_serde_expression_5461() -> None:
    e = nwp.col("a").sqrt() / nwp.col("b").alias("c")
    roundtrip = pickle.loads(pickle.dumps(e))
    meta_eq(roundtrip, e)


def test_pickling_simple_expression() -> None:
    e = nwp.col("foo").sum()
    roundtrip = pickle.loads(pickle.dumps(e))
    meta_eq(roundtrip, e)


def test_pickling_as_struct_11100() -> None:
    e = nwp.struct("a")
    roundtrip = pickle.loads(pickle.dumps(e))
    meta_eq(roundtrip, e)


# NOTE: Handle in `AnonymousExpr`, maybe raise a useful error for `lambda`?
# - think that polars is rewriting them so they aren't raising
# - 0% interest in doing that here
# https://github.com//pola-rs/polars/blob/1c11555550f8772dd4378b729069fd8c19e2d2dc/py-polars/tests/unit/io/test_serde.py#L99-L120
@pytest.mark.xfail(
    reason="TODO @dangotbanned: Cover `Expr.map_batches`", raises=NotImplementedError
)
def test_pickle_udf_expression() -> None:
    raise NotImplementedError
