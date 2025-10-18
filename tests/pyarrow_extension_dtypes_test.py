from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa

import narwhals as nw


def get_struct(data_type: pa.DataType) -> pa.StructType:
    return pa.struct([("num", data_type), ("den", data_type)])


class RationalType(pa.ExtensionType):
    def __init__(self, data_type: pa.DataType) -> None:
        super().__init__(get_struct(data_type=data_type), "rational_type")

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: Any, serialized: Any
    ) -> RationalType:
        return RationalType(storage_type[0].type)


class HashableRationalType(pa.ExtensionType):
    def __init__(self, data_type: pa.DataType) -> None:
        super().__init__(get_struct(data_type=data_type), "hashable_rational_type")

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: Any, serialized: Any
    ) -> HashableRationalType:
        return HashableRationalType(storage_type[0].type)

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.storage_type))


pa.register_extension_type(RationalType(pa.int64()))  # type: ignore[arg-type]
pa.register_extension_type(HashableRationalType(pa.int64()))  # type: ignore[arg-type]

rational_type = RationalType(pa.int32())
hash_rational_type = HashableRationalType(pa.int32())


def test_table_with_ext() -> None:
    array = pa.array([{"num": 10, "den": 17}, {"num": 20, "den": 13}])
    rational_array = rational_type.wrap_array(array.cast(rational_type.storage_type))
    hash_rational_array = rational_type.wrap_array(
        array.cast(hash_rational_type.storage_type)
    )

    table = pa.table({"rational": rational_array, "hashable": hash_rational_array})

    assert nw.from_native(table).schema == {
        "rational": nw.Unknown(),
        "hashable": nw.Unknown(),
    }


def test_schema_with_ext() -> None:
    pa_schema = pa.schema(
        [("a", pa.int16()), ("rational", rational_type), ("hashable", hash_rational_type)]
    )
    nw_schema = nw.Schema.from_arrow(pa_schema)

    assert nw_schema == nw.Schema(
        {"a": nw.Int16(), "rational": nw.Unknown(), "hashable": nw.Unknown()}
    )
