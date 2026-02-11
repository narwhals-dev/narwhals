from __future__ import annotations

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa

import narwhals as nw


class CustomInt16(pa.ExtensionType):
    def __init__(self) -> None:
        super().__init__(pa.int16(), "custom_int_16")

    def __arrow_ext_serialize__(self) -> bytes:
        return b""


class CustomInt32(pa.ExtensionType):
    def __init__(self) -> None:
        super().__init__(pa.int32(), "custom_int_32")

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    def __hash__(self) -> int:  # pragma: no cover
        return hash((self.__class__.__name__, self.storage_type))


pa.register_extension_type(CustomInt16())  # type: ignore[arg-type]
pa.register_extension_type(CustomInt32())  # type: ignore[arg-type]

custom_16 = CustomInt16()
custom_32 = CustomInt32()


def test_table_with_ext() -> None:
    array = pa.array([1, 2])
    int16_array = custom_16.wrap_array(array.cast(custom_16.storage_type))
    int32_array = custom_32.wrap_array(array.cast(custom_32.storage_type))

    table = pa.table({"non-hash-int16": int16_array, "hash-int-32": int32_array})

    assert nw.from_native(table).schema == {
        "non-hash-int16": nw.Unknown(),
        "hash-int-32": nw.Unknown(),
    }


def test_schema_with_ext() -> None:
    pa_schema = pa.schema(
        [("a", pa.int16()), ("non-hash-int16", custom_16), ("hash-int-32", custom_32)]
    )
    nw_schema = nw.Schema.from_arrow(pa_schema)
    assert nw_schema == nw.Schema(
        {"a": nw.Int16(), "non-hash-int16": nw.Unknown(), "hash-int-32": nw.Unknown()}
    )
