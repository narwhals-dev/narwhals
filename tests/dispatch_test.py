from __future__ import annotations

import string
from collections import deque
from typing import Any

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals._dispatch import just_dispatch
from narwhals.dtypes import DType, SignedIntegerType, UnsignedIntegerType


def test_just_dispatch() -> None:  # noqa: PLR0915
    i8 = nw.Int8()
    i64 = nw.Int64()
    u32 = nw.UInt32()
    u128 = nw.UInt128()
    bool_ = nw.Boolean()
    f64 = nw.Float64()
    arr_bool_4 = nw.Array(bool_, 4)

    def dtype_name(dtype: DType) -> str:
        return type(dtype).__name__

    # Default implementation serves as a fallback for subclasses of `upper_bound`
    @just_dispatch(upper_bound=DType)
    def dtype_repr_code(dtype: DType) -> str:
        return dtype_name(dtype).lower()

    assert dtype_repr_code(i64) == "int64"
    assert dtype_repr_code(u128) == "uint128"
    assert dtype_repr_code(bool_) == "boolean"

    # We can register specializations
    @dtype_repr_code.register(nw.Boolean, nw.String, nw.Categorical)
    def repr_remap(dtype: nw.Boolean | nw.String | nw.Categorical) -> str:
        return {nw.Boolean: "bool", nw.String: "str", nw.Categorical: "cat"}[type(dtype)]

    # And they immediately change the algorithm
    assert dtype_repr_code(bool_) == "bool"
    assert dtype_repr_code(nw.String()) == "str"
    assert dtype_repr_code(nw.Categorical()) == "cat"

    # But unhandled cases remain the same
    assert dtype_repr_code(i64) == "int64"
    assert dtype_repr_code(u128) == "uint128"

    @dtype_repr_code.register(*SignedIntegerType.__subclasses__())
    def repr_int(dtype: SignedIntegerType) -> str:
        return f"i{dtype_name(dtype).strip(string.ascii_letters)}"

    assert dtype_repr_code(i64) == "i64"
    assert dtype_repr_code(u128) == "uint128"
    assert dtype_repr_code(i8) == "i8"

    # The original function can be used directly
    assert repr_int(i64) == "i64"
    assert repr_int(i8) == "i8"

    # At runtime, we return the original function unchanged
    assert dtype_repr_code.dispatch(type(i64)) is repr_int
    assert dtype_repr_code.dispatch(type(i8)) is repr_int

    # Statically, we preserve the signature so warnings still appear
    assert repr_int(u128)  # type: ignore[arg-type]
    # Since that type would not dispatch to `repr_int`
    assert dtype_repr_code.dispatch(type(u128)) is dtype_repr_code.__wrapped__

    @dtype_repr_code.register(*UnsignedIntegerType.__subclasses__())
    def repr_uint(dtype: UnsignedIntegerType) -> str:
        return f"u{dtype_name(dtype).strip(string.ascii_letters)}"

    assert dtype_repr_code(u128) == "u128"
    assert dtype_repr_code(u32) == "u32"
    assert dtype_repr_code(f64) == "float64"

    @dtype_repr_code.register(nw.Float32, nw.Float64)
    def repr_float(dtype: nw.Float32 | nw.Float64) -> str:
        return "f32" if dtype.base_type() is nw.Float32 else "f64"

    assert dtype_repr_code(f64) == "f64"
    assert dtype_repr_code(nw.Float32()) == "f32"

    @dtype_repr_code.register(nw.Datetime)
    def repr_datetime(dtype: nw.Datetime) -> str:
        if dtype.time_zone is None:
            args: str = dtype.time_unit
        else:
            args = f"{dtype.time_unit}, {dtype.time_zone}"
        return f"datetime[{args}]"

    # Non-decorating registration is possible, if you really dislike names
    dtype_repr_code.register(nw.Duration)(lambda dtype: f"duration[{dtype.time_unit}]")

    @dtype_repr_code.register(nw.Duration)
    def repr_duration(dtype: nw.Duration) -> str:
        return f"duration[{dtype.time_unit}]"

    assert dtype_repr_code(nw.Datetime()) == "datetime[us]"
    assert dtype_repr_code(nw.Datetime("s")) == "datetime[s]"
    assert (
        dtype_repr_code(nw.Datetime(time_zone="Africa/Accra"))
        == "datetime[us, Africa/Accra]"
    )
    assert dtype_repr_code(nw.Duration()) == "duration[us]"
    assert dtype_repr_code(nw.Duration("ns")) == "duration[ns]"

    # Registration does not extend to subclasses
    assert dtype_repr_code(nw_v1.Datetime("ms")) == "datetime"
    assert dtype_repr_code(nw_v1.Duration("us")) == "duration"

    # And unrelated types will error out
    with pytest.raises(TypeError, match=r"dtype_repr_code.+not.+int.+upper bound.+DType"):
        dtype_repr_code(123)
    with pytest.raises(
        TypeError, match=r"dtype_repr_code.+not.+deque.+upper bound.+DType"
    ):
        dtype_repr_code(deque([1, 2, 3]))

    @dtype_repr_code.register(nw.Struct)
    def repr_struct(dtype: nw.Struct) -> str:
        return f"struct[{len(dtype.fields)}]"

    assert dtype_repr_code(nw.Struct({"a": i64})) == "struct[1]"
    assert dtype_repr_code(nw.Struct({"b": i64, "d": nw.Enum(["q", "d"])})) == "struct[2]"

    @dtype_repr_code.register(nw.List)
    def repr_list(dtype: nw.List) -> str:
        return f"list[{dtype_repr_code(dtype.inner())}]"

    # We can recurse through the dispatch from inside
    assert dtype_repr_code(nw.List(i64)) == "list[i64]"
    assert dtype_repr_code(nw.List(nw.Time)) == "list[time]"
    assert dtype_repr_code(nw.List(nw.Time())) == "list[time]"
    # And that extends to self-recursion
    assert dtype_repr_code(nw.List(nw.List(nw.List(f64)))) == "list[list[list[f64]]]"

    # Just one unhandled type left now!
    assert dtype_repr_code(nw.List(arr_bool_4)) == "list[array]"

    @dtype_repr_code.register(nw.Array)
    def repr_array(dtype: nw.Array) -> str:
        leaf: Any = dtype
        for _ in dtype.shape:
            leaf = leaf.inner
        leaf_dtype: DType = leaf()
        shape = dtype.shape
        return f"array[{dtype_repr_code(leaf_dtype)}, {(shape[0] if len(shape) == 1 else shape)!r}]"

    assert dtype_repr_code(arr_bool_4) == "array[bool, 4]"
    assert dtype_repr_code(nw.Array(u32, shape=(5, 10))) == "array[u32, (5, 10)]"
    assert dtype_repr_code(nw.List(arr_bool_4)) == "list[array[bool, 4]]"

    # Using `register` *without parenthesis* will warn statically
    @dtype_repr_code.register  # type: ignore[arg-type]
    def no_register(dtype: nw.Int16) -> str:
        return str(dtype)

    # Because at runtime, it was never added to the registry
    assert repr_array in dtype_repr_code.registry.values()
    assert no_register not in dtype_repr_code.registry.values()

    # Using `register` *without arguments* will raise at runtime too
    with pytest.raises(TypeError):

        @dtype_repr_code.register()  # type: ignore[call-arg]
        def no_types(dtype: DType) -> str:
            return str(dtype)

    assert repr(dtype_repr_code) == "JustDispatch<dtype_repr_code>"
