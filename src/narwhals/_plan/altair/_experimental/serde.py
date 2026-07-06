"""JSON serialization.

Will adapt this later for the json part of:

- https://github.com/narwhals-dev/narwhals/compare/expr-ir/docs/fluff-1...expr-ir/serde-2
- https://github.com/narwhals-dev/narwhals/blob/d7c4ccba57e00a35e05833e9d1af5edacd7f7a9e/docs/plan/related-issues.md#L56-L59
- [#2704](https://github.com/narwhals-dev/narwhals/issues/2704)
"""

from __future__ import annotations

import functools
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def serialize(
    obj: Any,
    /,
    *,
    deterministic: bool = False,
    default: Callable[[Any], Any] | None = None,
) -> bytes:
    """Serialize an object to bytes.

    Arguments:
        obj: An object composed of JSON compatible types.
        deterministic: Ensure the same input produces the same output bytes.
            Use this when serializing to generate a hash value.

            Does not guarantee anything beyond the description in [`msgspec.json.encode(order="deterministic")`]

        default: A hook for objects that can't otherwise be serialized.
            The caller is responsible for providing a hook that roundtrips,
            which `default=str` cannot.

    Important:
        Uses the fastest available json serializer, in priority of [`msgspec`] > [`orjson`] > [`json`].

    [`msgspec`]: https://github.com/msgspec/msgspec
    [`orjson`]: https://github.com/ijl/orjson
    [`json`]: https://docs.python.org/3/library/json.html
    [`msgspec.json.encode(order="deterministic")`]: https://msgspec.dev/api#msgspec.json.encode
    """
    return _build_serializer(deterministic=deterministic, default=default)(obj)


@functools.cache
def _build_serializer(
    *, deterministic: bool, default: Callable[[Any], Any] | None
) -> Callable[[Any], bytes]:
    """Return the fastest available json serializer."""
    # NOTE: `marimo` depends on `msgspec`, so take it if available
    if find_spec("msgspec"):
        from msgspec.json import Encoder

        return Encoder(
            order="deterministic" if deterministic else None, enc_hook=default
        ).encode

    # NOTE: `mypy`, `jupyter_client` optionally depend on `orjson`
    if find_spec("orjson"):
        import orjson  # type: ignore[import-not-found]

        orjson_encode: Callable[[Any], bytes] = functools.partial(
            orjson.dumps,
            default=default,
            option=orjson.OPT_SORT_KEYS if deterministic else None,
        )
        return orjson_encode
    from json import JSONEncoder

    _encode = JSONEncoder(
        sort_keys=deterministic, default=default, separators=(",", ":")
    ).encode

    def encode(obj: Any, /) -> bytes:
        return _encode(obj).encode()

    return encode
