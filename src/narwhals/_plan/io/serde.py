from __future__ import annotations

import os

# ruff: noqa: S301
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from narwhals._plan import common

if TYPE_CHECKING:
    from io import StringIO

    from _typeshed import SupportsNoArgReadline, SupportsRead
    from typing_extensions import TypeIs

    from narwhals._plan.typing import SerdeFormat, SerdeSink, SerdeSource
    from narwhals.typing import FileSource

    class _PickleLoad(SupportsRead[bytes], SupportsNoArgReadline[bytes], Protocol):
        """https://github.com/python/typeshed/blob/abbf4372552d78cdd4514db2a2d855658c1a98a5/stdlib/_pickle.pyi#L7-L10."""


_Self = TypeVar("_Self")


def _is_file_source(source: Any) -> TypeIs[FileSource]:
    return isinstance(source, (str, Path, os.PathLike))


def _can_pickle_load(source: Any) -> TypeIs[_PickleLoad]:
    return common.hasattrs_static(source, "read", "readline")


def deserialize(
    cls: type[_Self], source: SerdeSource, format: SerdeFormat = "binary"
) -> _Self:
    if format == "json":
        return _deserialize_json(cls, source)
    # NOTE: `pickle` is implemented as a C-extension.
    # - `load` expects a specific interface https://github.com/python/cpython/blob/90748760d38ca3ac5fc6788a69becab905c95598/Modules/_pickle.c#L1772-L1775
    # - This dance helps us spend as little time in python as possible
    if _is_file_source(source):
        with open(source, "rb") as fd:  # noqa: PTH123
            obj = pickle.load(fd)
    elif _can_pickle_load(source):
        obj = pickle.load(source)
    else:
        # https://github.com/python/cpython/blob/90748760d38ca3ac5fc6788a69becab905c95598/Modules/_pickle.c#L1239-L1253
        obj = pickle.loads(source)
    return _ensure_owner(obj, cls)


def serialize_binary(obj: Any, file: SerdeSink | None, /) -> bytes | None:
    if file is None:
        return pickle.dumps(obj)
    if _is_file_source(file):
        with open(file, "wb") as fd:  # noqa: PTH123
            pickle.dump(obj, fd)
    else:
        pickle.dump(obj, file)
    return None


def serialize_json(obj: Any, file: SerdeSink | StringIO | None, /) -> str | None:
    msg = "`serialize(format='json')` is not yet implemented"
    raise NotImplementedError(msg)


def _deserialize_json(cls: type[_Self], source: SerdeSource | StringIO) -> _Self:
    msg = "`deserialize(format='json')` is not yet implemented"
    raise NotImplementedError(msg)


def _ensure_owner(obj: Any, cls: type[_Self], /) -> _Self:  # pragma: no cover
    if isinstance(obj, cls):
        return obj

    import inspect

    module_name = "" if not (module := inspect.getmodule(obj)) else f"{module.__name__}."
    tp = obj if isinstance(obj, type) else type(obj)
    msg = (
        f"`{cls.__name__}.deserialize` returned an unexpected type:\n\n"
        f"Expected: '{cls.__module__}.{cls.__name__}'\n"
        f"Actual  : '{module_name}{tp.__name__}'"
    )
    raise TypeError(msg)
