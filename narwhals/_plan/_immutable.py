from __future__ import annotations

from typing import TYPE_CHECKING

# ruff: noqa: N806
from narwhals._plan._meta import ImmutableMeta

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any, ClassVar, Final

    from typing_extensions import Never, Self


_HASH_NAME: Final = "__immutable_hash_value__"


class Immutable(metaclass=ImmutableMeta):
    """A poor man's frozen dataclass.

    - Keyword-only constructor (IDE supported)
    - Manual `__slots__` required
    - Compatible with [`copy.replace`]
    - No handling for default arguments

    [`copy.replace`]: https://docs.python.org/3.13/library/copy.html#copy.replace
    """

    __slots__ = (_HASH_NAME,)
    if not TYPE_CHECKING:
        # NOTE: Trying to avoid this being added to synthesized `__init__`
        # Seems to be the only difference when decorating the metaclass
        __immutable_hash_value__: int
    else:  # pragma: no cover
        ...

    __immutable_keys__: ClassVar[tuple[str, ...]]

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        """Override to configure hash seed."""
        getattr_ = getattr
        for name in self.__immutable_keys__:
            yield getattr_(self, name)

    @property
    def __immutable_items__(self) -> Iterator[tuple[str, Any]]:
        getattr_ = getattr
        for name in self.__immutable_keys__:
            yield name, getattr_(self, name)

    @property
    def __immutable_hash__(self) -> int:
        HASH = _HASH_NAME
        if hasattr(self, HASH):
            hash_value: int = getattr(self, HASH)
        else:
            hash_value = hash((self.__class__, *self.__immutable_values__))
            object.__setattr__(self, HASH, hash_value)
        return hash_value

    def __setattr__(self, name: str, value: Never) -> Never:
        msg = f"{type(self).__name__!r} is immutable, {name!r} cannot be set."
        raise AttributeError(msg)

    def __replace__(self, **changes: Any) -> Self:
        """https://docs.python.org/3.13/library/copy.html#copy.replace."""
        if len(changes) == 1:
            # The most common case is a single field replacement.
            # Iff that field happens to be equal, we can noop, preserving the current object's hash.
            name, value_changed = next(iter(changes.items()))
            if getattr(self, name) == value_changed:
                return self
            changes = dict(self.__immutable_items__, **changes)
        else:
            for name, value_current in self.__immutable_items__:
                if name not in changes or value_current == changes[name]:
                    changes[name] = value_current
        return type(self)(**changes)

    def __hash__(self) -> int:
        return self.__immutable_hash__

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        getattr_ = getattr
        return all(
            getattr_(self, key) == getattr_(other, key) for key in self.__immutable_keys__
        )

    def __str__(self) -> str:
        fields = ", ".join(f"{_field_str(k, v)}" for k, v in self.__immutable_items__)
        return f"{type(self).__name__}({fields})"

    def __init__(self, **kwds: Any) -> None:
        if (keys := self.__immutable_keys__) or kwds:
            required = set(keys)
            if required == kwds.keys():
                object__setattr__ = object.__setattr__
                for name, value in kwds.items():
                    object__setattr__(self, name, value)
            elif missing := required.difference(kwds):
                raise _init_missing_error(self, required, missing)
            else:
                raise _init_extra_error(self, required, set(kwds).difference(required))


def _field_str(name: str, value: Any) -> str:
    if isinstance(value, tuple):
        inner = ", ".join(f"{v}" for v in value)
        return f"{name}=[{inner}]"
    if isinstance(value, str):
        return f"{name}={value!r}"
    return f"{name}={value}"


def _init_missing_error(
    obj: object, required: Iterable[str], missing: Iterable[str]
) -> TypeError:
    msg = (
        f"{type(obj).__name__!r} requires attributes {sorted(required)!r}, \n"
        f"but missing values for {sorted(missing)!r}"
    )
    return TypeError(msg)


def _init_extra_error(
    obj: object, required: Iterable[str], extra: Iterable[str]
) -> TypeError:
    msg = (
        f"{type(obj).__name__!r} only supports attributes {sorted(required)!r}, \n"
        f"but got unknown arguments {sorted(extra)!r}"
    )
    return TypeError(msg)
