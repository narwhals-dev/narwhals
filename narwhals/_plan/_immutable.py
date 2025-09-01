from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Callable

    from typing_extensions import Never, Self, dataclass_transform

else:
    # NOTE: This isn't important to the proposal, just wanted IDE support
    # for the **temporary** constructors.
    # It is interesting how much boilerplate this avoids though 🤔
    # https://docs.python.org/3/library/typing.html#typing.dataclass_transform
    def dataclass_transform(
        *,
        eq_default: bool = True,
        order_default: bool = False,
        kw_only_default: bool = False,
        frozen_default: bool = False,
        field_specifiers: tuple[type[Any] | Callable[..., Any], ...] = (),
        **kwargs: Any,
    ) -> Callable[[T], T]:
        def decorator(cls_or_fn: T) -> T:
            cls_or_fn.__dataclass_transform__ = {
                "eq_default": eq_default,
                "order_default": order_default,
                "kw_only_default": kw_only_default,
                "frozen_default": frozen_default,
                "field_specifiers": field_specifiers,
                "kwargs": kwargs,
            }
            return cls_or_fn

        return decorator


T = TypeVar("T")
_IMMUTABLE_HASH_NAME: Literal["__immutable_hash_value__"] = "__immutable_hash_value__"


@dataclass_transform(kw_only_default=True, frozen_default=True)
class Immutable:
    """A poor man's frozen dataclass.

    - Keyword-only constructor (IDE supported)
    - Manual `__slots__` required
    - Compatible with [`copy.replace`]
    - No handling for default arguments

    [`copy.replace`]: https://docs.python.org/3.13/library/copy.html#copy.replace
    """

    __slots__ = (_IMMUTABLE_HASH_NAME,)
    __immutable_hash_value__: int

    @property
    def __immutable_keys__(self) -> Iterator[str]:
        slots: tuple[str, ...] = self.__slots__
        for name in slots:
            if name != _IMMUTABLE_HASH_NAME:
                yield name

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        for name in self.__immutable_keys__:
            yield getattr(self, name)

    @property
    def __immutable_items__(self) -> Iterator[tuple[str, Any]]:
        for name in self.__immutable_keys__:
            yield name, getattr(self, name)

    @property
    def __immutable_hash__(self) -> int:
        if hasattr(self, _IMMUTABLE_HASH_NAME):
            return self.__immutable_hash_value__
        hash_value = hash((self.__class__, *self.__immutable_values__))
        object.__setattr__(self, _IMMUTABLE_HASH_NAME, hash_value)
        return self.__immutable_hash_value__

    def __setattr__(self, name: str, value: Never) -> Never:
        msg = f"{type(self).__name__!r} is immutable, {name!r} cannot be set."
        raise AttributeError(msg)

    def __replace__(self, **changes: Any) -> Self:
        """https://docs.python.org/3.13/library/copy.html#copy.replace"""  # noqa: D415
        if len(changes) == 1:
            k_new, v_new = next(iter(changes.items()))
            # NOTE: Will trigger an attribute error if invalid name
            if getattr(self, k_new) == v_new:
                return self
            changed = dict(self.__immutable_items__)
            # Now we *don't* need to check the key is valid
            changed[k_new] = v_new
        else:
            changed = dict(self.__immutable_items__)
            changed |= changes
        return type(self)(**changed)

    def __init_subclass__(cls, *args: Any, **kwds: Any) -> None:
        super().__init_subclass__(*args, **kwds)
        if cls.__slots__:
            ...
        else:
            cls.__slots__ = ()

    def __hash__(self) -> int:
        return self.__immutable_hash__

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        return all(
            getattr(self, key) == getattr(other, key) for key in self.__immutable_keys__
        )

    def __str__(self) -> str:
        # NOTE: Debug repr, closer to constructor
        fields = ", ".join(f"{_field_str(k, v)}" for k, v in self.__immutable_items__)
        return f"{type(self).__name__}({fields})"

    def __init__(self, **kwds: Any) -> None:
        # NOTE: DUMMY CONSTRUCTOR - don't use beyond prototyping!
        # Just need a quick way to demonstrate `ExprIR` and interactions
        required: set[str] = set(self.__immutable_keys__)
        if not required and not kwds:
            # NOTE: Fastpath for empty slots
            ...
        elif required == set(kwds):
            # NOTE: Everything is as expected
            for name, value in kwds.items():
                object.__setattr__(self, name, value)
        elif missing := required.difference(kwds):
            msg = (
                f"{type(self).__name__!r} requires attributes {sorted(required)!r}, \n"
                f"but missing values for {sorted(missing)!r}"
            )
            raise TypeError(msg)
        else:
            extra = set(kwds).difference(required)
            msg = (
                f"{type(self).__name__!r} only supports attributes {sorted(required)!r}, \n"
                f"but got unknown arguments {sorted(extra)!r}"
            )
            raise TypeError(msg)


def _field_str(name: str, value: Any) -> str:
    if isinstance(value, tuple):
        inner = ", ".join(f"{v}" for v in value)
        return f"{name}=[{inner}]"
    if isinstance(value, str):
        return f"{name}={value!r}"
    return f"{name}={value}"
