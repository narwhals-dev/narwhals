from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

# ruff: noqa: N806
from narwhals._plan._meta import ImmutableMeta

if TYPE_CHECKING:
    import dataclasses
    from collections.abc import Iterator
    from typing import Any, ClassVar, Final

    from typing_extensions import Never, Self


_HASH_NAME: Final = "__immutable_hash_value__"
_OBJ_SETATTR = object.__setattr__
"""Borrowed from [`attrs`](https://github.com/search?q=repo%3Apython-attrs%2Fattrs%20%22_OBJ_SETATTR%22&type=code).

- Adds the reference to `object.__setattr__` into globals
- `__init__` goes further and binds the descriptor
"""


class Immutable(metaclass=ImmutableMeta):
    """A poor man's [frozen `dataclass`].

    `Immutable` is used as a base class *very heavily* in this code base:
    - writing a constructor shouldn't be a barrier to using a class
    - mutability is hard to reason about in *arbitrarily* nested objects
    - the ability to enable caching **anywhere** is fun 🙂

    Note:
        Examples adapted from [`dataclasses_transform_meta.py`] conformance suite.

    An `Immutable` class declares it's fields via `__slots__`:
    >>> class Customer1(Immutable):
    ...     __slots__ = ("id", "name", "name2")
    ...     id: int
    ...     name: str
    ...     name2: str

    These names are accepted in the synthesized constructor (thanks to [`@dataclass_transform`]):
    >>> c1_1 = Customer1(id=3, name="Sue", name2="Susan")
    >>> print(c1_1)
    Customer1(id=3, name='Sue', name2='Susan')

    All fields are required:
    >>> Customer1(id=3, name="John")
    Traceback (most recent call last):
    TypeError...

    And must be passed as keywords:
    >>> Customer1(3, "Sue", "Susan")
    Traceback (most recent call last):
    TypeError...

    Default values are not supported:
    >>> class Customer2(Immutable):
    ...     __slots__ = ("id", "name", "name2")
    ...     id: int
    ...     name: str
    ...     name2: str = "None"
    Traceback (most recent call last):
    ValueError...

    Assignment is quite frowned upon:
    >>> c1_1.id = 4
    Traceback (most recent call last):
    AttributeError: 'Customer1' is immutable, 'id' cannot be set.

    But you can create a new instance by [replacing] the values of one or more fields:
    >>> c1_2 = c1_1.__replace__(id=4)
    >>> print(c1_2)
    Customer1(id=4, name='Sue', name2='Susan')

    Instances compare equal if all fields do:
    >>> c1_1 == Customer1(name="Sue", id=3, name2="Susan")
    True
    >>> c1_1 == c1_2
    False
    >>> c1_2 == Customer1(id=4, name="Sue", name2="Susan")
    True

    When fields use [atomic] or `Immutable` types, instances are hash-friendly:
    >>> customers = {
    ...     c1_1,
    ...     c1_2,
    ...     Customer1(name="Sue", id=3, name2="Susan"),
    ...     Customer1(id=4, name="Sue", name2="Susan"),
    ... }
    >>> len(customers) == 2
    True
    >>> customers == {c1_1, c1_2}
    True

    A subclass can extend the fields of a parent:
    >>> class Customer1Sub(Customer1):
    ...     __slots__ = ("salary",)
    ...     salary: float
    >>> print(Customer1Sub(id=3, name="Sue", name2="Susan", salary=1))
    Customer1Sub(id=3, name='Sue', name2='Susan', salary=1)

    Although this does break the [Liskov substitution principle]:
    >>> Customer1Sub(id=3, name="Sue", name2="Susan")
    Traceback (most recent call last):
    TypeError...

    [frozen `dataclass`]: https://docs.python.org/3/library/dataclasses.html#frozen-instances
    [`dataclasses_transform_meta.py`]: https://github.com/python/typing/blob/1df1565c69730d88ce6877009d268ba1d602af1e/conformance/tests/dataclasses_transform_meta.py
    [`@dataclass_transform`]: https://typing.python.org/en/latest/spec/dataclasses.html#the-dataclass-transform-decorator
    [replacing]: https://docs.python.org/3.13/library/copy.html#copy.replace
    [atomic]: https://github.com/python/cpython/blob/656abe3c9a228d20b2455f216a5a94b1a752495f/Lib/copy.py#L103-L107
    [Liskov substitution principle]: https://en.wikipedia.org/wiki/Liskov_substitution_principle
    """

    __slots__ = (_HASH_NAME,)
    if TYPE_CHECKING:
        # NOTE: Omiting the annotation avoids this being added to synthesized `__init__`
        # https://typing.python.org/en/latest/spec/dataclasses.html#field-specifier-parameters
        __immutable_hash_value__ = dataclasses.field(init=False)
    else:
        __immutable_hash_value__: int

    __immutable_keys__: ClassVar[tuple[str, ...]]
    """The names of fields defined for the class.

    Each is a required, keyword-only parameter to `__init__`.

    Populated via piggy-backing off the names added to `__slots__`.
    """

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        """Yield the values of fields for this instance.

        Alongside `__immutable_{keys,items}__`, provides a `Mapping`-like
        iterable protocol.

        Consider overriding to omit specific fields from the hash.
        """
        get = self.__getattribute__
        for name in self.__immutable_keys__:
            yield get(name)

    @property
    @final
    def __immutable_items__(self) -> Iterator[tuple[str, Any]]:
        """Yield `(name, value)` pairs describing this instance's fields."""
        get = self.__getattribute__
        for name in self.__immutable_keys__:
            yield name, get(name)

    @property
    def __immutable_hash__(self) -> int:
        """Return the hash value of the instance.

        Lazily computed **once-per-instance** and reused for quick comparisons.
        """
        # TODO @dangotbanned: Would prefer to work more like `@functools.cached_property`
        # but using a pre-defined slot vs `__dict__`
        HASH = _HASH_NAME
        if hasattr(self, HASH):
            hash_value: int = self.__immutable_hash_value__
        else:
            hash_value = hash((self.__class__, *self.__immutable_values__))
            _OBJ_SETATTR(self, HASH, hash_value)
        return hash_value

    def to_dict(self) -> dict[str, Any]:
        """Convert this instance's fields to a dictionary.

        Similar to [`dataclasses.asdict`], but **non-recursive**.

        Tip:
            Unless you plan to use `**to_dict()`, prefer iterating over `__immutable_items__` instead.

        [`dataclasses.asdict`]: https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
        """
        return dict(self.__immutable_items__)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: Any) -> Self:
        return self

    def __hash__(self) -> int:
        """Do not override [`__hash__`] in an `Immutable` subclass.

        If you want to:
        - Omit specific values from the hash?
          - override `__immutable_values__`
        - Override `__eq__` in a subclass, but now [`__hash__`] is broken?
          - use `__hash__ = Immutable.__hash__`
        - Change how the write-once hash value is stored?
          - override `__immutable_hash__`

        [`__hash__`]: https://docs.python.org/3/reference/datamodel.html#object.__hash__
        """
        return self.__immutable_hash__

    def __eq__(self, other: object) -> bool:
        # NOTE: Same as `@dataclass(eq=True)`
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        # NOTE: No observable difference (in current tests) when switching to:
        #   `hash(self) == hash(other)`
        # Save more optimizing until there's some granular benchmarks, e.g.:
        # - https://github.com/python-attrs/attrs/blob/4885c5b1af4e9fa4d97b6bffa2fb78a2efa5f047/bench/test_benchmarks.py
        # - https://github.com/python-attrs/attrs/blob/4885c5b1af4e9fa4d97b6bffa2fb78a2efa5f047/tox.ini#L69-L77
        # - https://github.com/python-attrs/attrs/blob/4885c5b1af4e9fa4d97b6bffa2fb78a2efa5f047/.github/workflows/codspeed.yml
        get_self, get_other = self.__getattribute__, other.__getattribute__
        return all(get_self(key) == get_other(key) for key in self.__immutable_keys__)

    def __str__(self) -> str:
        """Simple-minded `Class(key_n=value_n, ...)` debug representation.

        Instead of overriding `__str__` in a subclass, consider implementing `__repr__`
        for something more fancy.
        """
        fields = ", ".join(f"{_field_str(k, v)}" for k, v in self.__immutable_items__)
        return f"{type(self).__name__}({fields})"

    def __init__(self, **kwds: Any) -> None:
        # NOTE: Matching lengths means either:
        # 1. We have the correct args (great)
        # 2. At least one is incorrect (__slots__ will raise an `AttributeError`)
        if (n := len(kwds)) == len(self.__immutable_keys__):
            if n:
                self__setattr__ = _OBJ_SETATTR.__get__(self)
                for name, value in kwds.items():
                    self__setattr__(name, value)
        else:
            raise _init_error(self, kwds)

    if TYPE_CHECKING:
        ...
    else:
        # NOTE: Everything here is synthesized by `@dataclass_transform`
        # Having it visible to a type checker can cause weird things to happen
        # - `mypy` pretty much doesn't understand it
        # - `pyright` is fine until a subclass overrides something
        # https://github.com/pydantic/pydantic/blob/ac249284616890d91c746dc890fb0f6407df2843/pydantic/main.py#L1011-L1143

        def __setattr__(self, name: str, value: Never) -> Never:
            msg = f"{type(self).__name__!r} is immutable, {name!r} cannot be set."
            raise AttributeError(msg)

        def __delattr__(self, name: str) -> Never:
            msg = f"{type(self).__name__!r} is immutable, {name!r} cannot be deleted."
            raise AttributeError(msg)

        def __replace__(self, **changes: Any) -> Self:
            """Create a new object of the same type, [replacing] fields with values from `changes`.

            [replacing]: https://docs.python.org/3.13/library/copy.html#copy.replace
            """
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


# TODO @dangotbanned: Replace with `@singledispatch` on `type(value)`
def _field_str(name: str, value: Any) -> str:
    if isinstance(value, tuple):
        inner = ", ".join(
            (f"{v!s}" if not isinstance(v, str) else f"{v!r}") for v in value
        )
        return f"{name}=[{inner}]"
    if isinstance(value, str):
        return f"{name}={value!r}"
    return f"{name}={value}"


def _init_error(obj: Immutable, got: dict[str, Any], /) -> TypeError:
    # Slightly better error than you'd get from `inspect.Signature`
    want = obj.__immutable_keys__
    n = len(want)
    invalid = set(want).difference(got)
    wants, gots = ", ".join(map(repr, want)), ", ".join(map(repr, got))
    kind: Literal["got unexpected", "expected", "missing required"]
    if invalid and not set(want).intersection(got):
        kind = "got unexpected"
        body = gots
        n = len(got)
    elif invalid:
        kind = "expected"
        body = f"{wants}, \nbut got: {gots}"
    else:
        kind = "missing required"
        body = wants
    msg = f"{type(obj).__name__!r} {kind} {'argument' if n == 1 else 'arguments'}: {body}"
    return TypeError(msg)
