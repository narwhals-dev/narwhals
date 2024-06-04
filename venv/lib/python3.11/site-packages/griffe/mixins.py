"""This module contains some mixins classes about accessing and setting members."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Sequence, TypeVar

from griffe.enumerations import Kind
from griffe.exceptions import AliasResolutionError, CyclicAliasError
from griffe.logger import get_logger
from griffe.merger import merge_stubs

if TYPE_CHECKING:
    from griffe.dataclasses import Alias, Attribute, Class, Function, Module, Object

logger = get_logger(__name__)
_ObjType = TypeVar("_ObjType")


def _get_parts(key: str | Sequence[str]) -> Sequence[str]:
    if isinstance(key, str):
        if not key:
            raise ValueError("Empty strings are not supported")
        parts = key.split(".")
    else:
        parts = list(key)
    if not parts:
        raise ValueError("Empty tuples are not supported")
    return parts


class GetMembersMixin:
    """Mixin class to share methods for accessing members."""

    def __getitem__(self, key: str | Sequence[str]) -> Any:
        """Get a member with its name or path.

        This method is part of the consumer API:
        do not use when producing Griffe trees!

        Members will be looked up in both declared members and inherited ones,
        triggering computation of the latter.

        Parameters:
            key: The name or path of the member.

        Examples:
            >>> foo = griffe_object["foo"]
            >>> bar = griffe_object["path.to.bar"]
            >>> qux = griffe_object[("path", "to", "qux")]
        """
        parts = _get_parts(key)
        if len(parts) == 1:
            return self.all_members[parts[0]]  # type: ignore[attr-defined]
        return self.all_members[parts[0]][parts[1:]]  # type: ignore[attr-defined]

    def get_member(self, key: str | Sequence[str]) -> Any:
        """Get a member with its name or path.

        This method is part of the producer API:
        you can use it safely while building Griffe trees
        (for example in Griffe extensions).

        Members will be looked up in declared members only, not inherited ones.

        Parameters:
            key: The name or path of the member.

        Examples:
            >>> foo = griffe_object["foo"]
            >>> bar = griffe_object["path.to.bar"]
            >>> bar = griffe_object[("path", "to", "bar")]
        """
        parts = _get_parts(key)
        if len(parts) == 1:
            return self.members[parts[0]]  # type: ignore[attr-defined]
        return self.members[parts[0]].get_member(parts[1:])  # type: ignore[attr-defined]


class DelMembersMixin:
    """Mixin class to share methods for deleting members."""

    def __delitem__(self, key: str | Sequence[str]) -> None:
        """Delete a member with its name or path.

        This method is part of the consumer API:
        do not use when producing Griffe trees!

        Members will be looked up in both declared members and inherited ones,
        triggering computation of the latter.

        Parameters:
            key: The name or path of the member.

        Examples:
            >>> del griffe_object["foo"]
            >>> del griffe_object["path.to.bar"]
            >>> del griffe_object[("path", "to", "qux")]
        """
        parts = _get_parts(key)
        if len(parts) == 1:
            name = parts[0]
            try:
                del self.members[name]  # type: ignore[attr-defined]
            except KeyError:
                del self.inherited_members[name]  # type: ignore[attr-defined]
        else:
            del self.all_members[parts[0]][parts[1:]]  # type: ignore[attr-defined]

    def del_member(self, key: str | Sequence[str]) -> None:
        """Delete a member with its name or path.

        This method is part of the producer API:
        you can use it safely while building Griffe trees
        (for example in Griffe extensions).

        Members will be looked up in declared members only, not inherited ones.

        Parameters:
            key: The name or path of the member.

        Examples:
            >>> griffe_object.del_member("foo")
            >>> griffe_object.del_member("path.to.bar")
            >>> griffe_object.del_member(("path", "to", "qux"))
        """
        parts = _get_parts(key)
        if len(parts) == 1:
            name = parts[0]
            del self.members[name]  # type: ignore[attr-defined]
        else:
            self.members[parts[0]].del_member(parts[1:])  # type: ignore[attr-defined]


class SetMembersMixin:
    """Mixin class to share methods for setting members."""

    def __setitem__(self, key: str | Sequence[str], value: Object | Alias) -> None:
        """Set a member with its name or path.

        This method is part of the consumer API:
        do not use when producing Griffe trees!

        Parameters:
            key: The name or path of the member.
            value: The member.

        Examples:
            >>> griffe_object["foo"] = foo
            >>> griffe_object["path.to.bar"] = bar
            >>> griffe_object[("path", "to", "qux")] = qux
        """
        parts = _get_parts(key)
        if len(parts) == 1:
            name = parts[0]
            self.members[name] = value  # type: ignore[attr-defined]
            if self.is_collection:  # type: ignore[attr-defined]
                value._modules_collection = self  # type: ignore[union-attr]
            else:
                value.parent = self  # type: ignore[assignment]
        else:
            self.members[parts[0]][parts[1:]] = value  # type: ignore[attr-defined]

    def set_member(self, key: str | Sequence[str], value: Object | Alias) -> None:
        """Set a member with its name or path.

        This method is part of the producer API:
        you can use it safely while building Griffe trees
        (for example in Griffe extensions).

        Parameters:
            key: The name or path of the member.
            value: The member.

        Examples:
            >>> griffe_object.set_member("foo", foo)
            >>> griffe_object.set_member("path.to.bar", bar)
            >>> griffe_object.set_member(("path", "to", "qux"), qux)
        """
        parts = _get_parts(key)
        if len(parts) == 1:
            name = parts[0]
            if name in self.members:  # type: ignore[attr-defined]
                member = self.members[name]  # type: ignore[attr-defined]
                if not member.is_alias:
                    # When reassigning a module to an existing one,
                    # try to merge them as one regular and one stubs module
                    # (implicit support for .pyi modules).
                    if member.is_module and not (member.is_namespace_package or member.is_namespace_subpackage):
                        with suppress(AliasResolutionError, CyclicAliasError):
                            if value.is_module and value.filepath != member.filepath:
                                with suppress(ValueError):
                                    value = merge_stubs(member, value)  # type: ignore[arg-type]
                    for alias in member.aliases.values():
                        with suppress(CyclicAliasError):
                            alias.target = value
            self.members[name] = value  # type: ignore[attr-defined]
            if self.is_collection:  # type: ignore[attr-defined]
                value._modules_collection = self  # type: ignore[union-attr]
            else:
                value.parent = self  # type: ignore[assignment]
        else:
            self.members[parts[0]].set_member(parts[1:], value)  # type: ignore[attr-defined]


class SerializationMixin:
    """A mixin that adds de/serialization conveniences."""

    def as_json(self, *, full: bool = False, **kwargs: Any) -> str:
        """Return this object's data as a JSON string.

        Parameters:
            full: Whether to return full info, or just base info.
            **kwargs: Additional serialization options passed to encoder.

        Returns:
            A JSON string.
        """
        from griffe.encoders import JSONEncoder  # avoid circular import

        return json.dumps(self, cls=JSONEncoder, full=full, **kwargs)

    @classmethod
    def from_json(cls: type[_ObjType], json_string: str, **kwargs: Any) -> _ObjType:  # noqa: PYI019
        """Create an instance of this class from a JSON string.

        Parameters:
            json_string: JSON to decode into Object.
            **kwargs: Additional options passed to decoder.

        Returns:
            An Object instance.

        Raises:
            TypeError: When the json_string does not represent and object
                of the class from which this classmethod has been called.
        """
        from griffe.encoders import json_decoder  # avoid circular import

        kwargs.setdefault("object_hook", json_decoder)
        obj = json.loads(json_string, **kwargs)
        if not isinstance(obj, cls):
            raise TypeError(f"provided JSON object is not of type {cls}")
        return obj


class ObjectAliasMixin(GetMembersMixin, SetMembersMixin, DelMembersMixin, SerializationMixin):
    """A mixin for methods that appear both in objects and aliases, unchanged."""

    @property
    def all_members(self) -> dict[str, Object | Alias]:
        """All members (declared and inherited).

        This method is part of the consumer API:
        do not use when producing Griffe trees!
        """
        return {**self.inherited_members, **self.members}  # type: ignore[attr-defined]

    @property
    def modules(self) -> dict[str, Module]:
        """The module members.

        This method is part of the consumer API:
        do not use when producing Griffe trees!
        """
        return {name: member for name, member in self.all_members.items() if member.kind is Kind.MODULE}  # type: ignore[misc]

    @property
    def classes(self) -> dict[str, Class]:
        """The class members.

        This method is part of the consumer API:
        do not use when producing Griffe trees!
        """
        return {name: member for name, member in self.all_members.items() if member.kind is Kind.CLASS}  # type: ignore[misc]

    @property
    def functions(self) -> dict[str, Function]:
        """The function members.

        This method is part of the consumer API:
        do not use when producing Griffe trees!
        """
        return {name: member for name, member in self.all_members.items() if member.kind is Kind.FUNCTION}  # type: ignore[misc]

    @property
    def attributes(self) -> dict[str, Attribute]:
        """The attribute members.

        This method is part of the consumer API:
        do not use when producing Griffe trees!
        """
        return {name: member for name, member in self.all_members.items() if member.kind is Kind.ATTRIBUTE}  # type: ignore[misc]

    def is_exported(self, *, explicitely: bool = True) -> bool:
        """Tell if this object/alias is implicitely exported by its parent.

        Parameters:
            explicitely: Whether to only return True when `__all__` is defined.

        Returns:
            True or False.
        """
        return self.parent.member_is_exported(self, explicitely=explicitely)  # type: ignore[attr-defined]

    @property
    def is_explicitely_exported(self) -> bool:
        """Whether this object/alias is explicitely exported by its parent."""
        return self.is_exported(explicitely=True)

    @property
    def is_implicitely_exported(self) -> bool:
        """Whether this object/alias is implicitely exported by its parent."""
        return self.parent.exports is None  # type: ignore[attr-defined]

    def is_public(
        self,
        *,
        strict: bool = False,
        check_name: bool = True,
    ) -> bool:
        """Whether this object is considered public.

        In modules, developers can mark objects as public thanks to the `__all__` variable.
        In classes however, there is no convention or standard to do so.

        Therefore, to decide whether an object is public, we follow this algorithm:

        - If the object's `public` attribute is set (boolean), return its value.
        - In strict mode, the object is public only if it is explicitely exported (listed in `__all__`).
            Strict mode should only be used for module members.
        - Otherwise, if name checks are enabled, the object is private if its name starts with an underscore.
        - Otherwise, if the object is an alias, and is neither inherited from a base class,
            nor a member of a parent alias, it is not public.
        - Otherwise, the object is public.
        """
        if self.public is not None:  # type: ignore[attr-defined]
            return self.public  # type: ignore[attr-defined]
        if self.is_explicitely_exported:
            return True
        if strict:
            return False
        if check_name and self.name.startswith("_"):  # type: ignore[attr-defined]
            return False
        if self.is_alias and not (self.inherited or self.parent.is_alias):  # type: ignore[attr-defined]
            return False
        return True


__all__ = [
    "DelMembersMixin",
    "GetMembersMixin",
    "ObjectAliasMixin",
    "SerializationMixin",
    "SetMembersMixin",
]
