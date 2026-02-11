from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol

from narwhals._plan._immutable import Immutable
from narwhals._typing_compat import TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

T_co = TypeVar("T_co", bound="Traversable[Any]", covariant=True)


class Traversable(Protocol[T_co]):
    def iter_left(self) -> Iterator[Traversable[T_co]]:
        """Yield nodes recursively from root->leaf."""
        ...

    def iter_right(self) -> Iterator[Traversable[T_co]]:
        """Yield nodes recursively from leaf->root."""
        ...

    def iter_inputs(self) -> Iterator[Traversable[T_co]]:
        """Yield direct input nodes to leaf."""
        ...


class _BasePlan(Immutable, Generic[T_co]):
    has_inputs: ClassVar[bool]
    """Cheap check for `Scan` vs `SingleInput | MultipleInputs`"""

    def __init_subclass__(
        cls: type[Self],
        *args: Any,
        has_inputs: bool | None = None,
        _root: bool = False,
        **kwds: Any,
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if has_inputs is not None:
            cls.has_inputs = has_inputs
        elif getattr(cls, "has_inputs", None) is None and not _root:
            raise _invalid_subclass_error(cls)

    def iter_left(self) -> Iterator[T_co]:
        """Yield nodes recursively from root->leaf."""
        msg = f"TODO: `{type(self).__name__}.iter_left`"
        raise NotImplementedError(msg)

    def iter_right(self) -> Iterator[T_co]:
        """Yield nodes recursively from leaf->root."""
        msg = f"TODO: `{type(self).__name__}.iter_right`"
        raise NotImplementedError(msg)

    def iter_inputs(self) -> Iterator[T_co]:
        """Yield direct input nodes to leaf.

        Equivalent to [`IR.inputs`] and [`ir::Inputs`].

        [`IR.inputs`]: https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/ir/inputs.rs#L204-L239
        [`ir::Inputs`]: https://github.com/pola-rs/polars/blob/40c171f9725279cd56888f443bd091eea79e5310/crates/polars-plan/src/plans/ir/inputs.rs#L301-L335
        """
        msg = f"TODO: `{type(self).__name__}.iter_inputs`"
        raise NotImplementedError(msg)


def _invalid_subclass_error(child: type[_BasePlan[Any]]) -> TypeError:
    # https://docs.python.org/3/reference/datamodel.html#type.__base__
    # https://docs.python.org/3/reference/datamodel.html#type.__bases__
    bases = child.__bases__
    used_in_class_def = ", ".join(base.__name__ for base in bases)
    direct_immutable_parent = (child.__base__ or bases[0]).__name__
    msg = (
        f"`has_inputs` is a required argument in immediate subclasses of {direct_immutable_parent!r}.\n"
        f"Hint: instead try\n"
        f"    class {child.__name__}({used_in_class_def}, has_inputs=True): ..."
    )
    return TypeError(msg)
