"""Hoping to add better support for [PEP 681].

See [Supporting custom decorators], [extensions/dataclasses.py], [griffe_pydantic]

Run with:

    uvx griffe dump --extensions utils/griffe_extension.py src/narwhals

[PEP 681]: https://peps.python.org/pep-0681/
[Supporting custom decorators]: https://mkdocstrings.github.io/griffe/guide/users/how-to/support-decorators/
[extensions/dataclasses.py]: https://github.com/mkdocstrings/griffe/blob/5dc97b78f318ac97016fbaee83a574ff15516f58/packages/griffelib/src/griffe/_internal/extensions/dataclasses.py
[griffe_pydantic]: https://github.com/mkdocstrings/griffe-pydantic/blob/9df87609b4edc2e548d15691e4d6c26710861daa/src/griffe_pydantic/_internal/extension.py
"""

from __future__ import annotations

import datetime as dt
import logging
import pathlib

# ruff: noqa: DTZ005, G004
from functools import cache
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import griffe
from griffe import Attribute, Class, ExprName, Function, Module, Parameter, Parameters

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from narwhals.typing import FileSource

    logger = logging.getLogger(__name__)
else:
    # NOTE: griff's logger has a `__getattr__`
    logger = griffe.get_logger(__name__)


NEEDS_FIX = "IRDateTimeNamespace"
CANONICAL_PATH_PLAN = "narwhals._plan"
CANONICAL_PATH_IMMUTABLE = f"{CANONICAL_PATH_PLAN}._immutable.Immutable"
"""The metaclass of `Immutable` is decorated with `@dataclass_transform`.

- The simple check is to see if `Immutable` is in our mro.
- A more complex alternative would be using [`griffe.get_class_keyword`] to find `metaclass=ImmutableMeta`
    - Would make sense here if I'd used any variations other besides
      `@dataclass_transform(kw_only_default=True, frozen_default=True)`

[`griffe.get_class_keyword`]: https://mkdocstrings.github.io/griffe/reference/api/expressions/#griffe.get_class_keyword
"""

positional_or_keyword: Final = griffe.ParameterKind.positional_or_keyword
"""`self`."""
keyword_only: Final = griffe.ParameterKind.keyword_only
"""All fields are keyword-only."""


class PEP681Extension(griffe.Extension):
    """An extension to support `@dataclass_transform`.

    Important:
        Worked backwords from [`extensions/dataclasses.py`](https://github.com/mkdocstrings/griffe/blob/5dc97b78f318ac97016fbaee83a574ff15516f58/packages/griffelib/src/griffe/_internal/extensions/dataclasses.py)
    """

    def __init__(self) -> None:
        super().__init__()

    def on_package(self, *, pkg: Module, **kwargs: Any) -> None:
        path = pkg.canonical_path
        logger.info("Starting: %r", path)
        _apply_recursively(pkg, set())
        logger.info("Finished: %r", path)


def _apply_recursively(mod_cls: Module | Class, processed: set[str]) -> None:
    """`griffe._internal.extensions.dataclasses.__apply_recursively`.

    `processed` means **seen**.
    """
    if mod_cls.canonical_path in processed:
        return
    processed.add(mod_cls.canonical_path)
    if isinstance(mod_cls, Class):
        if "__init__" not in mod_cls.members:
            _set_dataclass_init(mod_cls)
        for member in mod_cls.members.values():
            if not member.is_alias and member.is_class:
                _apply_recursively(member, processed)  # pyright: ignore[reportArgumentType]
    elif isinstance(mod_cls, Module):
        for member in mod_cls.members.values():
            if not member.is_alias and (member.is_module or member.is_class):
                _apply_recursively(member, processed)  # pyright: ignore[reportArgumentType]


def _set_dataclass_init(class_: Class) -> None:
    """`griffe._internal.extensions.dataclasses._set_dataclass_init`."""
    # Retrieve parameters from all parent dataclasses.
    parameters: list[Parameter] = []
    for parent in reversed(class_.mro()):
        if inherits_immutable(parent):
            parameters.extend(_dataclass_parameters(parent))
            class_.labels.add("dataclass")

    if not inherits_immutable(class_):
        return

    _logger = logger.info if class_.name == NEEDS_FIX else logger.debug
    _logger("Handling: %r", class_.canonical_path.removeprefix(CANONICAL_PATH_PLAN))

    # Add current class parameters.
    parameters.extend(_dataclass_parameters(class_))
    if not parameters:
        _logger("|   Skipped (no params)")
        return

    _logger(
        f"|   Parameters(*, {', '.join(f'{p.name}: {p.annotation}' for p in parameters)})"
    )

    init = init_fn(class_, parameters)
    class_.set_member("__init__", init)
    _logger(f"|   {init.signature(name=class_.name + '.__init__')}")


def init_fn(cls: Class, parameters: Iterable[Parameter]) -> Function:
    self = Parameter("self", kind=positional_or_keyword)
    p = Parameters(self, *parameters)
    return Function(
        "__init__", lineno=0, endlineno=0, parent=cls, parameters=p, returns="None"
    )


@cache
def inherits_immutable(cls: Class) -> bool:
    root = CANONICAL_PATH_IMMUTABLE
    if any(
        (base if isinstance(base, str) else base.canonical_path) == root
        for base in cls.bases
    ):
        return True
    return any(
        inherits_immutable(parent)
        for parent in cls.mro()
        if parent.canonical_path.startswith(CANONICAL_PATH_PLAN)
    )


# TODO @dangotbanned: `ExprIR` is showing `node(s)` as a default
@cache
def _dataclass_parameters(class_: Class) -> tuple[Parameter, ...]:
    if class_.name == NEEDS_FIX and logger.isEnabledFor(logging.DEBUG):
        write_griffe(class_)
    return tuple(iter_dataclass_parameters(class_))


def iter_dataclass_parameters(cls: Class) -> Iterator[Parameter]:
    members = cast("Iterable[Attribute]", cls.members.values())
    for member in members:
        if member.is_attribute and is_dataclass_field(member):
            yield Parameter(
                member.name,
                annotation=member.annotation,
                kind=keyword_only,
                default=member.value,
                docstring=member.docstring,
            )


def is_dataclass_field(member: Attribute) -> bool:
    """Fixes logic from upstream inference of `ClassVar`.

    ([mkdocstrings/griffe#253]) handled the `ClassVar[<type>]` case, but following ([python/typing#1931])
    we can write a "bare" `ClassVar` too.

    [mkdocstrings/griffe#253]: https://github.com/mkdocstrings/griffe/pull/253
    [python/typing#1931]: https://github.com/python/typing/pull/1931
    """
    # All dataclass parameters have annotations.
    if member.annotation is None:
        return False
    # Attributes that have labels for these characteristics are not class parameters:
    # - @property
    # - @cached_property
    # - ClassVar annotation
    labels = member.labels
    return not (
        "property" in labels
        or ("class-attribute" in labels and "instance-attribute" not in labels)
        or (
            isinstance(member.annotation, ExprName)
            and "ClassVar" in member.annotation.name
        )
    )


class GriffeExportable(Protocol):
    def as_json(self, **kwds: Any) -> str: ...


def write_griffe(
    obj: GriffeExportable, file: FileSource | None = None, **kwds: Any
) -> None:
    """Export a `griffe` object as json.

    This is a quick debugging tool to help see what a specific slice of the ast look like.

    Arguments:
        obj: An object from `griffe` that supports `as_json`.
        file: File path to write to.
            By default, writes to the cwd using a best-effort name, followed by a timestamp.
        **kwds: Arguments forwarded to `json.dumps`
    """
    serde = obj.as_json(**kwds)
    if file is None:
        if (
            name := getattr(
                obj, "canonical_path", getattr(obj, "path", getattr(obj, "name", None))
            )
        ) is None:
            name = type(obj).__name__
        now = dt.datetime.now(tz=None).isoformat(timespec="seconds").replace(":", "-")
        file = f"{name.removeprefix(CANONICAL_PATH_PLAN + '.')}_{now}.json"

    path = pathlib.Path(file)
    path.touch()
    path.write_text(serde, "utf8")
    logger.info(f"Exported: {path.as_posix()}")
