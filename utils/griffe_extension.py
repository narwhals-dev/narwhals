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

# ruff: noqa: DTZ005, G004
import datetime as dt
import logging
import pathlib
from functools import cache
from typing import TYPE_CHECKING, Any, Protocol, cast

import griffe

if TYPE_CHECKING:
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


class PEP681Extension(griffe.Extension):
    """An extension to support `@dataclass_transform`.

    Important:
        Worked backwords from [`extensions/dataclasses.py`](https://github.com/mkdocstrings/griffe/blob/5dc97b78f318ac97016fbaee83a574ff15516f58/packages/griffelib/src/griffe/_internal/extensions/dataclasses.py)
    """

    def __init__(self) -> None:
        super().__init__()

    def on_package(
        self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any
    ) -> None:
        path = pkg.canonical_path
        logger.info("Starting: %r", path)
        _apply_recursively(pkg, set())
        logger.info("Finished: %r", path)


def _apply_recursively(
    mod_cls: griffe.Module | griffe.Class, processed: set[str]
) -> None:
    """`griffe._internal.extensions.dataclasses.__apply_recursively`.

    `processed` means **seen**.
    """
    if mod_cls.canonical_path in processed:
        return
    processed.add(mod_cls.canonical_path)
    if isinstance(mod_cls, griffe.Class):
        if "__init__" not in mod_cls.members:
            _set_dataclass_init(mod_cls)
        for member in mod_cls.members.values():
            if not member.is_alias and member.is_class:
                _apply_recursively(member, processed)  # pyright: ignore[reportArgumentType]
    elif isinstance(mod_cls, griffe.Module):
        for member in mod_cls.members.values():
            if not member.is_alias and (member.is_module or member.is_class):
                _apply_recursively(member, processed)  # pyright: ignore[reportArgumentType]


def relative_path(cls: griffe.Object | griffe.Expr) -> str:
    """Strip the `narwhals._plan` prefix, for logging."""
    return _relative_path(cls.canonical_path)


@cache
def _relative_path(path: str) -> str:
    r = path.removeprefix(CANONICAL_PATH_PLAN)
    if r == path:
        msg = f"Expected all classes that reach here to start with {CANONICAL_PATH_PLAN!r}\nGot: {path!r}"
        raise NotImplementedError(msg)
    return r


def _set_dataclass_init(class_: griffe.Class) -> None:
    """`griffe._internal.extensions.dataclasses._set_dataclass_init`."""
    # Retrieve parameters from all parent dataclasses.
    parameters: list[griffe.Parameter] = []
    for parent in reversed(class_.mro()):
        if inherits_immutable(parent):
            parameters.extend(_dataclass_parameters(parent))
            class_.labels.add("dataclass")

    if not inherits_immutable(class_):
        return

    _logger = logger.info if class_.name == NEEDS_FIX else logger.debug

    _logger("Handling: %r", relative_path(class_))

    # Add current class parameters.
    parameters.extend(_dataclass_parameters(class_))
    if not parameters:
        _logger("|   Skipped (no params)")
        return

    _logger(
        f"|   Parameters(*, {', '.join(f'{p.name}: {p.annotation}' for p in parameters)})"
    )

    # Create `__init__` method with re-ordered parameters.
    init = griffe.Function(
        "__init__",
        lineno=0,
        endlineno=0,
        parent=class_,
        parameters=griffe.Parameters(
            griffe.Parameter(
                name="self",
                annotation=None,
                kind=griffe.ParameterKind.positional_or_keyword,
                default=None,
            ),
            *_reorder_parameters(parameters),
        ),
        returns="None",
    )
    class_.set_member("__init__", init)
    _logger(f"|   {init.signature(name=class_.name + '.__init__')}")


@cache
def inherits_immutable(cls: griffe.Class) -> bool:
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
def _dataclass_parameters(class_: griffe.Class) -> list[griffe.Parameter]:
    # Iterate on current attributes to find parameters.
    if class_.name == NEEDS_FIX and logger.isEnabledFor(logging.DEBUG):
        write_griffe(class_)

    parameters = []
    for member in class_.members.values():
        if member.is_attribute:
            member = cast("griffe.Attribute", member)

            if is_dataclass_field(member):
                parameters.append(
                    griffe.Parameter(
                        member.name,
                        annotation=member.annotation,
                        # All parameters marked as keyword-only.
                        kind=griffe.ParameterKind.keyword_only,
                        default=member.value,
                        docstring=member.docstring,
                    )
                )

    return parameters


def is_dataclass_field(member: griffe.Attribute) -> bool:
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
            isinstance(member.annotation, griffe.ExprName)
            and "ClassVar" in member.annotation.name
        )
    )


def _reorder_parameters(parameters: list[griffe.Parameter]) -> list[griffe.Parameter]:
    # De-duplicate, overwriting previous parameters.
    params_dict = {param.name: param for param in parameters}

    # Re-order, putting positional-only in front and keyword-only at the end.
    pos_only = []
    pos_kw = []
    kw_only = []
    for param in params_dict.values():
        if param.kind is griffe.ParameterKind.positional_only:
            pos_only.append(param)
        elif param.kind is griffe.ParameterKind.keyword_only:
            kw_only.append(param)
        else:
            pos_kw.append(param)
    return pos_only + pos_kw + kw_only


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
