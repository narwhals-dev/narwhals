"""Our first example plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator


class ExampleOne:
    """First Example Plugin."""

    def __init__(self, tree: Any) -> None:
        self.tree = tree

    def run(self) -> Generator[Any, Any, None]:
        """Do nothing."""
        yield from []
