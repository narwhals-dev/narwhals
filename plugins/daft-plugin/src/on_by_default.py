"""Our first example plugin."""

from __future__ import annotations


class ExampleOne:
    """First Example Plugin."""

    def __init__(self):
        pass

    def run(self) -> None:
        """Do nothing."""
        print('ExampleOne just ran!')  
