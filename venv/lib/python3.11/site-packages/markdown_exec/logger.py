"""This module contains logging utilities.

We provide the [`patch_loggers`][markdown_exec.logger.patch_loggers]
function so dependant libraries can patch loggers as they see fit.

For example, to fit in the MkDocs logging configuration
and prefix each log message with the module name:

```python
import logging
from markdown_exec.logger import patch_loggers


class LoggerAdapter(logging.LoggerAdapter):
    def __init__(self, prefix, logger):
        super().__init__(logger, {})
        self.prefix = prefix

    def process(self, msg, kwargs):
        return f"{self.prefix}: {msg}", kwargs


def get_logger(name):
    logger = logging.getLogger(f"mkdocs.plugins.{name}")
    return LoggerAdapter(name.split(".", 1)[0], logger)


patch_loggers(get_logger)
```
"""

from __future__ import annotations

import logging
from typing import Any, Callable, ClassVar


class _Logger:
    _default_logger: Any = logging.getLogger
    _instances: ClassVar[dict[str, _Logger]] = {}

    def __init__(self, name: str) -> None:
        # default logger that can be patched by third-party
        self._logger = self.__class__._default_logger(name)
        # register instance
        self._instances[name] = self

    def __getattr__(self, name: str) -> Any:
        # forward everything to the logger
        return getattr(self._logger, name)

    @classmethod
    def _patch_loggers(cls, get_logger_func: Callable) -> None:
        # patch current instances
        for name, instance in cls._instances.items():
            instance._logger = get_logger_func(name)
        # future instances will be patched as well
        cls._default_logger = get_logger_func


def get_logger(name: str) -> _Logger:
    """Create and return a new logger instance.

    Parameters:
        name: The logger name.

    Returns:
        The logger.
    """
    return _Logger(name)


def patch_loggers(get_logger_func: Callable[[str], Any]) -> None:
    """Patch loggers.

    Parameters:
        get_logger_func: A function accepting a name as parameter and returning a logger.
    """
    _Logger._patch_loggers(get_logger_func)
