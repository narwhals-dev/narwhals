"""Formatter for executing shell code."""

from __future__ import annotations

import subprocess
from typing import Any

from markdown_exec.formatters.base import ExecutionError, base_format
from markdown_exec.rendering import code_block


def _run_bash(
    code: str,
    returncode: int | None = None,
    session: str | None = None,  # noqa: ARG001
    id: str | None = None,  # noqa: A002,ARG001
    **extra: str,
) -> str:
    process = subprocess.run(
        ["bash", "-c", code],  # noqa: S603,S607
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if process.returncode != returncode:
        raise ExecutionError(code_block("sh", process.stdout, **extra), process.returncode)
    return process.stdout


def _format_bash(**kwargs: Any) -> str:
    return base_format(language="bash", run=_run_bash, **kwargs)
