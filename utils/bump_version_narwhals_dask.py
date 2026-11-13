"""Bump narwhals-dask's version, then commit, tag and push it. Pushing the tag publishes to PyPI.

See RELEASING.md
"""

from __future__ import annotations

import argparse
import os
import subprocess as sp
import sys
from pathlib import Path
from typing import Literal, get_args

BumpKind = Literal["patch", "minor", "major"]

GIT = "git"
UV = "uv"
REMOTE = "upstream"
BRANCH = "bump-narwhals-dask"
PACKAGE = "narwhals-dask"
PACKAGE_DIR = Path("packages") / PACKAGE
TAG_PREFIX = f"{PACKAGE}-v"


def run(*args: str) -> str:
    try:
        return sp.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except sp.CalledProcessError as exc:
        sys.exit(f"`{' '.join(args)}` failed:\n{exc.stderr.strip()}")


def bump_lock_entry(old: str, new: str) -> None:
    lock = PACKAGE_DIR / "uv.lock"
    old_entry = f'name = "{PACKAGE}"\nversion = "{old}"\n'.encode()
    new_entry = f'name = "{PACKAGE}"\nversion = "{new}"\n'.encode()
    content = lock.read_bytes()
    if (count := content.count(old_entry)) != 1:
        sys.exit(f"Expected one {PACKAGE} {old} entry in {lock}, found {count}.")
    lock.write_bytes(content.replace(old_entry, new_entry))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "bump",
        choices=get_args(BumpKind),
        help="Part of the version to bump (major.minor.patch).",
    )
    bump: BumpKind = parser.parse_args().bump

    os.chdir(Path(__file__).resolve().parents[1])
    # NOTE: narwhals-dask has its own environment. Never let `uv` resolve it to an
    # active virtualenv or a global `UV_PROJECT_ENVIRONMENT`.
    os.environ["UV_PROJECT_ENVIRONMENT"] = str((PACKAGE_DIR / ".venv").resolve())
    os.environ.pop("VIRTUAL_ENV", None)
    uv_version = (UV, "version", "--directory", str(PACKAGE_DIR))

    if run(GIT, "status", "--porcelain", "--untracked-files=no"):
        sys.exit("Commit or stash your changes to tracked files first.")

    run(GIT, "fetch", REMOTE, "--prune", "--prune-tags", "--force")
    run(GIT, "switch", "-C", BRANCH, f"{REMOTE}/main")

    old_version = run(*uv_version, "--short")
    new_version = run(*uv_version, "--bump", bump, "--short", "--dry-run", "--frozen")
    if not new_version:
        sys.exit("`uv version --bump` returned an empty version.")
    tag = f"{TAG_PREFIX}{new_version}"
    if run(GIT, "tag", "--list", tag):
        sys.exit(f"Tag {tag} already exists on {REMOTE}.")

    pyproject, lock = str(PACKAGE_DIR / "pyproject.toml"), str(PACKAGE_DIR / "uv.lock")
    try:
        run(*uv_version, "--bump", bump, "--frozen")
        bump_lock_entry(old_version, new_version)
        run(UV, "lock", "--check", "--directory", str(PACKAGE_DIR))

        run(GIT, "add", pyproject, lock)
        staged = run(GIT, "diff", "--cached", "--numstat").splitlines()
        if staged != [f"1\t1\t{pyproject}", f"1\t1\t{lock}"]:
            sys.exit(
                f"Expected a one-line change in {pyproject} and {lock}, got:\n"
                + "\n".join(staged)
            )
        run(GIT, "commit", "-m", f"release: Bump {PACKAGE} version to {new_version}")
        print(run(GIT, "show", "--stat", "HEAD"))

        answer = input(
            f"\nTag and push {tag} to {REMOTE}? This publishes {PACKAGE} to PyPI. [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            print(f"Nothing pushed. The next run resets {BRANCH}.")
            return

        run(GIT, "tag", "-a", tag, "-m", tag)
        run(
            GIT,
            "push",
            "--atomic",
            REMOTE,
            f"HEAD:refs/heads/{BRANCH}",
            f"refs/tags/{tag}",
        )
    except BaseException:
        # The tree was clean before the bump, so this only undoes the script's changes.
        sp.run([GIT, "restore", "--staged", "--worktree", pyproject, lock], check=False)
        sp.run([GIT, "tag", "--delete", tag], capture_output=True, check=False)
        raise

    print(
        f"Pushed {BRANCH} and {tag}. Open a PR from {BRANCH} and squash-merge it.\n"
        "Do not amend, force-push or re-tag: the tag and the PyPI release use this commit."
    )


if __name__ == "__main__":
    main()
