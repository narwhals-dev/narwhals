"""Bump the version, then commit, tag and push it. Pushing the tag publishes to PyPI.

See https://github.com/narwhals-dev/narwhals/wiki#release-process
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
BRANCH = "bump-version"


def run(*args: str) -> str:
    try:
        return sp.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except sp.CalledProcessError as exc:
        sys.exit(f"`{' '.join(args)}` failed:\n{exc.stderr.strip()}")


def bump_lock_entry(old: str, new: str) -> None:
    lock = Path("uv.lock")
    old_entry = f'name = "narwhals"\nversion = "{old}"\n'.encode()
    new_entry = f'name = "narwhals"\nversion = "{new}"\n'.encode()
    content = lock.read_bytes()
    if (count := content.count(old_entry)) != 1:
        sys.exit(f"Expected one narwhals {old} entry in uv.lock, found {count}.")
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
    if run(GIT, "status", "--porcelain", "--untracked-files=no"):
        sys.exit("Commit or stash your changes to tracked files first.")

    run(GIT, "fetch", REMOTE, "--prune", "--prune-tags", "--force")
    run(GIT, "switch", "-C", BRANCH, f"{REMOTE}/main")

    old_version = run(UV, "version", "--short")
    new_version = run(UV, "version", "--bump", bump, "--short", "--dry-run", "--frozen")
    if not new_version:
        sys.exit("`uv version --bump` returned an empty version.")
    tag = f"v{new_version}"
    if run(GIT, "tag", "--list", tag):
        sys.exit(f"Tag {tag} already exists on {REMOTE}.")

    try:
        # `uv lock` re-spells every fork's resolution markers and Renovate's
        # `uv lock --upgrade` reverts them, so edit only the narwhals entry.
        run(UV, "version", "--bump", bump, "--frozen")
        bump_lock_entry(old_version, new_version)
        run(UV, "lock", "--check")

        run(GIT, "add", "pyproject.toml", "uv.lock")
        staged = run(GIT, "diff", "--cached", "--numstat").splitlines()
        if staged != ["1\t1\tpyproject.toml", "1\t1\tuv.lock"]:
            sys.exit(
                "Expected a one-line change in pyproject.toml and uv.lock, got:\n"
                + "\n".join(staged)
            )
        run(GIT, "commit", "-m", f"release: Bump version to {new_version}")
        print(run(GIT, "show", "--stat", "HEAD"))

        answer = input(
            f"\nTag and push {tag} to {REMOTE}? This publishes to PyPI. [y/N] "
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
        sp.run(
            [GIT, "restore", "--staged", "--worktree", "pyproject.toml", "uv.lock"],
            check=False,
        )
        sp.run([GIT, "tag", "--delete", tag], capture_output=True, check=False)
        raise

    print(
        f"Pushed {BRANCH} and {tag}. Open a PR from {BRANCH} and squash-merge it.\n"
        "Do not amend, force-push or re-tag: the tag and the PyPI release use this commit."
    )


if __name__ == "__main__":
    main()
