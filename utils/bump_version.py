# mypy: ignore
from __future__ import annotations

import subprocess
import sys

out = subprocess.run(["git", "fetch", "upstream", "--tags"], check=False)
if out.returncode != 0:
    print(
        "Something went wrong with the release process, please check the Narwhals Wiki for "
        "at https://github.com/narwhals-dev/narwhals/wiki#release-process and try again."
    )
    print(out)
    sys.exit(1)
subprocess.run(["git", "reset", "--hard", "upstream/main"], check=False)

if (
    subprocess.run(
        ["git", "branch", "--show-current"], text=True, capture_output=True, check=False
    ).stdout.strip()
    != "bump-version"
):
    msg = "`bump_version.py` should be run from `bump-version` branch"
    raise RuntimeError(msg)

# Delete local tags, if present
try:
    # Get the list of all tags
    result = subprocess.run(
        ["git", "tag", "-l"], capture_output=True, text=True, check=True
    )
    tags = result.stdout.splitlines()  # Split the tags into a list by lines

    # Delete each tag using git tag -d
    for tag in tags:
        subprocess.run(["git", "tag", "-d", tag], check=True)
    print("All local tags have been deleted.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

subprocess.run(["git", "fetch", "upstream", "--tags"], check=False)
subprocess.run(["git", "fetch", "upstream", "--prune", "--tags"], check=False)

how = sys.argv[1]

subprocess.run(["uv", "version", "--bump", how], check=False)
version = subprocess.run(
    ["uv", "version", "--short"], capture_output=True, text=True, check=False
).stdout

subprocess.run(
    ["git", "commit", "-a", "-m", f"release: Bump version to {version}"], check=False
)
subprocess.run(["git", "tag", "-a", f"v{version}", "-m", f"v{version}"], check=False)
subprocess.run(["git", "push", "upstream", "HEAD", "--follow-tags"], check=False)
subprocess.run(
    ["git", "push", "upstream", "HEAD:stable", "-f", "--follow-tags"], check=False
)
