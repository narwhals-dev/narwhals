# mypy: ignore
# ruff: noqa
import re
import subprocess
import sys

out = subprocess.run(["git", "fetch", "upstream", "--tags"])
if out.returncode != 0:
    print(
        "Something went wrong with the release process, please check the Narwhals Wiki for "
        "at https://github.com/narwhals-dev/narwhals/wiki#release-process and try again."
    )
    print(out)
    sys.exit(1)
subprocess.run(["git", "reset", "--hard", "upstream/main"])

if (
    subprocess.run(
        ["git", "branch", "--show-current"], text=True, capture_output=True
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

subprocess.run(["git", "fetch", "upstream", "--tags"])
subprocess.run(["git", "fetch", "upstream", "--prune", "--tags"])

how = sys.argv[1]

subprocess.run(["uv", "version", "--bump", how]) 
version = subprocess.run(["uv", "version", "--short"], capture_output=True).stdout.decode("utf-8")

subprocess.run(["git", "commit", "-a", "-m", f"release: Bump version to {version}"])
subprocess.run(["git", "tag", "-a", f"v{version}", "-m", f"v{version}"])
subprocess.run(["git", "push", "upstream", "HEAD", "--follow-tags"])
subprocess.run(["git", "push", "upstream", "HEAD:stable", "-f", "--follow-tags"])
