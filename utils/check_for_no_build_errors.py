"""
Check that `output.txt` doesn't contain the string
"exited with errors". If it does, exit with status 1.

This is just used in CI.
"""

import sys

with open("output.txt") as fd:
    content = fd.read()

if "exited with errors" in content:
    sys.exit(1)
sys.exit(0)
