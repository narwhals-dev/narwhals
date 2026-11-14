# Releasing

Pushing a version tag publishes to PyPI. Each repository needs an `upstream` remote
pointing at its narwhals-dev repo over SSH (e.g. `git@github.com:narwhals-dev/narwhals.git`).

## Narwhals

1. Check that the [cuDF notebook](https://www.kaggle.com/code/marcogorelli/narwhals-pre-release-test-cudf)
   and `pytest --constructors=pyspark` pass. If not, report it on Discord before releasing.
2. In the draft "Narwhals Unreleased" [release](https://github.com/narwhals-dev/narwhals/releases),
   fix any missing PR labels, then copy the notes somewhere.
3. With a clean working tree, run:

   ```terminal
   python utils/bump_version.py minor  # `patch` for an emergency mid-week release
   ```

   Check the version it prints, and answer `y` to tag and push.
4. Open the PR, labelled `release`:

   ```terminal
   gh pr create --repo narwhals-dev/narwhals --base main --head bump-version --label release --fill
   ```

5. Once the release for the new tag appears (e.g. `v1.30.0`, with empty notes), edit it: paste
   the notes, set the title to e.g. `Narwhals v1.30.0`, and press "Update release notes".
   Don't publish "Narwhals Unreleased".
6. Squash-merge the PR. A `bypassed rule violations` message is expected.

Don't amend or force-push `bump-version` after step 3: the tag points at it. If the script
fails, it rolls back and explains why; fix the cause and rerun.

## narwhals-dask

Lives in [`packages/narwhals-dask`](packages/narwhals-dask). Release it right after the
Narwhals release it needs.

1. If it needs a newer Narwhals, raise the `narwhals>=` floor in
   `packages/narwhals-dask/pyproject.toml`, run `uv lock` in that directory, and merge to `main`.
2. With a clean working tree, run:

   ```terminal
   python utils/bump_version.py minor --package narwhals-dask
   ```

   Check the version it prints, and answer `y` to tag and push. If the `pypi-narwhals-dask`
   environment requires a reviewer, approve the upload in the
   [workflow run](https://github.com/narwhals-dev/narwhals/actions/workflows/publish_narwhals_dask.yml).
3. Open a PR from `bump-narwhals-dask` to `main` and squash-merge it.

## narwhals-daft

From a clone of [narwhals-daft](https://github.com/narwhals-dev/narwhals-daft):

```terminal
git switch -C bump-version
python utils/bump_version.py minor
```

This resets `bump-version` to `upstream/main`, bumps the version, then pushes the branch, the
`v0.7.0` tag and `stable`. Open a PR from `bump-version` to `main` and merge it.
