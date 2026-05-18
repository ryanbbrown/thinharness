# Releasing

ThinHarness uses manual versions in `pyproject.toml` and publishes from GitHub tag workflows.

## Release PR

1. Update `version` in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Run:

   ```bash
   uv build
   uv run --with twine twine check dist/*
   uv run ruff check .
   uv run pytest
   ```

4. Open a PR titled `chore: release X.Y.Z`.
5. Merge after CI is green.

## Publish

1. Tag the merged commit:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

2. The `release.yml` workflow builds the distributions and publishes them to PyPI with Trusted Publishing.
3. Create a GitHub Release from the same tag and paste the changelog notes.

PyPI versions are immutable. If a release is broken after upload, fix it in a new patch version.
